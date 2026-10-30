"""Correction-training wrapper for the Refusion IR-SDE inpainting model.

This module implements the second contribution of the project: a *corrector*
network trained jointly with a frozen denoiser, following "Correction Training
for Diffusion Models" (https://github.com/sunovivid/correction-training,
arXiv 2610.14136) and adapted to a pixel-space IR-SDE with an
epsilon-predicting NAFNet denoiser.

Data flow per training step
---------------------------
1. the frozen denoiser predicts ``eps_b`` for the noisy state ``x_t``;
2. ``x0_b`` is derived with the exact IR-SDE map;
3. the corrector refines ``eps_b`` (``K`` iterations, stop-gradient through
   ``eps_b``) into ``eps_c``;
4. ``eps_c`` is pushed through the *unchanged* reverse-step formula, and the
   result is matched against the same optimal ``x_{t-1}`` target the denoiser
   was trained against.

Nothing in the diffusion formulation, the noise schedule, the sampler or the
supervised objective is modified.  Only an extra network is inserted between the
denoiser's prediction and the score.
"""

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.modules as M
from models.modules.correction import (
    CorrectionGate,
    DiffusionCorrector,
    count_parameters,
    init_corrector_from_base,
    normalize_correction_config,
    refine,
)
from models.modules.lowfreq import LowFreqCorrectorNet
from models.modules.loss import MatchingLoss
from models.optimizer import Lion

from .base_model import BaseModel

logger = logging.getLogger("base")


def _build_net(opt, extra_setting=None):
    """Instantiate ``network_G`` with optional extra constructor arguments."""
    which_model = opt["network_G"]["which_model_G"]
    setting = dict(opt["network_G"]["setting"])
    if extra_setting:
        setting.update(extra_setting)
    return getattr(M, which_model)(**setting)


class CorrectionModel(BaseModel):
    """Frozen denoiser + trainable corrector."""

    def __init__(self, opt):
        super(CorrectionModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt["train"]

        cfg = normalize_correction_config(opt.get("correction"))
        self.corr_cfg = cfg
        self.spec = cfg["spec"]
        self.correction_enabled = bool(cfg["enabled"])

        img_channel = int(dict(opt["network_G"]["setting"]).get("img_channel", 3))
        self.img_channel = img_channel
        proposal_ch = img_channel if self.spec.proposal_injection else 0
        self.proposal_ch = proposal_ch

        # ---- frozen denoiser ------------------------------------------------
        base_net = _build_net(opt).to(self.device)
        self.model = self._wrap(base_net)
        self.load()  # path.pretrain_model_G

        # ---- corrector ------------------------------------------------------
        self.corrector = None
        self.correction_calls = 0
        self.arch = cfg["arch"]
        if self.correction_enabled:
            if self.arch == "lowfreq":
                corr_net = LowFreqCorrectorNet(
                    in_channels=img_channel,
                    hidden_channels=cfg["lowfreq_hidden"],
                    num_blocks=cfg["lowfreq_blocks"],
                    num_timesteps=int(opt["sde"]["T"]),
                    cells=cfg["lowfreq_cells"],
                    iters=cfg["lowfreq_iters"],
                    coarse_iters=cfg["lowfreq_coarse_iters"],
                    alpha_init=cfg["lowfreq_alpha_init"],
                    alpha_final_init=cfg["lowfreq_alpha_final_init"],
                    alpha_range=cfg["lowfreq_alpha_range"],
                    alpha_final_range=cfg["lowfreq_final_mod_range"],
                    trainable=cfg["trainable"],
                ).to(self.device)
                report = {
                    "copied": 0,
                    "new": ["harmonic fill (no parameters)", "alpha gate"],
                    "zero_init_head": "out (alpha starts at %.3g)" % cfg["lowfreq_alpha_init"],
                }
            else:
                corr_net = _build_net(opt, {"proposal_ch": proposal_ch}).to(self.device)
                report = init_corrector_from_base(corr_net, base_net, self.spec)

            gate = None
            if self.arch == "copy" and self.spec.combine == "gated":
                gate = CorrectionGate(
                    channels=img_channel,
                    shape=self.spec.gate_shape,
                    init=self.spec.gate_init,
                    activation=self.spec.gate_activation,
                ).to(self.device)

            self.corrector = DiffusionCorrector(
                corr_net,
                self.spec,
                gate=gate,
                apply_t_range=cfg["apply_t_range"],
                steps=cfg["steps"],
            ).to(self.device)

            if self.rank <= 0:
                logger.info(
                    "Correction module built | arch=%s | copied %d tensor(s) from "
                    "the denoiser | new: %s | head=%s",
                    self.arch,
                    report["copied"],
                    report["new"] or "none",
                    report["zero_init_head"] or "kept from denoiser",
                )

        # an existing corrector checkpoint always wins (test / resume)
        self.load_corrector()

        # ---- freezing -------------------------------------------------------
        self.freeze_base = bool(cfg["freeze_base"]) and self.is_train
        if self.is_train and self.freeze_base:
            for param in self.model.parameters():
                param.requires_grad_(False)
            if self.rank <= 0:
                logger.info("Correction training: denoiser frozen, only the corrector trains.")

        if self.corrector is not None and not self.is_train:
            self.corrector.eval()

        if self.is_train:
            self.model.train()
            if self.freeze_base:
                # keep frozen BatchNorm/statistics out of the picture and make
                # the frozen forward pass cheap
                self.model.eval()

            self.loss_fn = MatchingLoss(
                opt["train"]["loss_type"], opt["train"]["is_weighted"]
            ).to(self.device)
            self.weight = opt["train"]["weight"]

            if self.corrector is None:
                raise RuntimeError(
                    "model=correction but correction.enabled is false; either "
                    "enable it or use model=denoising for the baseline."
                )

            lr_corr = cfg["lr_correction"]
            lr_corr = float(lr_corr) if lr_corr else float(train_opt["lr_G"])
            lr_backbone = lr_corr * float(cfg["lr_backbone_scale"])
            wd_corr = cfg["weight_decay_correction"] or 0.0

            # two groups: the head/proposal (new, zero-initialised, must move)
            # and the copied backbone (pretrained, only needs a nudge).
            # The lowfreq gate is entirely new, so it all goes in the head group.
            new_names = ("proposal_proj.", "ending.", "gate.")
            new_params, backbone_params = [], []
            for name, p in self.corrector.named_parameters():
                if not p.requires_grad:
                    continue
                if self.arch == "lowfreq" or name.startswith(new_names):
                    new_params.append(p)
                else:
                    backbone_params.append(p)

            param_groups = []
            if backbone_params:
                param_groups.append(
                    {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd_corr}
                )
            if new_params:
                param_groups.append(
                    {"params": new_params, "lr": lr_corr, "weight_decay": wd_corr}
                )
            if not param_groups:
                # trainable=False: the module is the training-free fixed blend,
                # so there is nothing to optimise. Only valid for evaluation.
                if self.is_train:
                    raise RuntimeError(
                        "correction.trainable is false, so the corrector has no "
                        "parameters to optimise; use a test config "
                        "(correction.trainable: false + model: correction) for the "
                        "training-free ablation."
                    )
            if not self.freeze_base:
                base_params = [p for p in self.model.parameters() if p.requires_grad]
                if base_params:
                    param_groups.insert(
                        0,
                        {
                            "params": base_params,
                            "lr": float(train_opt["lr_G"]),
                            "weight_decay": train_opt["weight_decay_G"]
                            if train_opt["weight_decay_G"]
                            else 0.0,
                        },
                    )

            betas = (train_opt["beta1"], train_opt["beta2"])
            opt_name = cfg["optimizer"] or train_opt["optimizer"]
            if opt_name == "Adam":
                self.optimizer = torch.optim.Adam(param_groups, lr=lr_corr, betas=betas)
            elif opt_name == "AdamW":
                self.optimizer = torch.optim.AdamW(param_groups, lr=lr_corr, betas=betas)
            elif opt_name == "Lion":
                self.optimizer = Lion(param_groups, lr=lr_corr, betas=betas)
            else:
                raise NotImplementedError("optimizer [{}] not implemented".format(opt_name))
            self.optimizers.append(self.optimizer)

            if train_opt["lr_scheme"] == "MultiStepLR":
                for opt_i in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            opt_i,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for opt_i in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt_i, T_max=train_opt["niter"], eta_min=train_opt["eta_min"]
                        )
                    )
            else:
                raise NotImplementedError("unsupported lr_scheme")

            self.log_dict = OrderedDict()

            if self.rank <= 0:
                base_counts = count_parameters(self.model)
                corr_counts = count_parameters(self.corrector)
                logger.info(
                    "Correction setup | arch=%s | denoiser %s params (frozen=%s) | "
                    "corrector %s params | input=%s output=%s combine=%s | "
                    "train_steps=%d sample_steps=%d | apply_t=%s train_t=%s | "
                    "lr: head=%g backbone=%g",
                    self.arch,
                    "{:,}".format(base_counts["total"]),
                    self.freeze_base,
                    "{:,}".format(corr_counts["total"]),
                    self.spec.input,
                    self.spec.output,
                    self.spec.combine,
                    cfg["train_steps"],
                    cfg["steps"],
                    list(cfg["apply_t_range"]),
                    list(cfg["train_t_range"]),
                    lr_corr,
                    lr_backbone,
                )

    # -- plumbing ---------------------------------------------------------
    def _wrap(self, net):
        if self.opt["dist"]:
            return DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])
        return DataParallel(net)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is None:
            raise ValueError(
                "model=correction needs a trained denoiser: set "
                "path.pretrain_model_G in the config. Training a corrector on "
                "top of a randomly initialised denoiser is meaningless."
            )
        if not os.path.exists(load_path_G):
            raise FileNotFoundError(
                "path.pretrain_model_G does not exist: {}".format(load_path_G)
            )
        if self.rank <= 0:
            logger.info("Loading frozen denoiser from [{:s}] ...".format(load_path_G))
        self.load_network(load_path_G, self.model, strict=True)

    def load_corrector(self):
        if self.corrector is None:
            return
        if getattr(self.corrector.net, "trainable", True) is False:
            # correction.trainable: false -> the training-free fixed blend,
            # so there is nothing to load and nothing to save
            if self.rank <= 0:
                logger.info(
                    "Corrector is the training-free fixed blend (trainable: false); "
                    "no checkpoint needed."
                )
            return
        path = self.opt["path"].get("pretrain_model_C", None)
        if not path:
            # a trained pair is stored side by side (<iter>_G.pth / <iter>_C.pth),
            # so an explicit path is only needed when the two differ
            g_path = self.opt["path"].get("pretrain_model_G", None)
            if g_path:
                candidate = os.path.join(
                    os.path.dirname(g_path),
                    os.path.basename(g_path).replace("_G.pth", "_C.pth"),
                )
                if os.path.isfile(candidate):
                    path = candidate
        if not path:
            if self.is_train:
                if self.rank <= 0:
                    logger.info(
                        "No corrector checkpoint: the corrector starts from the "
                        "identity (bit-identical to the frozen denoiser)."
                    )
                return
            # At evaluation time a missing corrector would silently reproduce the
            # baseline, which looks exactly like "the module did nothing". Refuse
            # instead; use correction.enabled: false for the baseline arm.
            raise FileNotFoundError(
                "model=correction with correction.enabled=true needs a trained "
                "corrector, but neither path.pretrain_model_C nor a paired "
                "<iter>_C.pth next to path.pretrain_model_G was found "
                "(denoiser: {}). Set path.pretrain_model_C, or set "
                "correction.enabled: false to evaluate the baseline through the "
                "same code path.".format(self.opt["path"].get("pretrain_model_G"))
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                "path.pretrain_model_C does not exist: {}".format(path)
            )
        self.load_network(path, self.corrector, strict=True)
        if self.rank <= 0:
            logger.info("Corrector restored from [{:s}]".format(path))

    # -- training ---------------------------------------------------------
    def feed_data(self, state, LQ, GT=None, mask=None):
        self.state = state.to(self.device)  # x_t
        self.condition = LQ.to(self.device)  # mu
        if GT is not None:
            self.state_0 = GT.to(self.device)  # x_0
        self.mask = mask.to(self.device) if mask is not None else None
        # the harmonic fill only depends on (mu, mask); drop the cached field so
        # a new image never reuses the previous one's (data_ptr may be recycled)
        if self.corrector is not None and hasattr(self.corrector.net, "clear_cache"):
            self.corrector.net.clear_cache()

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)
        sde.set_mask(self.mask)

        self.optimizer.zero_grad(set_to_none=True)
        timesteps = timesteps.to(self.device)
        # reshape(-1), not squeeze(): with a batch of one, squeeze() would give a
        # 0-dim tensor and the sinusoidal embedding indexes with x[:, None]
        t = timesteps.reshape(-1) if timesteps.dim() > 1 else timesteps

        # Training window. Outside [1, T] the corrector is not applied at
        # sampling, and the step-size weighting sigma^2*dt/sigma_bar makes the
        # loss at high t dominate, so training on a narrower window is both
        # cheaper and better aligned with inference.
        t_lo, t_hi = self.corr_cfg["train_t_range"]
        if (t_lo, t_hi) != (1, sde.T) and getattr(self, "state_0", None) is not None:
            # For IR-SDE x_t = mu + (x0-mu)*A(t) + sigma_bar(t)*eps, so the
            # noise realisation is recoverable exactly and the redrawn state is
            # still a valid marginal sample of the same clean image.
            sig = sde.sigma_bar(timesteps)
            eps_true = (self.state - sde.mu_bar(self.state_0, timesteps)) / sig.clamp_min(1e-8)
            timesteps = torch.randint(
                t_lo, t_hi + 1, timesteps.shape, device=self.device
            ).long()
            self.state = sde.mu_bar(self.state_0, timesteps) + sde.sigma_bar(timesteps) * eps_true
            t = timesteps.reshape(-1)

        # With the denoiser frozen this forward pass builds no autograd graph at
        # all (all of its parameters have requires_grad=False), so it costs one
        # extra forward only.  With freeze_base=False it is the normal joint
        # training path.
        eps_b = sde.noise_fn(self.state, t)
        score_b = sde.get_score_from_noise(eps_b, timesteps)
        xt1_b = sde.reverse_sde_step_mean(self.state, score_b, timesteps)
        xt1_opt = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        loss_b = self.weight * self.loss_fn(xt1_b, xt1_opt)

        eps_c = refine(
            self.corrector,
            sde,
            self.state,
            eps_b,
            timesteps,
            steps=self.corr_cfg["train_steps"],
            detach_input=self.spec.detach_input,
            detach_skip=self.spec.detach_skip,
        )
        score_c = sde.get_score_from_noise(eps_c, timesteps)
        xt1_c = sde.reverse_sde_step_mean(self.state, score_c, timesteps)
        loss_c = self.weight * self.loss_fn(xt1_c, xt1_opt)

        loss = self.corr_cfg["loss_weight"] * loss_c
        if not self.freeze_base and self.corr_cfg["base_loss_weight"] > 0:
            loss = loss + self.corr_cfg["base_loss_weight"] * loss_b

        # ---- low-t branch for the post-sampling composite -------------------
        # ``finalize`` only ever runs on a finished sample, so supervising it at
        # a uniformly drawn t would teach it the wrong thing (at t=T the clean
        # estimate is noise and any alpha looks good).  This branch draws its own
        # low t, where x0_hat really is the image the composite will see.
        net = self.corrector.net
        if self.corr_cfg["final_branch"] and hasattr(net, "finalize"):
            lo, hi = self.corr_cfg["final_t_range"]
            t_fin = torch.randint(
                lo, hi + 1, timesteps.shape, device=self.device
            ).long()
            x_fin = sde.mu_bar(self.state_0, t_fin) + sde.sigma_bar(t_fin) * torch.randn_like(
                self.state_0
            )
            with torch.no_grad():
                eps_fin = sde.noise_fn(x_fin, t_fin.reshape(-1))
            x0_fin = sde.get_init_state_from_noise(x_fin, eps_fin, t_fin)
            x0_fin = net.finalize(x0_fin, self.condition, self.mask)
            loss_fin = self.loss_fn(x0_fin, self.state_0)
            loss = loss + self.corr_cfg["final_loss_weight"] * loss_fin
            self.log_dict["loss_final"] = float(loss_fin.item())
            # the *effective* strength inside the hole, i.e. the constant plus
            # the learned modulation.  This is the number that has to stay near
            # 0.2: the alpha sweep peaks there and 0.3 already loses 0.12 dB.
            if getattr(net, "_last_alpha", None) is not None:
                self.log_dict["alpha_final"] = float(net._last_alpha)

        self.log_dict["loss"] = float(loss_c.item())
        self.log_dict["loss_base"] = float(loss_b.item())
        # This, not x0_gain, is the number to watch: x0 MSE weights the noise
        # error by (sigma_bar*A)^2, which at t=T is ~1e3 times the weight the
        # sampler step itself uses, so it is dominated by the highest t.
        self.log_dict["loss_gain"] = float(
            1.0 - loss_c.item() / max(loss_b.item(), 1e-12)
        )

        if not torch.isfinite(loss):
            # never let one bad batch poison the corrector: skip the update
            self.optimizer.zero_grad(set_to_none=True)
            self.log_dict["grad_norm"] = float("nan")
            if self.rank <= 0:
                logger.warning("non-finite correction loss at iter %d; step skipped", step)
            return

        loss.backward()

        clip = self.corr_cfg["grad_clip"]
        params = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if clip and clip > 0 and params:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(clip))
            self.log_dict["grad_norm"] = float(grad_norm)
        else:
            self.log_dict["grad_norm"] = float("nan")

        self.optimizer.step()

        # ---- diagnostics ---------------------------------------------------
        # Paired comparison on the *same* x_t and the same t, so this ratio is
        # directly interpretable: > 0 means the correction moved the clean
        # estimate closer to the ground truth on this batch.
        with torch.no_grad():
            x0_b = sde.get_init_state_from_noise(self.state, eps_b, timesteps)
            x0_c = sde.get_init_state_from_noise(self.state, eps_c.detach(), timesteps)
            err_b = (x0_b - self.state_0).pow(2).mean()
            err_c = (x0_c - self.state_0).pow(2).mean()
            self.log_dict["x0_mse_base"] = float(err_b)
            self.log_dict["x0_mse_corr"] = float(err_c)
            self.log_dict["x0_gain"] = float(1.0 - err_c / err_b.clamp_min(1e-12))
            self.log_dict["delta_rms"] = float((eps_c.detach() - eps_b).pow(2).mean().sqrt())
            if self.corrector.gate is not None:
                self.log_dict["gate"] = float(
                    self.corrector.gate.value().abs().mean()
                )

    # -- inference --------------------------------------------------------
    def test(self, sde=None, save_states=False):
        sde.set_mu(self.condition)
        sde.set_mask(self.mask)

        self.model.eval()
        if self.corrector is not None:
            self.corrector.eval()
            sde.set_corrector(self.corrector)
        with torch.no_grad():
            self.output = sde.reverse_sde(self.state, save_states=save_states)
        if self.corrector is not None:
            self.correction_calls += sde.correction_calls
            sde.set_corrector(None)

        if self.is_train:
            if self.freeze_base:
                self.model.eval()
            else:
                self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT and getattr(self, "state_0", None) is not None:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    # -- persistence ------------------------------------------------------
    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        if self.corrector is not None:
            self.save_network(self.corrector, "C", iter_label)
