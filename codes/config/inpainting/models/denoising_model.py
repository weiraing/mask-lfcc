import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.model = DataParallel(self.model)

        # MTSCE (Multi-scale Texture and Structure Collaborative Enhancement)
        # is optional and driven by network_G.setting.mtsce in the config file
        mtsce_opt = opt["network_G"]["setting"].get("mtsce", None)
        self.mtsce_enabled = bool(mtsce_opt.get("enabled", False)) if mtsce_opt else False
        self.freeze_base = False

        # print network
        # self.print_network()
        self.load()

        if self.mtsce_enabled:
            self._log_mtsce_summary()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            finetune_opt = train_opt.get("finetune", None) or {}
            lr_G = train_opt["lr_G"]
            lr_mtsce = finetune_opt.get("lr_mtsce", None)

            # stage-1 warm-up: keep the pretrained base network frozen and only
            # optimise the newly initialised MTSCE parameters
            if self.mtsce_enabled and bool(finetune_opt.get("freeze_base", False)):
                n_frozen = 0
                for name, param in self.model.named_parameters():
                    if not self._is_mtsce_param(name) and param.requires_grad:
                        param.requires_grad_(False)
                        n_frozen += 1
                self.freeze_base = True
                if self.rank <= 0:
                    logger.info(
                        "MTSCE stage-1 warm-up: %d base parameter tensor(s) frozen; "
                        "only MTSCE parameters are trainable.",
                        n_frozen,
                    )

            base_params, mtsce_params = [], []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    if self._is_mtsce_param(k):
                        mtsce_params.append(v)
                    else:
                        base_params.append(v)
                else:
                    if self.rank <= 0 and not self.freeze_base:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            param_groups = []
            if base_params:
                param_groups.append({"params": base_params, "lr": lr_G})
            if mtsce_params:
                param_groups.append(
                    {
                        "params": mtsce_params,
                        "lr": float(lr_mtsce)
                        if (self.mtsce_enabled and lr_mtsce)
                        else lr_G,
                    }
                )
            if not param_groups:
                raise RuntimeError(
                    "no trainable parameter found; check the freeze settings "
                    "(train.finetune.freeze_base) and the network config."
                )
            if self.rank <= 0 and self.mtsce_enabled and lr_mtsce:
                logger.info(
                    "Parameter groups: base lr=%g, MTSCE lr=%g",
                    lr_G,
                    float(lr_mtsce),
                )

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    param_groups,
                    lr=lr_G,
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=lr_G,
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    param_groups, 
                    lr=lr_G,
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, state, LQ, GT=None, mask=None):
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT
        # degradation mask (1 = degraded/missing region), for mask-conditioned
        # networks and MGDC sampling; None for plain models
        self.mask = mask.to(self.device) if mask is not None else None

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)
        sde.set_mask(self.mask)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)

        # Get noise and score
        noise = sde.noise_fn(self.state, timesteps.squeeze())
        score = sde.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def test(self, sde=None, save_states=False):
        sde.set_mu(self.condition)
        sde.set_mask(self.mask)

        self.model.eval()
        with torch.no_grad():
            self.output = sde.reverse_sde(self.state, save_states=save_states)

        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    @staticmethod
    def _is_mtsce_param(name):
        """True for parameters that belong to an MTSCE module."""
        return "mtsce" in name.split(".")

    def _log_mtsce_summary(self):
        net = self.model
        if isinstance(net, (nn.DataParallel, DistributedDataParallel)):
            net = net.module
        counts = net.parameter_count()
        logger.info(
            "MTSCE enabled | insertion_points=%s | mtsce params=%s | "
            "base params=%s | total=%s (+%.2f%%)",
            getattr(net, "mtsce_points", []),
            "{:,}".format(counts["mtsce"]),
            "{:,}".format(counts["base"]),
            "{:,}".format(counts["total"]),
            100.0 * counts["mtsce"] / max(1, counts["base"]),
        )

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            if not os.path.exists(load_path_G):
                raise FileNotFoundError(
                    "path.pretrain_model_G does not exist: {}\n"
                    "  check the config file; for the MTSCE stage-2 config this "
                    "must be the checkpoint produced by stage 1.".format(load_path_G)
                )
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            strict = self.opt["path"]["strict_load"]
            if strict is None:
                # an unspecified flag must not silently degrade into a loose
                # load: structural mismatches have to be reported
                strict = True
            if strict:
                # MTSCE parameters are new: they may be absent from an existing
                # checkpoint (they keep their zero-init) and, when the module is
                # disabled for an ablation, checkpointed MTSCE tensors are
                # reported and ignored. Everything else must match exactly.
                self.load_network(
                    load_path_G,
                    self.model,
                    True,
                    allow_missing_prefixes=("mtsce.",) if self.mtsce_enabled else None,
                    allow_unexpected_prefixes=("mtsce.",),
                )
            else:
                self.load_network(load_path_G, self.model, False)

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
