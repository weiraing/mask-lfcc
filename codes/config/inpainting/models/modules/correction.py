"""Correction training for the Refusion IR-SDE inpainting denoiser.

Adapted from "Correction Training for Diffusion Models"
(https://github.com/sunovivid/correction-training, arXiv 2610.14136) to a
pixel-space IR-SDE whose denoiser predicts the noise (epsilon).

Original recipe
---------------
* keep the denoiser frozen;
* add a second network (the *corrector*), initialised from the denoiser weights;
* both networks are supervised by the *same* clean target, but the gradient of
  the correction loss stops at the denoiser's prediction (stop-gradient), so the
  corrector learns the denoiser's residual error instead of perturbing it;
* at sampling the corrector is reused inside every reverse step, K times.

Adaptation for IR-SDE
---------------------
IR-SDE has no velocity parameterisation.  The two equivalent prediction spaces
are the noise ``eps`` and the clean estimate ``x0``, related by the exact affine
maps ``get_init_state_from_noise`` / ``get_noise_from_init_state``.  They are
exact inverses, so a correction applied in either space can be mapped back into
the score without touching the SDE update, the noise schedule or the sampler.

Why this is not just "another denoiser"
---------------------------------------
With ``combine: residual`` the corrector's output layer is zero-initialised, so
at step 0 the corrected prediction is *bit-identical* to the frozen denoiser's
prediction.  Training therefore starts exactly at the baseline and can only move
away from it by reducing the same supervised objective -- the correction is a
learned residual on top of a fixed, already-good denoiser.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger("base")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # master switch; false keeps the sampler on the original code path
    "enabled": True,
    # which correction network to build:
    #   "copy"    -> a full copy of the denoiser (the reference recipe)
    #   "lowfreq" -> a small boundary-consistency gate (see modules/lowfreq.py)
    "arch": "copy",
    # -- lowfreq arch ------------------------------------------------------
    "lowfreq_hidden": 48,
    "lowfreq_blocks": 4,
    # side of the coarse grid used for the harmonic fill; fixing the *grid*
    # keeps the prior's physical scale the same for 128 px crops and 256 px
    # validation images
    "lowfreq_cells": 64,
    "lowfreq_iters": 160,
    "lowfreq_coarse_iters": 200,
    # alpha_init = 0 makes the module the exact identity at step 0
    "lowfreq_alpha_init": 0.0,
    "lowfreq_alpha_range": 0.5,
    # strength of the one-shot blend of the finished sample towards the
    # boundary field (the operation whose gain was measured on the val split)
    "lowfreq_alpha_final_init": 0.2,
    # amplitude of the per-pixel modulation around that constant.  The alpha
    # sweep peaks sharply at 0.2 (0.3 already loses 0.12 dB inside the hole),
    # so a large modulation can cancel the entire gain -- 0.05 keeps the
    # effective alpha inside [0.15, 0.25].
    "lowfreq_final_mod_range": 0.05,
    # the final composite only ever runs on a finished sample, so it is trained
    # on its own low-t branch instead of being pushed through the high-t states
    "final_branch": True,
    "final_t_range": [1, 5],
    "final_loss_weight": 1.0,
    # False -> no parameters at all (the training-free fixed blend)
    "trainable": True,
    # diffusion steps used for training; None -> reuse apply_t_range
    "train_t_range": None,
    # space the corrector *reads*: "x0" (clean estimate) or "eps" (noise)
    "input": "x0",
    # space the corrector *writes*: "eps" or "x0"
    "output": "eps",
    # how the corrector output is merged with the denoiser prediction
    "combine": "residual",
    # bounded gate (only used when combine == "gated"); 0.0 == identity
    "gate_init": 0.0,
    "gate_shape": "channel",  # channel | scalar
    "gate_activation": "tanh",  # tanh | none
    # reverse steps at which the corrector runs: [start, end] in IR-SDE t units
    # (t = 1 is the cleanest state, t = T is pure noise)
    "apply_t_range": [1, 100],
    # correction iterations per reverse step at sampling / during training
    "steps": 1,
    "train_steps": 1,
    # gradient boundaries, as in the reference implementation
    "detach_input": True,
    "detach_skip": True,
    # feed the denoiser prediction to the corrector as an extra feature
    "proposal_injection": True,
    # freeze the denoiser and train the corrector only
    "freeze_base": True,
    # optimiser (None -> reuse train.optimizer)
    "optimizer": None,  # Adam | AdamW | Lion
    "lr_correction": None,  # None -> train.lr_G
    # the copied backbone only needs a gentle nudge; the new head has to move
    "lr_backbone_scale": 0.2,
    # 0 by default: AdamW weight decay would slowly erode the denoiser weights
    # the corrector was initialised from
    "weight_decay_correction": 0.0,
    "grad_clip": 1.0,
    # loss weights (base loss is a monitor only when the base is frozen)
    "loss_weight": 1.0,
    "base_loss_weight": 0.0,
}

_VALID_INPUT = ("x0", "eps")
_VALID_OUTPUT = ("x0", "eps")
_VALID_COMBINE = ("direct", "residual", "gated")


@dataclass(frozen=True)
class CorrectionSpec:
    """Resolved, validated correction configuration."""

    input: str = "x0"
    output: str = "eps"
    combine: str = "residual"
    gate_init: float = 0.0
    gate_shape: str = "channel"
    gate_activation: str = "tanh"
    proposal_injection: bool = True
    detach_input: bool = True
    detach_skip: bool = True

    @property
    def identity_at_init(self) -> bool:
        """Whether a zero-initialised head makes the corrector a no-op."""
        return self.combine in ("residual", "gated")

    def validate(self):
        if self.input not in _VALID_INPUT:
            raise ValueError(
                "correction.input must be one of {}, got {!r}".format(_VALID_INPUT, self.input)
            )
        if self.output not in _VALID_OUTPUT:
            raise ValueError(
                "correction.output must be one of {}, got {!r}".format(_VALID_OUTPUT, self.output)
            )
        if self.combine not in _VALID_COMBINE:
            raise ValueError(
                "correction.combine must be one of {}, got {!r}".format(
                    _VALID_COMBINE, self.combine
                )
            )
        if self.gate_shape not in ("channel", "scalar"):
            raise ValueError("correction.gate_shape must be channel or scalar")
        if self.gate_activation not in ("tanh", "none"):
            raise ValueError("correction.gate_activation must be tanh or none")
        if self.combine == "direct" and self.output == "x0":
            # a zero-initialised head would emit x0 = 0, which is *not* the
            # identity in eps space, so there is no safe initialisation
            raise ValueError(
                "correction.combine='direct' is only supported with "
                "correction.output='eps'; use combine='residual' or 'gated' "
                "when the corrector emits a clean-image estimate"
            )
        return self


def normalize_correction_config(cfg) -> dict:
    """Merge ``cfg`` over ``DEFAULT_CONFIG`` and validate the result."""
    out = dict(DEFAULT_CONFIG)
    if cfg:
        unknown = [k for k in cfg if k not in DEFAULT_CONFIG]
        if unknown:
            raise ValueError(
                "unknown key(s) in the 'correction' block: {} (allowed: {})".format(
                    sorted(unknown), sorted(DEFAULT_CONFIG)
                )
            )
        out.update({k: v for k, v in cfg.items() if v is not None or k == "lr_correction"})
    spec = CorrectionSpec(
        input=str(out["input"]),
        output=str(out["output"]),
        combine=str(out["combine"]),
        gate_init=float(out["gate_init"]),
        gate_shape=str(out["gate_shape"]),
        gate_activation=str(out["gate_activation"]),
        proposal_injection=bool(out["proposal_injection"]),
        detach_input=bool(out["detach_input"]),
        detach_skip=bool(out["detach_skip"]),
    ).validate()
    out["spec"] = spec

    opt_name = out["optimizer"]
    if opt_name is not None:
        opt_name = str(opt_name)
        if opt_name not in ("Adam", "AdamW", "Lion"):
            raise ValueError(
                "correction.optimizer must be Adam, AdamW, Lion or null, got {!r}".format(
                    opt_name
                )
            )
    out["optimizer"] = opt_name

    scale = float(out["lr_backbone_scale"])
    if not (0.0 < scale <= 1.0):
        raise ValueError(
            "correction.lr_backbone_scale must be in (0, 1], got {}".format(scale)
        )
    out["lr_backbone_scale"] = scale

    # the final composite strength is a sharply peaked optimum, so its
    # modulation has to stay small or training walks off the peak
    mod_range = float(out["lowfreq_final_mod_range"])
    if not (0.0 <= mod_range <= 0.5):
        raise ValueError(
            "correction.lowfreq_final_mod_range must be in [0, 0.5], got {}".format(
                mod_range
            )
        )
    out["lowfreq_final_mod_range"] = mod_range

    t_range = out["apply_t_range"]
    if isinstance(t_range, (int, float)):
        t_range = [1, int(t_range)]
    if len(t_range) != 2:
        raise ValueError("correction.apply_t_range must be [start, end]")
    lo, hi = int(t_range[0]), int(t_range[1])
    if lo > hi:
        raise ValueError("correction.apply_t_range must satisfy start <= end")
    out["apply_t_range"] = (lo, hi)

    for key in ("steps", "train_steps"):
        out[key] = int(out[key])
        if out[key] < 0:
            raise ValueError("correction.{} must be >= 0".format(key))

    arch = str(out["arch"])
    if arch not in ("copy", "lowfreq"):
        raise ValueError(
            "correction.arch must be 'copy' or 'lowfreq', got {!r}".format(arch)
        )
    out["arch"] = arch
    if arch == "lowfreq":
        # the low-frequency gate emits a displacement in x0 space
        if out["output"] != "x0" or out["combine"] != "residual":
            raise ValueError(
                "correction.arch='lowfreq' works in the clean-image space: set "
                "correction.output='x0' and correction.combine='residual' "
                "(got output={!r}, combine={!r})".format(out["output"], out["combine"])
            )
        if out["spec"].input != "x0":
            raise ValueError(
                "correction.arch='lowfreq' reads the clean estimate: set "
                "correction.input='x0'"
            )

    ft = out["final_t_range"]
    if isinstance(ft, (int, float)):
        ft = [int(ft), int(ft)]
    if len(ft) != 2 or int(ft[0]) > int(ft[1]):
        raise ValueError("correction.final_t_range must be [low, high] with low <= high")
    out["final_t_range"] = (int(ft[0]), int(ft[1]))

    t_tr = out["train_t_range"]
    if t_tr is None:
        out["train_t_range"] = tuple(out["apply_t_range"])
    else:
        if isinstance(t_tr, (int, float)):
            t_tr = [int(t_tr), int(t_tr)]
        if len(t_tr) != 2:
            raise ValueError("correction.train_t_range must be [start, end]")
        lo, hi = int(t_tr[0]), int(t_tr[1])
        if lo > hi:
            raise ValueError("correction.train_t_range must satisfy start <= end")
        out["train_t_range"] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


class CorrectionGate(nn.Module):
    """Bounded scalar/channel gain applied to the corrector's residual.

    ``gate_init = 0`` with ``tanh`` means the gate starts at exactly 0, so the
    correction starts at exactly 0 -- the network cannot damage the baseline
    before it has learned anything.
    """

    def __init__(self, channels=3, shape="channel", init=0.0, activation="tanh"):
        super().__init__()
        n = int(channels) if shape == "channel" else 1
        self.raw = nn.Parameter(torch.full((n,), float(init)))
        self.activation = activation

    def value(self) -> torch.Tensor:
        if self.activation == "tanh":
            return torch.tanh(self.raw)
        return self.raw

    def forward(self, like: torch.Tensor) -> torch.Tensor:
        g = self.value().to(dtype=like.dtype, device=like.device)
        return g.view(1, -1, *([1] * (like.dim() - 2)))


# ---------------------------------------------------------------------------
# the corrector container
# ---------------------------------------------------------------------------


class DiffusionCorrector(nn.Module):
    """Wraps the correction network plus everything the sampler needs to know."""

    def __init__(self, net, spec: CorrectionSpec, gate=None, apply_t_range=(1, 100), steps=1):
        super().__init__()
        self.net = net
        self.spec = spec
        self.gate = gate
        self.apply_t_range = (int(apply_t_range[0]), int(apply_t_range[1]))
        self.steps = int(steps)

    def should_apply(self, t) -> bool:
        """Whether the corrector acts on reverse step ``t``."""
        if self.steps <= 0:
            return False
        lo, hi = self.apply_t_range
        if torch.is_tensor(t):
            t = int(t.reshape(-1)[0].item())
        else:
            t = int(t)
        return lo <= t <= hi

    def finalize(self, x, mu, mask):
        """Optional post-sampling hook (only the lowfreq gate implements it)."""
        if hasattr(self.net, "finalize"):
            return self.net.finalize(x, mu, mask)
        return x

    def extra_repr(self):
        return "input={}, output={}, combine={}, steps={}, t_range={}".format(
            self.spec.input, self.spec.output, self.spec.combine, self.steps, self.apply_t_range
        )


# ---------------------------------------------------------------------------
# the refinement loop (single source of truth for training and sampling)
# ---------------------------------------------------------------------------


def refine(corrector, sde, x_t, eps, t, *, steps=1, detach_input=False, detach_skip=False):
    """Run ``steps`` correction iterations on the denoiser's noise prediction.

    Parameters
    ----------
    corrector : DiffusionCorrector
    sde : IRSDE
        Supplies ``mu``, ``mask`` and the exact ``eps <-> x0`` conversions.
    x_t : torch.Tensor
        Current noisy state; held fixed across all iterations (as in the
        reference implementation, ``z`` and ``t`` never change inside ``refine``).
    eps : torch.Tensor
        The denoiser's noise prediction at ``t``.
    t : int or torch.Tensor
        Diffusion step(s).

    Returns
    -------
    torch.Tensor
        The corrected noise prediction, in eps space.
    """
    if steps <= 0:
        return eps

    spec = corrector.spec
    net = corrector.net
    mu = sde.mu
    mask = sde.mask
    t_flat = t.reshape(-1) if torch.is_tensor(t) else t

    def to_output_space(noise):
        if spec.output == "eps":
            return noise
        return sde.get_init_state_from_noise(x_t, noise, t)

    def to_proposal_space(noise):
        if spec.input == "eps":
            return noise
        return sde.get_init_state_from_noise(x_t, noise, t)

    proposal = to_proposal_space(eps)

    for i in range(steps):
        value = proposal.detach() if detach_input else proposal
        pred = net(x_t, mu, t_flat, mask, proposal=value)

        if spec.combine == "direct":
            combined = pred
        else:
            skip = to_output_space(eps)
            if detach_skip:
                skip = skip.detach()
            if spec.combine == "residual":
                # ``pred`` is a displacement in the output space
                combined = skip + pred
            else:  # gated: bounded displacement, identity when the gate is 0
                combined = skip + corrector.gate(pred) * pred

        eps = combined if spec.output == "eps" else sde.get_noise_from_init_state(x_t, combined, t)

        if i + 1 < steps:
            proposal = to_proposal_space(eps)

    return eps


# ---------------------------------------------------------------------------
# initialisation helpers
# ---------------------------------------------------------------------------


def zero_init_output_head(net) -> str:
    """Zero the network's output convolution so its prediction starts at 0."""
    head = getattr(net, "ending", None)
    if head is None or not isinstance(head, nn.Conv2d):
        raise AttributeError(
            "cannot zero-initialise the correction head: {} has no "
            "'ending' Conv2d. Correction training needs an explicit output "
            "layer.".format(net.__class__.__name__)
        )
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return "ending"


def init_corrector_from_base(corr_net, base_net, spec: CorrectionSpec) -> dict:
    """Copy the denoiser weights into the corrector and make it the identity.

    The corrector is structurally the denoiser plus an optional zero-initialised
    proposal projection, so every key of the denoiser's state dict must be
    present in the corrector; only the proposal projection may be new.
    """
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    missing, unexpected = corr_net.load_state_dict(base_sd, strict=False)
    bad_missing = [k for k in missing if not k.startswith("proposal_proj.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            "corrector/base parameter mismatch: missing={} unexpected={}".format(
                bad_missing[:5], list(unexpected)[:5]
            )
        )

    report = {"copied": len(base_sd), "new": list(missing)}
    if spec.identity_at_init:
        report["zero_init_head"] = zero_init_output_head(corr_net)
    elif spec.output == "eps":
        # direct + eps: the corrector starts as an exact copy of the denoiser,
        # i.e. at the baseline quality, and then improves on it
        report["zero_init_head"] = None
    else:  # pragma: no cover - rejected by CorrectionSpec.validate()
        raise ValueError("unsupported correction initialisation")
    return report


def count_parameters(module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
