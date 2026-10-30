"""Multi-scale Texture and Structure Collaborative Enhancement (MTSCE).

.. deprecated::
   MTSCE has been **retired** as the project's contribution and replaced by
   correction training (``models/modules/correction.py``). The file is kept so
   that the historical experiment (``refusion_mask_mtsce_*``) stays
   reproducible; none of the current configs contain a ``mtsce`` block, so the
   base network is identical to the mask-only baseline. See
   ``CORRECTION_IMPLEMENTATION.md`` §4 and ``MTSCE_IMPLEMENTATION.md`` §9 for
   why it was dropped: two of its three branches duplicate operators that NAFNet
   already has, the mask gate re-injects information the input already carries,
   and at the bottleneck each of its 9.72M parameters saw only ~3 spatial
   positions over a full 30k-step run.

A lightweight, trainable, plug-and-play feature enhancement block for the
Refusion (IR-SDE + ConditionalNAFNet) inpainting denoiser. It does **not**
touch the diffusion formulation, the noise schedule, the sampling algorithm or
the supervised loss; it only re-weights / enhances intermediate features of the
score network.

Computation (see "Refusion 多尺度纹理与结构协同增强模块实现方案.md"):

    F0  = Proj(F)                       # 1x1 channel projection
    Fs  = S(F0)                         # multi-scale structure branch
    Ft  = T(F0)                         # local texture branch
    Ff  = A(Fs, Ft, E(Ms))              # mask-aware adaptive fusion
    Fout= F + alpha * Proj_out(Ff)      # residual output, zero-initialised

Mask convention (must match ``data/LQGTMask_dataset.py``):
    M(x, y) = 1 -> degraded / missing (hole) region
    M(x, y) = 0 -> known / clean region
Set ``mask_invert: true`` in the config if a dataset uses the opposite
convention; the conversion happens inside the module, so the rest of the
pipeline is untouched.

Design notes
------------
* ``Proj_out`` is zero-initialised, so an untrained MTSCE is an **exact
  identity**: enabling the module cannot change the behaviour of an already
  trained checkpoint before fine-tuning starts.
* Every sub-branch can be switched off individually (``use_structure``,
  ``use_texture``, ``fusion: adaptive|mean|add``) to support the D1-D5 ablation
  study. Disabled branches are replaced by zeros of the same shape so that the
  parameter layout stays identical across ablations (only the module config
  changes, never the checkpoint layout).
* ``use_mask_condition: false`` removes the mask from the fusion gate, which
  gives the "no mask prior" ablation without changing anything else.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("base")

__all__ = [
    "MTSCE",
    "MultiScaleStructureBranch",
    "LocalTextureBranch",
    "MaskAwareFusion",
    "resize_mask",
    "normalize_mtsce_config",
    "count_parameters",
]

# ---------------------------------------------------------------------------
# config handling
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "enabled": False,
    "insertion_points": ["bottleneck"],
    "hidden_ratio": 0.5,
    "min_hidden": 8,
    "structure_dilations": [1, 2, 3],
    "use_structure": True,
    "texture_kernel_size": 3,
    "use_texture": True,
    "texture_norm": True,
    "fusion": "adaptive",  # adaptive | mean | add
    "use_mask_condition": True,
    "mask_feat_ch": 8,
    "mask_invert": False,
    "residual_scale_init": 0.1,
}

_VALID_FUSIONS = ("adaptive", "mean", "add")


def normalize_mtsce_config(cfg=None):
    """Merge a user config dict with the defaults and validate it.

    Unknown keys raise, so a typo in a YAML file cannot silently disable a
    mechanism and invalidate an ablation run.
    """
    if cfg is None:
        return dict(_DEFAULT_CONFIG)
    if not isinstance(cfg, dict):
        raise TypeError(
            "mtsce config must be a dict, got {}".format(type(cfg).__name__)
        )

    unknown = set(cfg.keys()) - set(_DEFAULT_CONFIG.keys())
    if unknown:
        raise ValueError(
            "unknown mtsce config key(s): {}. Valid keys: {}".format(
                sorted(unknown), sorted(_DEFAULT_CONFIG.keys())
            )
        )

    out = dict(_DEFAULT_CONFIG)
    out.update(cfg)

    out["enabled"] = bool(out["enabled"])
    out["insertion_points"] = list(out["insertion_points"] or [])
    out["hidden_ratio"] = float(out["hidden_ratio"])
    out["min_hidden"] = int(out["min_hidden"])
    out["structure_dilations"] = [int(d) for d in (out["structure_dilations"] or [])]
    out["use_structure"] = bool(out["use_structure"])
    out["use_texture"] = bool(out["use_texture"])
    out["texture_norm"] = bool(out["texture_norm"])
    out["texture_kernel_size"] = int(out["texture_kernel_size"])
    out["use_mask_condition"] = bool(out["use_mask_condition"])
    out["mask_feat_ch"] = int(out["mask_feat_ch"])
    out["mask_invert"] = bool(out["mask_invert"])
    out["residual_scale_init"] = float(out["residual_scale_init"])

    if out["fusion"] not in _VALID_FUSIONS:
        raise ValueError(
            "mtsce.fusion must be one of {}, got {!r}".format(
                _VALID_FUSIONS, out["fusion"]
            )
        )
    if out["texture_kernel_size"] % 2 != 1 or out["texture_kernel_size"] < 1:
        raise ValueError(
            "mtsce.texture_kernel_size must be a positive odd number, got {}".format(
                out["texture_kernel_size"]
            )
        )
    if out["hidden_ratio"] <= 0:
        raise ValueError("mtsce.hidden_ratio must be > 0")
    if out["mask_feat_ch"] <= 0:
        raise ValueError("mtsce.mask_feat_ch must be > 0")
    if out["enabled"] and not out["insertion_points"]:
        raise ValueError("mtsce.enabled is true but insertion_points is empty")
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _norm_groups(channels, max_groups=8):
    """Largest group count <= max_groups that divides ``channels``."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def resize_mask(mask, size, invert=False, dtype=None, device=None):
    """Nearest-neighbour resize of a binary mask to ``size`` = (H, W).

    Nearest interpolation is mandatory here: bilinear/bicubic interpolation
    would create fractional mask values that have no physical meaning and
    would blur the hole boundary used by the fusion gate.

    ``mask=None`` yields an all-zero (fully known) mask, which keeps the module
    usable on datasets without an explicit mask.
    """
    if mask is None:
        if dtype is None:
            dtype = torch.float32
        return torch.zeros((1, 1, size[0], size[1]), dtype=dtype, device=device)

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(
            "mask must be [B, 1, H, W] or [B, H, W], got shape {}".format(
                tuple(mask.shape)
            )
        )
    if mask.shape[1] != 1:
        # collapse a (rare) multi-channel mask to one channel
        mask = mask.amax(dim=1, keepdim=True)

    if dtype is not None:
        mask = mask.to(dtype)
    if device is not None:
        mask = mask.to(device)

    if mask.shape[-2:] != tuple(size):
        mask = F.interpolate(mask, size=tuple(size), mode="nearest")

    if invert:
        mask = 1.0 - mask
    return mask


def count_parameters(module):
    """Number of trainable / total parameters of a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# sub-module 1: multi-scale structure branch
# ---------------------------------------------------------------------------


class MultiScaleStructureBranch(nn.Module):
    """Parallel dilated 3x3 convolutions for multi-scale structure context.

    ``F_s = Conv1x1(Concat(Conv3x3_d1(F0), Conv3x3_d2(F0), ...))``

    Channels are first reduced by a 1x1 convolution so the parallel dilated
    convolutions run on a narrow width (``hidden_ratio`` of the input width).
    """

    def __init__(self, in_ch, mid_ch, dilations=(1, 2, 3), act=True):
        super().__init__()
        dilations = [int(d) for d in dilations] or [1]
        if any(d < 1 for d in dilations):
            raise ValueError("dilations must be >= 1, got {}".format(dilations))
        self.dilations = dilations

        self.reduce = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=True)
        self.act = nn.GELU() if act else nn.Identity()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    mid_ch,
                    mid_ch,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                    bias=True,
                )
                for d in dilations
            ]
        )
        self.fuse = nn.Conv2d(mid_ch * len(dilations), mid_ch, kernel_size=1, bias=True)

    def forward(self, x):
        h = self.act(self.reduce(x))
        feats = [branch(h) for branch in self.branches]
        return self.fuse(torch.cat(feats, dim=1))


# ---------------------------------------------------------------------------
# sub-module 2: local texture branch
# ---------------------------------------------------------------------------


class LocalTextureBranch(nn.Module):
    """Depthwise-separable local detail extractor.

    DW 3x3 -> norm -> PW 1x1 -> GELU -> DW 3x3 -> norm -> PW 1x1

    Kept deliberately cheap: no Fourier transform and no attention in the first
    version, so any measured gain can be attributed to the structure/texture
    split rather than to a second, unrelated mechanism.
    """

    def __init__(self, in_ch, mid_ch, kernel_size=3, use_norm=True):
        super().__init__()
        pad = kernel_size // 2
        self.dw1 = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False
        )
        self.norm1 = (
            nn.GroupNorm(_norm_groups(in_ch), in_ch) if use_norm else nn.Identity()
        )
        self.pw1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.dw2 = nn.Conv2d(
            mid_ch, mid_ch, kernel_size, padding=pad, groups=mid_ch, bias=False
        )
        self.norm2 = (
            nn.GroupNorm(_norm_groups(mid_ch), mid_ch) if use_norm else nn.Identity()
        )
        self.pw2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=1, bias=True)

    def forward(self, x):
        h = self.pw1(self.norm1(self.dw1(x)))
        h = self.act(h)
        h = self.pw2(self.norm2(self.dw2(h)))
        return h


# ---------------------------------------------------------------------------
# sub-module 3: mask-aware adaptive fusion
# ---------------------------------------------------------------------------


class MaskAwareFusion(nn.Module):
    """Spatially-varying, mask-conditioned fusion of structure and texture.

    ``[A_s, A_t] = Softmax(Conv1x1([Fs, Ft, E(Ms)]))`` over the 2-channel axis,
    so ``A_s + A_t = 1`` at every pixel, and ``Ff = A_s * Fs + A_t * Ft``.

    ``fusion='mean'`` -> ``0.5 * (Fs + Ft)`` and ``fusion='add'`` ->
    ``Fs + Ft`` are the "simple fusion" baselines (ablation D4).
    """

    def __init__(self, mid_ch, mask_feat_ch=8, use_mask=True, mode="adaptive"):
        super().__init__()
        if mode not in _VALID_FUSIONS:
            raise ValueError("unknown fusion mode: {}".format(mode))
        self.mode = mode
        self.use_mask = bool(use_mask) and mode == "adaptive"

        self.mask_enc = None
        if mode == "adaptive":
            in_ch = mid_ch * 2 + (mask_feat_ch if self.use_mask else 0)
            self.gate = nn.Conv2d(in_ch, 2, kernel_size=1, bias=True)
            # neutral initialisation: softmax(0, 0) = 0.5 / 0.5 everywhere
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            if self.use_mask:
                self.mask_enc = nn.Sequential(
                    nn.Conv2d(1, mask_feat_ch, kernel_size=3, padding=1, bias=True),
                    nn.GELU(),
                    nn.Conv2d(
                        mask_feat_ch, mask_feat_ch, kernel_size=3, padding=1, bias=True
                    ),
                )

    def forward(self, f_struct, f_texture, mask_feat=None):
        if self.mode == "add":
            return f_struct + f_texture, None
        if self.mode == "mean":
            return 0.5 * (f_struct + f_texture), None

        feats = [f_struct, f_texture]
        if self.use_mask:
            if mask_feat is None:
                # no mask available: fall back to an all-known (zero) embedding
                n_mask = self.gate.in_channels - 2 * f_struct.shape[1]
                mask_feat = torch.zeros(
                    (f_struct.shape[0], n_mask) + tuple(f_struct.shape[-2:]),
                    dtype=f_struct.dtype,
                    device=f_struct.device,
                )
            feats.append(mask_feat)
        logits = self.gate(torch.cat(feats, dim=1))
        weights = torch.softmax(logits, dim=1)
        a_s, a_t = weights[:, 0:1], weights[:, 1:2]
        return a_s * f_struct + a_t * f_texture, a_s


# ---------------------------------------------------------------------------
# MTSCE
# ---------------------------------------------------------------------------


class MTSCE(nn.Module):
    """Multi-scale Texture and Structure Collaborative Enhancement module.

    Args:
        in_ch: channel count of the feature map this module is inserted into.
        hidden_ratio: width of the two branches as a fraction of ``in_ch``.
        min_hidden: lower bound for the branch width.
        structure_dilations: dilation rates of the parallel structure branch.
        texture_kernel_size: depthwise kernel size of the texture branch.
        use_structure / use_texture: ablation switches (D2 / D3).
        fusion: 'adaptive' | 'mean' | 'add' (D5 / D4).
        use_mask_condition: feed the mask to the fusion gate.
        mask_feat_ch: channels of the mask embedding ``E(Ms)``.
        mask_invert: set to true when the dataset uses 0 = hole, 1 = known.
        residual_scale_init: initial value of the learnable residual scale alpha.

    Shape: input and output are both ``[B, in_ch, H, W]``; the mask may have any
    spatial size and is resized with nearest interpolation internally.
    """

    def __init__(
        self,
        in_ch,
        hidden_ratio=0.5,
        min_hidden=8,
        structure_dilations=(1, 2, 3),
        texture_kernel_size=3,
        use_structure=True,
        use_texture=True,
        texture_norm=True,
        fusion="adaptive",
        use_mask_condition=True,
        mask_feat_ch=8,
        mask_invert=False,
        residual_scale_init=0.1,
    ):
        super().__init__()
        if in_ch <= 0:
            raise ValueError("in_ch must be positive")
        if not use_structure and not use_texture:
            raise ValueError(
                "MTSCE needs at least one of use_structure / use_texture; "
                "both off is the 'module disabled' case and should be expressed "
                "with mtsce.enabled = false."
            )

        mid_ch = max(int(min_hidden), int(round(in_ch * float(hidden_ratio))))
        self.in_ch = int(in_ch)
        self.mid_ch = mid_ch
        self.use_structure = bool(use_structure)
        self.use_texture = bool(use_texture)
        self.use_mask_condition = bool(use_mask_condition)
        self.mask_invert = bool(mask_invert)

        self.proj = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=True)

        self.structure = (
            MultiScaleStructureBranch(mid_ch, mid_ch, structure_dilations)
            if self.use_structure
            else None
        )
        self.texture = (
            LocalTextureBranch(
                mid_ch, mid_ch, texture_kernel_size, use_norm=texture_norm
            )
            if self.use_texture
            else None
        )

        self.fusion = MaskAwareFusion(
            mid_ch,
            mask_feat_ch=mask_feat_ch,
            use_mask=self.use_mask_condition,
            mode=fusion,
        )

        self.out = nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=True)
        # zero-init: an untrained MTSCE is an exact identity mapping
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # learnable residual scale, small at the beginning of fine-tuning
        self.alpha = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

        # populated by forward(..., return_aux=True); detached, for debugging
        self.last_aux = None

    # -- helpers ----------------------------------------------------------
    def _mask_feature(self, mask, size, ref):
        """E(M_s): embed the (resized) mask into ``mask_feat_ch`` channels."""
        if self.fusion.mask_enc is None:
            return None
        mask_s = resize_mask(
            mask,
            size,
            invert=self.mask_invert,
            dtype=ref.dtype,
            device=ref.device,
        )
        return self.fusion.mask_enc(mask_s)

    def extra_parameter_count(self):
        return count_parameters(self)

    # -- forward ----------------------------------------------------------
    def forward(self, x, mask=None, return_aux=False):
        if x.dim() != 4:
            raise ValueError(
                "MTSCE expects a 4D feature map [B, C, H, W], got shape {}".format(
                    tuple(x.shape)
                )
            )
        if x.shape[1] != self.in_ch:
            raise ValueError(
                "MTSCE was built for {} channels but received {}".format(
                    self.in_ch, x.shape[1]
                )
            )

        f0 = self.proj(x)

        f_struct = self.structure(f0) if self.structure is not None else None
        f_texture = self.texture(f0) if self.texture is not None else None

        # keep the parameter layout identical across the D2 / D3 ablations
        if f_struct is None:
            f_struct = torch.zeros_like(f_texture)
        if f_texture is None:
            f_texture = torch.zeros_like(f_struct)

        mask_feat = self._mask_feature(mask, f0.shape[-2:], f0)
        f_fused, gate = self.fusion(f_struct, f_texture, mask_feat)

        delta = self.out(f_fused)
        out = x + self.alpha * delta

        if return_aux:
            self.last_aux = {
                "f_struct": f_struct.detach(),
                "f_texture": f_texture.detach(),
                "gate": None if gate is None else gate.detach(),
                "delta": delta.detach(),
                "alpha": float(self.alpha.detach().cpu()),
            }
            return out, self.last_aux
        return out
