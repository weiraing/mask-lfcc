"""Boundary-consistent low-frequency correction (LFCC) for Refusion inpainting.

Why this replaces the full-copy epsilon corrector
-------------------------------------------------
Measured on the 604-image dark validation split (see ``LFCC_ANALYSIS.md``):

* 92% of the remaining error energy sits inside the hole;
* 87% of that error is carried by scales coarser than 4 px;
* replacing the coarse band of the model's hole content with a *harmonic
  (Laplace) fill driven by the observed boundary* raises hole PSNR by ~0.9 dB.

In other words the dominant residual is not texture and not per-pixel noise: it
is the large-scale level/structure of the invented content.  A corrector that
only refines the noise prediction cannot address it -- which is exactly what the
full-copy epsilon corrector showed: -20% training loss, +0.26 dB PSNR.

This module therefore corrects ``x0`` (the clean estimate) instead of ``eps``,
and it corrects it *towards* a boundary-derived field:

    x0' = x0 + M * alpha * (F - x0)

``F`` is the harmonic fill of the hole computed from the observed pixels only
(the same information the denoiser already had -- no leakage), and ``alpha`` is
either a constant (training-free variant) or a small learned gate that decides,
per pixel, how much of the boundary extrapolation to trust.

Design guarantees
-----------------
* ``alpha_init = 0`` -> the module is the exact identity, so training starts at
  the baseline and can only move away from it by lowering the supervised loss.
* ``F`` depends only on ``(mu, mask)``, so it is computed once per image and
  cached; inside the 100-step reverse loop it costs nothing.
* The gate has ~0.2M parameters, against 76.6M for a denoiser copy, so it cannot
  re-fit the 2138 training images the way the copy did.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["harmonic_fill", "LowFreqCorrectorNet", "count_parameters"]


# ---------------------------------------------------------------------------
# boundary-driven harmonic fill
# ---------------------------------------------------------------------------


def _masked_down(x, known, scale):
    """Masked average pooling: returns the coarse value and observed support."""
    num = F.avg_pool2d(x * known, scale)
    den = F.avg_pool2d(known, scale)
    return num / den.clamp_min(1e-6), den


def _jacobi(vals, hole, iters):
    """Solve Laplace inside ``hole`` with Dirichlet values from ``vals``."""
    c = vals.shape[1]
    k = torch.full((c, 1, 3, 3), 1.0 / 9.0, dtype=vals.dtype, device=vals.device)
    cur = vals
    for _ in range(iters):
        nb = F.conv2d(F.pad(cur, (1, 1, 1, 1), mode="replicate"), k, groups=c)
        cur = torch.where(hole, nb, vals)
    return cur


def harmonic_fill(mu, mask, cells=64, iters=160, coarse_iters=200):
    """Smooth (harmonic) extension of the observed pixels into the hole.

    Parameters
    ----------
    mu : [B, 3, H, W]
        The masked condition image.  Observed pixels are exact, hole pixels are
        ignored.
    mask : [B, 1, H, W]
        1 = missing/hole, 0 = observed (same convention as the denoiser).
    cells : int
        Side of the coarse grid used for the solve.  Fixing the *grid* rather
        than the stride keeps the prior's physical scale identical between the
        128 px training crops and the 256 px validation images.
    """
    if mask is None:
        return mu
    b, c, h, w = mu.shape
    scale = max(1, int(round(min(h, w) / float(cells))))
    known = (1.0 - mask).clamp(0.0, 1.0)

    val, den = _masked_down(mu, known, scale)
    hole = den < 0.5

    # seed the hole with the observed mean so large holes converge quickly
    if hole.any():
        seen = den.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        mean = (val * den).sum(dim=(2, 3), keepdim=True) / seen
        val = torch.where(hole, mean.expand_as(val), val)

    # two-level solve: coarse pass first, then refine on the target grid
    if coarse_iters > 0 and min(val.shape[-2:]) >= 8:
        small_val = F.avg_pool2d(val, 2)
        small_hole = F.avg_pool2d(hole.float(), 2) > 0.5
        small_val = _jacobi(small_val, small_hole, coarse_iters)
        up = F.interpolate(small_val, size=val.shape[-2:], mode="bilinear", align_corners=False)
        val = torch.where(hole, up, val)

    val = _jacobi(val, hole, iters)
    return F.interpolate(val, size=(h, w), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# the corrector network
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, ch, emb):
        super().__init__()
        self.norm = nn.GroupNorm(4, ch)
        self.conv1 = nn.Conv2d(ch, ch * 2, 1)
        self.dw = nn.Conv2d(ch * 2, ch * 2, 3, padding=1, groups=ch * 2)
        self.conv2 = nn.Conv2d(ch * 2, ch, 1)
        self.scale = nn.Linear(emb, ch)
        self.shift = nn.Linear(emb, ch)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, temb):
        h = self.norm(x)
        h = self.conv1(h)
        h = self.dw(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h * (1 + self.scale(temb)[:, :, None, None]) + self.shift(temb)[:, :, None, None]


class LowFreqCorrectorNet(nn.Module):
    """Predicts a per-pixel blend weight between ``x0`` and the boundary fill.

    Signature matches what :func:`models.modules.correction.refine` expects from
    a corrector network, so this plugs into the existing training loop and
    sampler without touching either.
    """

    def __init__(
        self,
        in_channels=3,
        hidden_channels=48,
        num_blocks=4,
        num_timesteps=100,
        cells=64,
        iters=160,
        coarse_iters=200,
        alpha_init=0.0,
        alpha_final_init=0.2,
        alpha_range=0.5,
        alpha_final_range=0.05,
        trainable=True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden = int(hidden_channels)
        self.num_timesteps = int(num_timesteps)
        self.cells = int(cells)
        self.iters = int(iters)
        self.coarse_iters = int(coarse_iters)
        self.trainable = bool(trainable)
        self.alpha_range = float(alpha_range)
        # Amplitude of the *final composite* modulation.  It must stay small:
        # the alpha sweep on the dark val is sharply peaked at 0.2 (0.3 already
        # costs 0.12 dB inside the hole, 0.4 costs 0.44 dB), so a modulation
        # with a 0.5 range can walk the effective alpha straight past the peak
        # and cancel the whole gain.  0.05 keeps it inside [0.15, 0.25].
        self.alpha_final_range = float(alpha_final_range)
        self._last_alpha = None
        # plain floats, not buffers: they must not enter the state_dict,
        # otherwise a test config silently inherits the training value
        self.alpha_init = float(alpha_init)
        self.alpha_final_init = float(alpha_final_init)

        # Final composite strength.  A single reverse step at t=1 only moves the
        # output by dt ~ 1.25%, so the in-sampler correction cannot deliver the
        # measured blend gain on its own; this term does, and it starts exactly
        # at the measured optimum so training can only improve on it.
        self._alpha_final_const = self.alpha_final_init
        if self.trainable:
            raw = max(min(self.alpha_final_init, 0.999), 0.001)
            self.alpha_final_raw = nn.Parameter(
                torch.tensor(math.log(raw / (1.0 - raw)))
            )
        else:
            self.alpha_final_raw = None
            # a frozen module is the training-free ablation: no parameters move
            for p in self.parameters():
                p.requires_grad_(False)

        emb = 64
        # x0 + fill + mu + mask + t map
        cin = 3 * self.in_channels + 1 + 1
        self.emb = nn.Sequential(
            nn.Linear(1, emb), nn.SiLU(), nn.Linear(emb, emb)
        )
        self.in_conv = nn.Conv2d(cin, self.hidden, 3, padding=1)
        self.blocks = nn.ModuleList([_Block(self.hidden, emb) for _ in range(int(num_blocks))])
        self.out = nn.Conv2d(self.hidden, self.in_channels, 3, padding=1)
        # start at exactly ``alpha_init`` (0 -> identity)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self._fill = None
        self._fill_key = None

    # -- boundary field (cached: it only depends on mu and mask) -----------
    def fill(self, mu, mask):
        key = (mu.data_ptr(), None if mask is None else mask.data_ptr(), tuple(mu.shape))
        if self._fill is not None and self._fill_key == key:
            return self._fill
        f = harmonic_fill(mu, mask, cells=self.cells, iters=self.iters,
                          coarse_iters=self.coarse_iters)
        self._fill, self._fill_key = f, key
        return f

    def clear_cache(self):
        self._fill, self._fill_key = None, None

    def forward(self, x_t, mu, t, mask=None, proposal=None):
        """Return a displacement in ``x0`` space.

        ``x_t`` is unused beyond dtype/device; ``proposal`` carries the clean
        estimate the sampler computed from the denoiser's prediction.
        """
        x0 = proposal if proposal is not None else mu
        fill = self.fill(mu, mask)
        if mask is None:
            mask = torch.zeros_like(fill[:, :1])

        if not self.trainable:
            alpha = torch.full(
                (x0.shape[0], 1, 1, 1), self.alpha_init,
                device=x0.device, dtype=x0.dtype,
            )
            return mask * alpha * (fill - x0)

        if torch.is_tensor(t):
            tt = t.reshape(-1).to(device=x0.device, dtype=x0.dtype)
        else:
            tt = torch.full((x0.shape[0],), float(t), device=x0.device, dtype=x0.dtype)
        tn = (tt / max(1, self.num_timesteps)).view(-1, 1)
        temb = self.emb(tn)
        tmap = tn.view(-1, 1, 1, 1).expand(-1, 1, x0.shape[-2], x0.shape[-1])

        h = self.in_conv(torch.cat([x0, fill, mu, mask, tmap], dim=1))
        for blk in self.blocks:
            h = blk(h, temb)
        logits = self.out(h)
        alpha = (self.alpha_init + self.alpha_range * torch.tanh(logits)).clamp(0.0, 1.0)
        return mask * alpha * (fill - x0)

    def finalize(self, x, mu, mask):
        """Blend the finished sample towards the boundary field, once.

        This is the operation whose gain was measured on the full validation
        split (alpha = 0.2 -> +0.40 dB inside the hole, per-image oracle
        +0.76 dB).  ``alpha`` starts at the measured optimum and, when the gate
        is trained, gets a per-pixel modulation around it.
        """
        if mask is None:
            return x
        fill = self.fill(mu, mask)
        if self.trainable:
            # a = measured optimum, b = zero-at-init per-pixel modulation
            alpha = (
                torch.sigmoid(self.alpha_final_raw)
                + self._final_modulation(x, mu, mask)
            ).clamp(0.0, 1.0)
        else:
            alpha = torch.full(
                (x.shape[0], 1, 1, 1), self._alpha_final_const,
                device=x.device, dtype=x.dtype,
            )
        if mask.any():
            # effective strength inside the hole, for the training log
            self._last_alpha = float(
                (alpha * mask).sum() / mask.sum().clamp_min(1)
            )
        return x + mask * alpha * (fill - x)

    def _final_modulation(self, x, mu, mask):
        """Zero-at-init per-pixel offset for the final composite strength."""
        tmap = torch.zeros(
            x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
        )
        h = self.in_conv(torch.cat([x, self.fill(mu, mask), mu, mask, tmap], dim=1))
        temb = self.emb(torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype))
        for blk in self.blocks:
            h = blk(h, temb)
        return self.alpha_final_range * torch.tanh(self.out(h))


def count_parameters(module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
