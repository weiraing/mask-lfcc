"""Mask-Guided Data Consistency (MGDC).

A training-free, plug-and-play module that enforces measurement consistency
during the reverse (sampling) process of conditional IR-SDE / Refusion-style
restoration models. It never touches network weights and only corrects the
posterior-mean estimate x0_hat at every reverse step, then maps the corrected
x0_hat back to the noise/score space so that the original SDE step (mean +
dispersion) stays intact.

Two consistency terms are applied to the estimate x0_hat at each reverse step
t (mask convention: 1 = degraded/missing region, 0 = clean/observed region):

1. Hard observed-pixel consistency (exact, always on):
       x0_hat <- (1 - M) * y + M * x0_hat,   y = (1 - M) * mu
   where mu is the masked condition image; observed pixels are pinned to
   their measured values.

2. Mask-guided low-frequency consistency inside the hole (MGDC core):
   the low-frequency content of the filled region is pushed towards a
   structure propagated from the *observed* pixels only, weighted by how
   much observed support exists around each pixel:
       LP_obs = (G_s * ((1-M) * y)) / max(G_s * (1-M), eps)
       x0_hat <- x0_hat + lambda_t * M * W * (LP_obs - G_s * x0_hat)
   with W the normalized support map (0 deep inside thick holes, ~1 near
   boundaries / for thin masks). An alternative 'global' mode replaces the
   low-frequency of the whole composite instead (ResDiff-MGDC style).
"""

import torch
import torch.nn.functional as F

__all__ = ["gaussian_blur", "MaskGuidedDataConsistency"]


def _gaussian_kernel1d(sigma, device, dtype=torch.float32):
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x, sigma):
    """Separable Gaussian blur for [B, C, H, W] tensors (reflect padding)."""
    if sigma <= 0:
        return x
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = k.numel() // 2
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=c)
    return x


class MaskGuidedDataConsistency:
    """Plug-in sampler module; constructed once per test run, called per step.

    Args:
        mode: 'masked_lowpass' (mask-normalized propagation from observed
            pixels, recommended) or 'global' (low-frequency of the whole
            composite, ResDiff-style baseline).
        sigma: low-frequency cutoff in pixels (Gaussian blur sigma).
        blend: consistency strength lambda in (0, 1].
        schedule: 'constant' | 'ramp_up' (0 at t=T -> 1 at t=1) |
            'ramp_down' (1 at t=T -> 0 at t=1); scales blend over time.
        support_tau: minimum observed support (G_s*(1-M)) below which the
            propagated low-frequency is considered unreliable and dropped.
        interval: apply the low-frequency guidance every `interval` steps
            (hard observed-pixel consistency is always applied).
        final_composite: paste observed pixels back onto the final output.
        eps: numerical floor for support normalization.
    """

    def __init__(
        self,
        mode="masked_lowpass",
        sigma=4.0,
        blend=1.0,
        schedule="constant",
        support_tau=0.05,
        interval=1,
        final_composite=True,
        eps=1e-6,
    ):
        assert mode in ("masked_lowpass", "global"), f"unknown MGDC mode: {mode}"
        assert schedule in ("constant", "ramp_up", "ramp_down"), (
            f"unknown MGDC schedule: {schedule}"
        )
        self.mode = mode
        self.sigma = float(sigma)
        self.blend = float(blend)
        self.schedule = schedule
        self.support_tau = float(support_tau)
        self.interval = max(1, int(interval))
        self.final_composite = bool(final_composite)
        self.eps = float(eps)

    # time schedule of the guidance strength
    def _lam(self, t, T):
        if self.schedule == "constant":
            return self.blend
        progress = float(t) / float(T)
        if self.schedule == "ramp_up":
            return self.blend * (1.0 - progress)
        return self.blend * progress

    def _lowfreq_guidance(self, x0, mu, mask):
        """Return (target - current) low-frequency correction and its support weight."""
        observed = 1.0 - mask
        y_obs = observed * mu
        if self.mode == "masked_lowpass":
            support = gaussian_blur(observed, self.sigma)
            lp_obs = gaussian_blur(y_obs, self.sigma) / torch.clamp(
                support, min=self.eps
            )
            weight = support / torch.clamp(
                support.amax(dim=(2, 3), keepdim=True), min=self.eps
            )
            weight = weight * (support > self.support_tau).to(x0.dtype)
        else:  # 'global': low-frequency of the current composite
            comp = y_obs + mask * x0
            lp_obs = gaussian_blur(comp, self.sigma)
            weight = torch.ones_like(mask)
        lp_pred = gaussian_blur(x0, self.sigma)
        return lp_obs - lp_pred, weight

    @torch.no_grad()
    def correct(self, x0, t, T, mu, mask):
        """Correct the posterior-mean estimate x0 at reverse step t."""
        observed = 1.0 - mask
        # 1) exact consistency on observed pixels
        x0 = observed * mu + mask * x0
        # 2) mask-guided low-frequency consistency inside the hole
        if self._lam(t, T) > 0 and (self.interval == 1 or t % self.interval == 0):
            corr, weight = self._lowfreq_guidance(x0, mu, mask)
            x0 = x0 + self._lam(t, T) * mask * weight * corr
        return x0

    def composite(self, x, mu, mask):
        """Pin observed pixels on a finished sample."""
        return (1.0 - mask) * mu + mask * x

    def __call__(self, x0, t, T, mu, mask):
        if mu is None or mask is None:
            raise ValueError(
                "MGDC requires both the condition image (mu) and the degradation "
                "mask; call sde.set_mu(...) and sde.set_mask(...) before sampling."
            )
        return self.correct(x0, t, T, mu, mask)
