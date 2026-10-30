"""Self-checks for the boundary-consistent low-frequency corrector (LFCC).

CPU only, no dataset, ~20 s.  Run before training:

    python verify_lfcc.py

Everything asserted here is a property the module must have for the experiment
to be interpretable; if one fails, any PSNR difference you measure is noise.
"""

import sys
import traceback

import torch

from models.modules.correction import (
    DiffusionCorrector,
    CorrectionSpec,
    normalize_correction_config,
    refine,
)
from models.modules.lowfreq import LowFreqCorrectorNet, count_parameters, harmonic_fill
from utils.sde_utils import IRSDE

FAILED = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def ramp_image(b=1, c=3, h=64, w=64):
    """A linear ramp: its harmonic extension into a hole is exact."""
    yy = torch.arange(h, dtype=torch.float32).view(1, 1, h, 1) / (h - 1)
    xx = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w) / (w - 1)
    img = (0.3 + 0.4 * yy + 0.2 * xx).expand(b, c, h, w).clone()
    return img


def main():
    torch.manual_seed(0)
    dev = torch.device("cpu")

    print("[1] harmonic fill reproduces a field the boundary determines")
    img = ramp_image()
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 20:44, 20:44] = 1.0
    mu = img * (1 - mask)
    fill = harmonic_fill(mu, mask, cells=32, iters=400, coarse_iters=400)
    err_in = float((fill - img)[mask.expand_as(img) > 0].abs().max())
    check(err_in < 0.02, "linear ramp recovered inside the hole (max err %.4f)" % err_in)

    print("\n[2] the fill ignores whatever is stored in the hole")
    mu2 = mu.clone()
    mu2[mask.expand_as(mu2) > 0] = 0.97
    fill2 = harmonic_fill(mu2, mask, cells=32, iters=400, coarse_iters=400)
    check(
        float((fill - fill2).abs().max()) < 1e-6,
        "fill is a function of the observed pixels only",
    )

    print("\n[3] config validation")
    try:
        normalize_correction_config(
            {"arch": "lowfreq", "input": "eps", "output": "x0", "combine": "residual"}
        )
        check(False, "input='eps' with arch='lowfreq' must be rejected")
    except ValueError:
        check(True, "input='eps' with arch='lowfreq' is rejected")
    try:
        normalize_correction_config(
            {"arch": "lowfreq", "input": "x0", "output": "eps", "combine": "residual"}
        )
        check(False, "output='eps' with arch='lowfreq' must be rejected")
    except ValueError:
        check(True, "output='eps' with arch='lowfreq' is rejected")
    cfg = normalize_correction_config(
        {"arch": "lowfreq", "input": "x0", "output": "x0", "combine": "residual",
         "apply_t_range": [1, 100]}
    )
    check(
        cfg["train_t_range"] == (1, 100),
        "train_t_range defaults to apply_t_range",
    )

    print("\n[4] alpha_init = 0 makes the module the exact identity")
    net = LowFreqCorrectorNet(hidden_channels=16, num_blocks=1, alpha_init=0.0).to(dev)
    x0 = torch.rand(2, 3, 32, 32)
    m = torch.zeros(2, 1, 32, 32)
    m[:, :, 8:24, 8:24] = 1
    mu = torch.rand(2, 3, 32, 32)
    out = net(mu, mu, torch.tensor([10, 90]), m, proposal=x0)
    check(float(out.abs().max()) == 0.0, "displacement is exactly zero at step 0")

    print("\n[5] the training-free blend is exactly mask * alpha * (fill - x0)")
    fixed = LowFreqCorrectorNet(
        hidden_channels=16, num_blocks=1, alpha_init=0.2, trainable=False,
        cells=32, iters=200, coarse_iters=200,
    ).to(dev)
    out = fixed(mu, mu, 10, m, proposal=x0)
    ref = m * 0.2 * (
        harmonic_fill(mu, m, cells=32, iters=200, coarse_iters=200) - x0
    )
    check(float((out - ref).abs().max()) < 1e-6, "closed-form blend matches")

    print("\n[6] the learned gate stays inside [0, 1] and never touches known pixels")
    gate = LowFreqCorrectorNet(
        hidden_channels=16, num_blocks=2, alpha_init=0.0, alpha_range=0.5
    ).to(dev)
    with torch.no_grad():
        gate.out.weight.normal_(0, 3.0)  # force the gate far from its init
        gate.out.bias.normal_(0, 3.0)
    out = gate(mu, mu, torch.tensor([1, 100]), m, proposal=x0)
    fill = gate.fill(mu, m)
    alpha = torch.where(
        (fill - x0).abs() > 1e-6, out / (fill - x0), torch.zeros_like(out)
    )
    check(float(alpha.min()) >= -1e-5 and float(alpha.max()) <= 1 + 1e-5,
          "alpha in [0, 1] (min %.3f max %.3f)" % (alpha.min(), alpha.max()))
    check(float(out[m.expand_as(out) == 0].abs().max()) == 0.0,
          "no displacement on observed pixels")

    print("\n[7] the fill is cached: repeated calls on the same image are free")
    gate.fill(mu, m)
    first = gate._fill
    gate.fill(mu, m)
    check(gate._fill is first, "second call reuses the cached field")
    gate.clear_cache()
    check(gate._fill is None, "clear_cache drops it")

    print("\n[8] end to end through refine(): identity in, identity out")
    sde = IRSDE(max_sigma=50, T=100, schedule="cosine", eps=0.005, device=dev)
    sde.set_mu(mu)
    sde.set_mask(m)
    eps = torch.randn(2, 3, 32, 32) * 0.1
    xt = torch.randn(2, 3, 32, 32) * 0.1
    t = torch.tensor([10, 10])
    spec = CorrectionSpec(input="x0", output="x0", combine="residual")
    corr = DiffusionCorrector(net, spec, apply_t_range=(1, 100), steps=1)
    out_eps = refine(corr, sde, xt, eps, t, steps=1, detach_input=True, detach_skip=True)
    check(
        float((out_eps - eps).abs().max()) < 1e-5,
        "identity corrector leaves the noise prediction unchanged (max err %.2e)"
        % float((out_eps - eps).abs().max()),
    )

    print("\n[9] a non-zero alpha actually reaches the score")
    corr_f = DiffusionCorrector(fixed, spec, apply_t_range=(1, 100), steps=1)
    out_eps = refine(corr_f, sde, xt, eps, t, steps=1, detach_input=True, detach_skip=True)
    check(float((out_eps - eps).abs().max()) > 1e-4,
          "alpha=0.2 changes the noise prediction")

    print("\n[10] parameter budget")
    counts = count_parameters(gate)
    check(counts["total"] < 1_000_000,
          "gate has %s parameters (%.2f%% of a 76.6M denoiser)"
          % ("{:,}".format(counts["total"]), 100.0 * counts["total"] / 76_608_963))

    print()
    if FAILED:
        print("FAILED (%d):" % len(FAILED))
        for f in FAILED:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
