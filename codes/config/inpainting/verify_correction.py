"""Self-check for the correction-training module (CPU, no data, no checkpoints).

Run this before any training run:

    cd codes/config/inpainting
    python verify_correction.py

Every check below corresponds to a property the method relies on.  If any of
them fails the run is meaningless, so the script exits non-zero.
"""

import argparse
import os
import sys
import traceback

# make ``models`` (this directory) and ``utils`` (codes/) importable no matter
# where the script is launched from
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODES = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _CODES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

import models.modules as M
from models.modules.correction import (
    CorrectionGate,
    CorrectionSpec,
    DiffusionCorrector,
    count_parameters,
    init_corrector_from_base,
    normalize_correction_config,
    refine,
)
from utils.sde_utils import IRSDE

PASSED = []
FAILED = []


def check(condition, message):
    if condition:
        PASSED.append(message)
        print("  [ok]   {}".format(message))
    else:
        FAILED.append(message)
        print("  [FAIL] {}".format(message))


def section(title):
    print("\n== {} ==".format(title))


def build_net(width=8, blocks=(1, 1), proposal_ch=0, use_mask=True):
    return M.ConditionalNAFNet(
        img_channel=3,
        width=width,
        middle_blk_num=1,
        enc_blk_nums=list(blocks),
        dec_blk_nums=[1] * len(blocks),
        use_mask=use_mask,
        proposal_ch=proposal_ch,
    )


def make_sde(T=100):
    return IRSDE(max_sigma=50, T=T, schedule="cosine", eps=0.005, device=torch.device("cpu"))


def make_corrector(spec, base, steps=1, t_range=(1, 100)):
    corr_net = build_net(proposal_ch=3 if spec.proposal_injection else 0)
    report = init_corrector_from_base(corr_net, base, spec)
    gate = None
    if spec.combine == "gated":
        gate = CorrectionGate(channels=3, shape=spec.gate_shape, init=spec.gate_init,
                              activation=spec.gate_activation)
    corr = DiffusionCorrector(corr_net, spec, gate=gate, apply_t_range=t_range, steps=steps)
    return corr, report


# ---------------------------------------------------------------------------
# 1. configuration
# ---------------------------------------------------------------------------
def test_config():
    section("configuration")

    cfg = normalize_correction_config(None)
    spec = cfg["spec"]
    check(spec.input == "x0" and spec.output == "eps" and spec.combine == "residual",
          "defaults resolve to input=x0, output=eps, combine=residual")
    check(spec.identity_at_init, "residual/gated combine is identity at initialisation")
    check(cfg["apply_t_range"] == (1, 100), "default apply_t_range is the full [1, 100]")

    try:
        normalize_correction_config({"combine": "direct", "output": "x0"})
        check(False, "combine=direct with output=x0 is rejected")
    except ValueError:
        check(True, "combine=direct with output=x0 is rejected")

    try:
        normalize_correction_config({"nonsense": 1})
        check(False, "unknown keys are rejected")
    except ValueError:
        check(True, "unknown keys are rejected")

    try:
        normalize_correction_config({"apply_t_range": [40, 1]})
        check(False, "reversed apply_t_range is rejected")
    except ValueError:
        check(True, "reversed apply_t_range is rejected")

    try:
        normalize_correction_config({"optimizer": "SGD"})
        check(False, "unknown optimiser is rejected")
    except ValueError:
        check(True, "unknown optimiser is rejected")


# ---------------------------------------------------------------------------
# 2. eps <-> x0 maps are exact inverses
# ---------------------------------------------------------------------------
def test_roundtrip():
    section("eps <-> x0 round trip (the SDE update depends on this)")

    sde = make_sde()
    torch.manual_seed(0)
    x = torch.rand(2, 3, 16, 16) * 0.5
    mu = torch.rand(2, 3, 16, 16) * 0.5
    sde.set_mu(mu)
    noise = torch.randn_like(x)

    worst = 0.0
    for t in [1, 2, 10, 40, 70, 100]:
        x0 = sde.get_init_state_from_noise(x, noise, t)
        back = sde.get_noise_from_init_state(x, x0, t)
        worst = max(worst, float((back - noise).abs().max()))
    check(worst < 1e-6, "noise -> x0 -> noise is the identity (max err {:.2e})".format(worst))


# ---------------------------------------------------------------------------
# 3. the corrector is the exact identity at initialisation
# ---------------------------------------------------------------------------
def test_identity_at_init():
    section("identity at initialisation (this is what makes C start at B)")

    sde = make_sde()
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32) * 0.5
    mu = torch.rand(2, 3, 32, 32) * 0.5
    mask = (torch.rand(2, 1, 32, 32) > 0.5).float()
    sde.set_mu(mu)
    sde.set_mask(mask)
    eps = torch.randn_like(x)

    for combine in ("residual", "gated"):
        spec = CorrectionSpec(combine=combine).validate()
        base = build_net().eval()
        corr, report = make_corrector(spec, base)
        check(report["zero_init_head"] == "ending",
              "{}: output head is zero-initialised".format(combine))
        for steps in (1, 2):
            for t in (1, 20, 60, 100):
                out = refine(corr, sde, x, eps, t, steps=steps)
                check(torch.equal(out, eps),
                      "{}: steps={} t={} returns the denoiser prediction bit-exactly".format(
                          combine, steps, t))

    spec = CorrectionSpec(combine="direct", output="eps").validate()
    base = build_net().eval()
    corr, report = make_corrector(spec, base)
    check(report["zero_init_head"] is None,
          "direct+eps keeps the denoiser head (starts equal to the denoiser, not at 0)")
    out = refine(corr, sde, x, eps, 20, steps=1)
    ref = base(x, mu, 20, mask)
    check(torch.allclose(out, ref, atol=1e-6),
          "direct+eps: the corrector starts as a copy of the denoiser")


# ---------------------------------------------------------------------------
# 4. gradients reach the corrector only
# ---------------------------------------------------------------------------
def test_gradients():
    section("gradient routing (stop-gradient through the denoiser)")

    sde = make_sde()
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32) * 0.5
    mu = torch.rand(2, 3, 32, 32) * 0.5
    sde.set_mu(mu)

    spec = CorrectionSpec().validate()
    base = build_net().eval()
    for p in base.parameters():
        p.requires_grad_(False)
    corr, _ = make_corrector(spec, base)

    # make the head non-zero so the residual path is exercised
    with torch.no_grad():
        corr.net.ending.weight.normal_(0, 0.02)
        corr.net.ending.bias.normal_(0, 0.02)

    eps = torch.randn_like(x)
    out = refine(corr, sde, x, eps, 20, steps=1)
    check(not torch.equal(out, eps), "a non-zero head does change the prediction")

    out.pow(2).mean().backward()
    head_grad = corr.net.ending.weight.grad
    check(head_grad is not None and float(head_grad.abs().sum()) > 0,
          "the output head receives a gradient")
    check(all(p.grad is None for p in base.parameters()),
          "the frozen denoiser receives no gradient")
    non_head = [n for n, p in corr.net.named_parameters()
                if p.grad is not None and float(p.grad.abs().sum()) > 0 and "ending" not in n]
    check(len(non_head) > 0,
          "gradient flows past the zero-initialised head ({} tensors)".format(len(non_head)))

    # stop-gradient: the returned prediction must not depend on the input eps
    eps_leaf = torch.randn_like(x).requires_grad_(True)
    out2 = refine(corr, sde, x, eps_leaf, 20, steps=1, detach_input=True, detach_skip=True)
    g = torch.autograd.grad(out2.sum(), eps_leaf, allow_unused=True)[0]
    check(g is None, "detach_input/detach_skip cut the graph at the denoiser prediction")

    out3 = refine(corr, sde, x, eps_leaf, 20, steps=1, detach_input=False, detach_skip=False)
    g3 = torch.autograd.grad(out3.sum(), eps_leaf, allow_unused=True)[0]
    check(g3 is not None, "with detach_* off the graph is kept (used by joint training)")


# ---------------------------------------------------------------------------
# 5. the proposal really reaches the features
# ---------------------------------------------------------------------------
def test_proposal_injection():
    section("proposal injection")

    sde = make_sde()
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32) * 0.5
    mu = torch.rand(2, 3, 32, 32) * 0.5
    sde.set_mu(mu)

    spec = CorrectionSpec().validate()
    base = build_net().eval()
    corr, _ = make_corrector(spec, base)
    check(corr.net.proposal_proj is not None, "corrector has a proposal projection")
    check(float(corr.net.proposal_proj.weight.abs().sum()) == 0.0,
          "proposal projection is zero-initialised (identity preserved)")

    with torch.no_grad():
        corr.net.proposal_proj.weight.normal_(0, 0.05)
        corr.net.ending.weight.normal_(0, 0.02)

    eps = torch.randn_like(x)
    a = refine(corr, sde, x, eps, 20, steps=1)
    b = refine(corr, sde, x, eps + 0.1, 20, steps=1)
    check(not torch.allclose(a, b, atol=1e-7),
          "changing the proposal changes the output (the feature path is live)")

    plain = build_net(proposal_ch=0)
    check(plain.proposal_proj is None, "proposal_ch=0 builds nothing (baseline unchanged)")


# ---------------------------------------------------------------------------
# 6. sampler integration
# ---------------------------------------------------------------------------
def test_sampler():
    section("sampler integration")

    sde = make_sde()
    torch.manual_seed(0)
    xt = torch.randn(1, 3, 16, 16) * 0.05
    mu = torch.rand(1, 3, 16, 16) * 0.5
    mask = (torch.rand(1, 1, 16, 16) > 0.5).float()
    sde.set_mu(mu)
    sde.set_mask(mask)
    sde.set_model(lambda x, m, t, **kw: torch.zeros_like(x))

    torch.manual_seed(1234)
    ref = sde.reverse_sde(xt.clone())
    check(sde.correction_calls == 0, "no corrector attached -> zero correction calls")

    spec = CorrectionSpec().validate()
    base = build_net().eval()
    corr, _ = make_corrector(spec, base, steps=1, t_range=(1, 100))
    sde.set_corrector(corr)
    torch.manual_seed(1234)
    out = sde.reverse_sde(xt.clone())
    check(sde.correction_calls == 100, "full range -> the corrector runs on all 100 steps")
    check(torch.equal(out, ref),
          "with an untrained (identity) corrector the output is BIT-IDENTICAL to the baseline")

    corr40, _ = make_corrector(spec, base, steps=1, t_range=(1, 40))
    sde.set_corrector(corr40)
    torch.manual_seed(1234)
    out40 = sde.reverse_sde(xt.clone())
    check(sde.correction_calls == 40, "apply_t_range=[1, 40] -> exactly 40 correction calls")
    check(torch.equal(out40, ref),
          "outside the window the sampler takes the original code path (bit-identical)")

    sde.set_corrector(None)
    torch.manual_seed(1234)
    again = sde.reverse_sde(xt.clone())
    check(torch.equal(again, ref), "detaching the corrector restores the baseline exactly")


# ---------------------------------------------------------------------------
# 7. parameter budget
# ---------------------------------------------------------------------------
def test_budget():
    section("parameter budget")

    base = build_net(width=64, blocks=(1, 1, 1, 28))
    corr = build_net(width=64, blocks=(1, 1, 1, 28), proposal_ch=3)
    b = count_parameters(base)["total"]
    c = count_parameters(corr)["total"]
    print("  denoiser (width=64, enc=[1,1,1,28]) : {:>12,} params".format(b))
    print("  corrector                             : {:>12,} params".format(c))
    print("  extra over the denoiser               : {:>12,} params (+{:.1f}%)".format(
        c - b, 100.0 * (c - b) / b))
    print("  inference cost                        : 2x NFE at steps=1 (denoiser + corrector)")
    check(c > b, "the corrector adds capacity")
    check((c - b) < 0.02 * b,
          "the only new weights are the proposal projection (< 2% of the network)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sampler", action="store_true",
                        help="skip the 100-step sampler test (it is the slowest)")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)

    tests = [
        test_config,
        test_roundtrip,
        test_identity_at_init,
        test_gradients,
        test_proposal_injection,
    ]
    if not args.skip_sampler:
        tests.append(test_sampler)
    tests.append(test_budget)

    for fn in tests:
        try:
            fn()
        except Exception:
            traceback.print_exc()
            FAILED.append("{} raised".format(fn.__name__))

    print("\n{} passed, {} failed".format(len(PASSED), len(FAILED)))
    if FAILED:
        print("FAILED:")
        for m in FAILED:
            print("  - {}".format(m))
        return 1
    print("All correction-training checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
