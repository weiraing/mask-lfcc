"""Self-check for the MTSCE module and its integration into ConditionalNAFNet.

Run from ``codes/config/inpainting``:

    python verify_mtsce.py                 # CPU, small network, fast
    python verify_mtsce.py --device cuda   # same checks on GPU
    python verify_mtsce.py --real-config   # additionally report the parameter
                                           # count / latency of the real config

The script checks, in order:

 1. config validation (unknown keys / illegal values are rejected),
 2. forward shape and padding behaviour (odd sizes, mask=None),
 3. zero-initialised MTSCE is an exact identity (bit-identical output),
 4. mask is resized with nearest interpolation and stays binary at every
    insertion point,
 5. gradients reach every MTSCE sub-module, no NaN / Inf,
 6. checkpoint compatibility: an old (mask-only) checkpoint loads into the
    MTSCE network, while missing / mismatched base tensors still raise,
 7. save / load round-trip,
 8. ``enabled: false`` reproduces the baseline network exactly,
 9. ablation switches (structure only / texture only / mean / add / no mask),
10. every insertion point can be used, invalid names are rejected.
"""

import argparse
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models.modules.mtsce as mtsce_mod  # noqa: E402
from models.base_model import load_state_dict_compatible  # noqa: E402
from models.modules.DenoisingNAFNet_arch import ConditionalNAFNet  # noqa: E402
from models.modules.mtsce import (  # noqa: E402
    MTSCE,
    normalize_mtsce_config,
    resize_mask,
)

# small network used for the functional tests (fast on CPU)
SMALL_NET = dict(
    img_channel=3,
    width=16,
    middle_blk_num=1,
    enc_blk_nums=[1, 1],
    dec_blk_nums=[1, 1],
    use_mask=True,
)

BASE_CFG = dict(
    enabled=True,
    insertion_points=["bottleneck"],
    hidden_ratio=0.5,
    structure_dilations=[1, 2, 3],
    texture_kernel_size=3,
    use_mask_condition=True,
    residual_scale_init=0.1,
)

_PASS = []
_VERBOSE = [False]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    _PASS.append(msg)
    if _VERBOSE[0]:
        print("    [ok] {}".format(msg))


def section(title):
    print("\n=== {} ===".format(title))


def make_inputs(batch=2, size=32, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    gt = torch.rand(batch, 3, size, size, generator=g)
    mask = (torch.rand(batch, 1, size, size, generator=g) > 0.6).float()
    mu = gt * (1.0 - mask)
    xt = gt + 0.1 * torch.randn(batch, 3, size, size, generator=g)
    t = torch.tensor([10] * batch)
    return xt.to(device), mu.to(device), gt.to(device), mask.to(device), t.to(device)


def build(cfg=None, device="cpu", seed=1):
    torch.manual_seed(seed)
    net = ConditionalNAFNet(**SMALL_NET, mtsce=cfg)
    return net.to(device)


# ---------------------------------------------------------------------------
# 1. config validation
# ---------------------------------------------------------------------------
def test_config_validation():
    section("1. config validation")

    cfg = normalize_mtsce_config(BASE_CFG)
    check(cfg["enabled"] is True, "config: defaults merged")
    check(cfg["min_hidden"] == 8, "config: undocumented keys get defaults")

    for bad, why in [
        (dict(BASE_CFG, typo_key=1), "unknown key"),
        (dict(BASE_CFG, fusion="attention"), "illegal fusion"),
        (dict(BASE_CFG, texture_kernel_size=2), "even texture kernel"),
        (dict(BASE_CFG, hidden_ratio=0.0), "non-positive hidden_ratio"),
        (dict(BASE_CFG, insertion_points=[]), "empty insertion_points"),
        (dict(BASE_CFG, use_structure=False, use_texture=False), "both branches off"),
    ]:
        try:
            if why == "both branches off":
                build(bad)
            else:
                normalize_mtsce_config(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("config validation missed: {}".format(why))
    check(True, "config: 6 illegal configurations rejected")


# ---------------------------------------------------------------------------
# 2. shapes / padding
# ---------------------------------------------------------------------------
def test_forward_shapes(device):
    section("2. forward shape & padding")

    net = build(BASE_CFG, device=device).eval()
    with torch.no_grad():
        for size in (32, 60):
            xt, mu, gt, mask, t = make_inputs(size=size, device=device)
            out = net(xt, mu, t, mask)
            check(
                out.shape == xt.shape,
                "shape: {}x{} in -> {} out (padding handled)".format(size, size, tuple(out.shape)),
            )
            check(torch.isfinite(out).all().item(), "forward finite at {}x{}".format(size, size))

        xt, mu, gt, mask, t = make_inputs(device=device)
        out_nomask = net(xt, mu, t, None)
        check(torch.isfinite(out_nomask).all().item(), "forward works with mask=None")

    # mask may be given at a different resolution than the image
    with torch.no_grad():
        small_mask = F.interpolate(mask, scale_factor=0.5, mode="nearest")
        out = net(xt, mu, t, small_mask)
        check(out.shape == xt.shape, "forward works with a lower-resolution mask")


def test_identity_at_init(device):
    section("3. zero-init MTSCE == baseline (exact)")

    base = build(None, device=device, seed=7).eval()
    net = build(BASE_CFG, device=device, seed=11).eval()
    report = load_state_dict_compatible(
        net, base.state_dict(), allow_missing_prefixes=("mtsce.",)
    )
    check(report["loaded"] > 0, "identity: base weights copied into MTSCE network")

    xt, mu, gt, mask, t = make_inputs(device=device)
    with torch.no_grad():
        out_base = base(xt, mu, t, mask)
        out_mtsce = net(xt, mu, t, mask)
    check(
        torch.equal(out_base, out_mtsce),
        "identity: untrained MTSCE output is bit-identical to the baseline "
        "(delta = 0 at init, max abs diff = {:.3e})".format(
            (out_base - out_mtsce).abs().max().item()
        ),
    )


# ---------------------------------------------------------------------------
# 4. mask alignment
# ---------------------------------------------------------------------------
def test_mask_alignment(device):
    section("4. mask resize / alignment")

    # direct unit test of resize_mask
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 10:30, 5:40] = 1.0
    for size in [(32, 32), (16, 16), (8, 8), (64, 64), (7, 9)]:
        out = resize_mask(mask, size)
        check(
            out.shape[-2:] == torch.Size(size),
            "resize_mask: 64x64 -> {}x{}".format(*size),
        )
        uniq = set(out.unique().tolist())
        check(
            uniq.issubset({0.0, 1.0}),
            "resize_mask: nearest keeps the mask binary at {}x{}".format(*size),
        )
    inv = resize_mask(mask, (32, 32), invert=True)
    check(
        torch.allclose(inv + resize_mask(mask, (32, 32)), torch.ones_like(inv)),
        "resize_mask: mask_invert flips the convention consistently",
    )
    z = resize_mask(None, (8, 8))
    check(float(z.sum()) == 0.0 and z.shape == (1, 1, 8, 8), "resize_mask: None -> all-known mask")

    # integration test: every insertion point must see a mask that matches its
    # own feature resolution, and the resized mask must stay binary
    cfg = dict(BASE_CFG, insertion_points=["encoder_1", "encoder_2", "bottleneck", "decoder_1", "decoder_2"])
    net = build(cfg, device=device).eval()
    check(
        len(net.mtsce_points) == 5,
        "alignment: 5 insertion points built ({})".format(net.mtsce_points),
    )

    records = []
    original = mtsce_mod.resize_mask

    def spy(mask, size, **kwargs):
        out = original(mask, size, **kwargs)
        records.append((tuple(size), tuple(out.shape[-2:]), sorted(set(out.unique().tolist()))))
        return out

    mtsce_mod.resize_mask = spy
    try:
        xt, mu, gt, mask, t = make_inputs(size=64, device=device)
        with torch.no_grad():
            net(xt, mu, t, mask)
    finally:
        mtsce_mod.resize_mask = original

    check(len(records) == 5, "alignment: one mask resize per insertion point")
    for size, got, uniq in records:
        check(size == got, "alignment: mask resized to feature size {}".format(size))
        check(
            set(uniq).issubset({0.0, 1.0}),
            "alignment: mask stays binary at feature size {}".format(size),
        )
    check(
        [r[0][0] for r in records] == [64, 32, 16, 32, 64],
        "alignment: mask resolutions follow the encoder/decoder pyramid "
        "({})".format([r[0][0] for r in records]),
    )


# ---------------------------------------------------------------------------
# 5. gradients
# ---------------------------------------------------------------------------
def test_gradients(device):
    section("5. gradient flow / numerical stability")

    net = build(BASE_CFG, device=device)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=1e-2, momentum=0.0)

    xt, mu, gt, mask, t = make_inputs(device=device)
    target = torch.randn_like(gt) * 0.1

    mtsce_params = [(n, p) for n, p in net.named_parameters() if "mtsce" in n.split(".")]
    check(len(mtsce_params) > 0, "grad: {} MTSCE parameter tensors found".format(len(mtsce_params)))

    zero_grad_after_warmup = []
    for step in range(4):
        opt.zero_grad()
        out = net(xt, mu, t, mask)
        loss = F.mse_loss(out, target)
        check(torch.isfinite(loss).item(), "grad: finite loss at step {}".format(step))
        loss.backward()

        for name, p in mtsce_params:
            check(p.grad is not None, "grad: {} has a gradient".format(name))
            check(torch.isfinite(p.grad).all().item(), "grad: {} gradient is finite".format(name))
        if step == 3:
            zero_grad_after_warmup = [n for n, p in mtsce_params if p.grad.abs().sum().item() == 0.0]
        opt.step()

    check(
        not zero_grad_after_warmup,
        "grad: after 4 steps every MTSCE tensor receives a non-zero gradient "
        "(no dead sub-module)",
    )

    key_names = [
        "mtsce.bottleneck.proj.weight",
        "mtsce.bottleneck.structure.branches.0.weight",
        "mtsce.bottleneck.structure.branches.2.weight",
        "mtsce.bottleneck.texture.dw1.weight",
        "mtsce.bottleneck.texture.pw2.weight",
        "mtsce.bottleneck.fusion.gate.weight",
        "mtsce.bottleneck.fusion.mask_enc.0.weight",
        "mtsce.bottleneck.out.weight",
    ]
    grads = dict((n, p) for n, p in net.named_parameters())
    for name in key_names:
        check(name in grads, "grad: {} exists".format(name))

    # the residual branch must actually move the features after training steps
    with torch.no_grad():
        out_after = net(xt, mu, t, mask)
    check(
        not torch.equal(out, out_after),
        "grad: output changes after a few optimisation steps (module is trainable)",
    )
    alpha = float(net.mtsce["bottleneck"].alpha.detach())
    check(abs(alpha - BASE_CFG["residual_scale_init"]) > 0.0, "grad: residual scale alpha is updated ({:.5f})".format(alpha))


# ---------------------------------------------------------------------------
# 6. checkpoint compatibility
# ---------------------------------------------------------------------------
def test_weight_compat(device):
    section("6. checkpoint compatibility")

    base = build(None, device=device, seed=3)
    net = build(BASE_CFG, device=device, seed=5)
    ckpt = base.state_dict()

    report = load_state_dict_compatible(
        net, ckpt, allow_missing_prefixes=("mtsce.",), allow_unexpected_prefixes=("mtsce.",)
    )
    n_mtsce_keys = len([k for k in net.state_dict() if k.startswith("mtsce.")])
    check(
        len(report["missing_allowed"]) == n_mtsce_keys,
        "compat: all {} MTSCE tensors are allowed to be missing".format(n_mtsce_keys),
    )
    check(report["unexpected_ignored"] == [], "compat: no unexpected key in a baseline checkpoint")

    # a real file round-trip through torch.save / torch.load
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "baseline.pth")
        torch.save(ckpt, path)
        loaded = torch.load(path, map_location="cpu")
        net2 = build(BASE_CFG, device=device, seed=9)
        report2 = load_state_dict_compatible(
            net2, loaded, allow_missing_prefixes=("mtsce.",), allow_unexpected_prefixes=("mtsce.",)
        )
        check(report2["loaded"] == report["loaded"], "compat: baseline .pth loads into the MTSCE network")

    # missing base tensor -> error
    broken = dict(ckpt)
    broken.pop("intro.weight")
    try:
        load_state_dict_compatible(net, broken, allow_missing_prefixes=("mtsce.",))
    except RuntimeError as err:
        check("intro.weight" in str(err), "compat: a missing base tensor raises a clear error")
    else:
        raise AssertionError("compat: missing base tensor was silently accepted")

    # shape mismatch -> error
    broken = dict(ckpt)
    broken["ending.weight"] = broken["ending.weight"][:, :1]
    try:
        load_state_dict_compatible(net, broken, allow_missing_prefixes=("mtsce.",))
    except RuntimeError as err:
        check("shape mismatch" in str(err), "compat: a shape mismatch raises a clear error")
    else:
        raise AssertionError("compat: shape mismatch was silently accepted")

    # unknown extra key -> error
    broken = dict(ckpt)
    broken["some_new_module.weight"] = torch.zeros(1)
    try:
        load_state_dict_compatible(net, broken, allow_missing_prefixes=("mtsce.",))
    except RuntimeError as err:
        check("some_new_module" in str(err), "compat: an unknown checkpoint key raises a clear error")
    else:
        raise AssertionError("compat: unknown checkpoint key was silently accepted")

    # a checkpoint that contains MTSCE tensors must be loadable by a network
    # that has the module disabled (ablation B with a C checkpoint)
    net_disabled = build(dict(BASE_CFG, enabled=False), device=device, seed=13)
    report3 = load_state_dict_compatible(
        net_disabled, net.state_dict(), allow_unexpected_prefixes=("mtsce.",)
    )
    check(
        len(report3["unexpected_ignored"]) == n_mtsce_keys,
        "compat: MTSCE tensors are reported and ignored when the module is off",
    )


# ---------------------------------------------------------------------------
# 7. save / load round-trip
# ---------------------------------------------------------------------------
def test_save_load(device):
    section("7. save / load round-trip")

    net = build(BASE_CFG, device=device, seed=21).eval()
    xt, mu, gt, mask, t = make_inputs(device=device)
    with torch.no_grad():
        out1 = net(xt, mu, t, mask)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mtsce_G.pth")
        torch.save(net.state_dict(), path)
        net2 = build(BASE_CFG, device=device, seed=22).eval()
        net2.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        with torch.no_grad():
            out2 = net2(xt, mu, t, mask)
    check(torch.equal(out1, out2), "checkpoint: full MTSCE state_dict round-trips exactly")


# ---------------------------------------------------------------------------
# 8. disabled path
# ---------------------------------------------------------------------------
def test_disabled(device):
    section("8. enabled: false reproduces the baseline")

    base = build(None, device=device, seed=31).eval()
    net = build(dict(BASE_CFG, enabled=False), device=device, seed=32).eval()
    check(len(net.mtsce) == 0, "disabled: no MTSCE sub-module is built")
    check(net.mtsce_parameter_count() == 0, "disabled: MTSCE parameter count is 0")
    check(
        set(net.state_dict().keys()) == set(base.state_dict().keys()),
        "disabled: state_dict layout is identical to the baseline",
    )
    net.load_state_dict(base.state_dict(), strict=True)

    xt, mu, gt, mask, t = make_inputs(device=device)
    with torch.no_grad():
        check(
            torch.equal(net(xt, mu, t, mask), base(xt, mu, t, mask)),
            "disabled: forward output is bit-identical to the baseline",
        )


# ---------------------------------------------------------------------------
# 9. ablations
# ---------------------------------------------------------------------------
def test_ablations(device):
    section("9. ablation switches")

    variants = {
        "D2 structure only": dict(use_texture=False),
        "D3 texture only": dict(use_structure=False),
        "D4 fusion mean": dict(fusion="mean"),
        "D4 fusion add": dict(fusion="add"),
        "no mask condition": dict(use_mask_condition=False),
        "inverted mask": dict(mask_invert=True),
        "single dilation": dict(structure_dilations=[1]),
    }
    xt, mu, gt, mask, t = make_inputs(device=device)
    for name, override in variants.items():
        net = build(dict(BASE_CFG, **override), device=device).eval()
        with torch.no_grad():
            out = net(xt, mu, t, mask)
        check(out.shape == xt.shape and torch.isfinite(out).all().item(),
              "ablation: {} builds and runs".format(name))

    net = build(dict(BASE_CFG, use_texture=False), device=device)
    check(
        not any("texture" in n for n in net.state_dict()),
        "ablation: D2 has no texture parameters",
    )
    net = build(dict(BASE_CFG, use_structure=False), device=device)
    check(
        not any("structure" in n for n in net.state_dict()),
        "ablation: D3 has no structure parameters",
    )
    net = build(dict(BASE_CFG, use_mask_condition=False), device=device)
    check(
        not any("mask_enc" in n for n in net.state_dict()),
        "ablation: no-mask variant has no mask encoder",
    )
    net = build(dict(BASE_CFG, fusion="mean"), device=device)
    check(
        not any("gate" in n for n in net.state_dict()),
        "ablation: mean fusion has no learnable gate",
    )


# ---------------------------------------------------------------------------
# 10. insertion points
# ---------------------------------------------------------------------------
def test_insertion_points(device):
    section("10. insertion points")

    for point in ["encoder_1", "encoder_2", "bottleneck", "decoder_1", "decoder_2"]:
        net = build(dict(BASE_CFG, insertion_points=[point]), device=device).eval()
        xt, mu, gt, mask, t = make_inputs(device=device)
        with torch.no_grad():
            out = net(xt, mu, t, mask)
        check(out.shape == xt.shape, "insertion: {} works".format(point))

    for bad in ["bottleneck2", "encoder_3", "decoder_0", "middle"]:
        try:
            build(dict(BASE_CFG, insertion_points=[bad]), device=device)
        except ValueError:
            pass
        else:
            raise AssertionError("insertion: invalid point {!r} accepted".format(bad))
    check(True, "insertion: 4 invalid point names rejected")


# ---------------------------------------------------------------------------
# 11. real config report
# ---------------------------------------------------------------------------
def test_real_config(device):
    section("11. real inpainting config (width=64, 4 stages)")

    setting = dict(
        img_channel=3,
        width=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
        use_mask=True,
    )
    base = ConditionalNAFNet(**setting).to(device).eval()
    net = ConditionalNAFNet(**setting, mtsce=BASE_CFG).to(device).eval()

    base_counts = base.parameter_count()
    counts = net.parameter_count()
    print(
        "    baseline params : {:>12,}\n"
        "    MTSCE params    : {:>12,}  (+{:.3f}%)\n"
        "    total params    : {:>12,}".format(
            base_counts["total"],
            counts["mtsce"],
            100.0 * counts["mtsce"] / base_counts["total"],
            counts["total"],
        )
    )
    check(counts["base"] == base_counts["total"], "real: base parameters are untouched by MTSCE")
    check(counts["mtsce"] > 0, "real: MTSCE adds {} parameters".format(counts["mtsce"]))

    # latency / memory at a typical crop size
    xt = torch.rand(1, 3, 256, 256, device=device)
    mu = torch.rand(1, 3, 256, 256, device=device)
    mask = (torch.rand(1, 1, 256, 256, device=device) > 0.5).float()
    t = torch.tensor([50], device=device)

    import time

    def bench(model, warmup=5, iters=10):
        with torch.no_grad():
            for _ in range(warmup):
                model(xt, mu, t, mask)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(iters):
                model(xt, mu, t, mask)
            if device == "cuda":
                torch.cuda.synchronize()
            return (time.time() - start) / iters

    t_base = bench(base)
    t_mtsce = bench(net)
    print(
        "    256x256 forward: baseline {:.1f} ms | +MTSCE {:.1f} ms ({:+.1f}%)".format(
            t_base * 1e3, t_mtsce * 1e3, 100.0 * (t_mtsce - t_base) / t_base
        )
    )
    check(t_mtsce < t_base * 1.5, "real: MTSCE overhead stays below 50% at 256x256")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--real-config", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _VERBOSE[0] = args.verbose

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment")
    device = args.device
    print("device: {} | torch {}".format(device, torch.__version__))

    test_config_validation()
    test_forward_shapes(device)
    test_identity_at_init(device)
    test_mask_alignment(device)
    test_gradients(device)
    test_weight_compat(device)
    test_save_load(device)
    test_disabled(device)
    test_ablations(device)
    test_insertion_points(device)
    if args.real_config:
        test_real_config(device)

    print("\n--------------------------------------------------------------")
    print("ALL CHECKS PASSED ({} assertions)".format(len(_PASS)))


if __name__ == "__main__":
    main()
