"""A/B check: does the LFCC final composite actually help?

The +0.42 dB in LFCC_ANALYSIS.md was measured offline on saved PNGs, so it has
never been confirmed through the real sampler.  This script runs the same
frozen denoiser twice on the *same* initial state -- once with the corrector
attached (in-sampler gate + final composite) and once without -- and reports
the mean PSNR of both arms, plus the PSNR inside the hole, where 97% of the
error lives.

If ``delta`` is not clearly positive here, the problem is the mask semantics or
the fill, not the training, and training the gate is pointless.

    cd codes/config/inpainting
    python check_lfcc_gain.py --n 30            # ~5 min on one GPU
    python check_lfcc_gain.py --n 30 -opt=options/test/refusion_mask_lfcc.yml
"""

import argparse
import itertools

import numpy as np
import torch

import options as option
import utils as util
from data import create_dataloader, create_dataset
from models import create_model


def psnr_np(a, b):
    """PSNR of two float arrays in [0, 1]."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", default="options/test/refusion_mask_lfcc_fixed.yml")
    parser.add_argument("--n", type=int, default=30, help="how many val images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    opt = option.dict_to_nonedict(option.parse(args.opt, is_train=False))
    dataset_opt = list(opt["datasets"].values())[0]
    test_set = create_dataset(dataset_opt)
    loader = create_dataloader(test_set, dataset_opt)

    model = create_model(opt)
    device = model.device
    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )
    sde.set_model(model.model)

    corrector = getattr(model, "corrector", None)
    if corrector is None:
        raise SystemExit(
            "this config has no corrector, so there is nothing to compare; "
            "use options/test/refusion_mask_lfcc_fixed.yml"
        )

    print(
        "corrector: trainable=%s | alpha_final=%s | steps=%d | t_range=%s"
        % (
            getattr(corrector.net, "trainable", None),
            "%.4f" % torch.sigmoid(corrector.net.alpha_final_raw).item()
            if getattr(corrector.net, "alpha_final_raw", None) is not None
            else "%.4f (const)" % getattr(corrector.net, "_alpha_final_const", float("nan")),
            corrector.steps,
            list(corrector.apply_t_range),
        )
    )
    print("%-6s %10s %10s %8s | %10s %10s %8s" % (
        "img", "base", "corr", "delta", "base_hole", "corr_hole", "d_hole"))

    rows = []
    for i, data in itertools.islice(enumerate(loader), args.n):
        GT, LQ, mask = data["GT"], data["LQ"], data.get("mask", None)
        # the reverse SDE injects Gaussian noise, so both arms must start from
        # the identical state -- otherwise the comparison is pure noise
        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        state = sde.noise_state(LQ)

        outs = {}
        for arm in ("base", "corr"):
            model.corrector = corrector if arm == "corr" else None
            if model.corrector is not None:
                model.corrector.net.clear_cache()
            model.feed_data(state.clone(), LQ, GT, mask)
            model.test(sde)
            outs[arm] = model.get_current_visuals()["Output"].float().cpu()
        model.corrector = corrector

        # get_current_visuals already returns a single [C, H, W] image
        gt_img = util.tensor2img(GT[0].float().cpu()) / 255.0
        base_img = util.tensor2img(outs["base"]) / 255.0
        corr_img = util.tensor2img(outs["corr"]) / 255.0

        if mask is not None:
            m = mask[0].float().cpu().numpy()
            hole = (m[0] if m.ndim == 3 else m) > 0.5
        else:
            hole = np.zeros(gt_img.shape[:2], dtype=bool)

        base_p, corr_p = psnr_np(base_img, gt_img), psnr_np(corr_img, gt_img)
        if hole.any():
            base_h = psnr_np(base_img[hole], gt_img[hole])
            corr_h = psnr_np(corr_img[hole], gt_img[hole])
        else:
            base_h = corr_h = float("nan")

        rows.append((base_p, corr_p, base_h, corr_h, hole.mean()))
        print("%-6d %10.4f %10.4f %+8.4f | %10.4f %10.4f %+8.4f" % (
            i, base_p, corr_p, corr_p - base_p, base_h, corr_h, corr_h - base_h))

    a = np.array(rows, dtype=np.float64)
    d = a[:, 1] - a[:, 0]
    dh = a[:, 3] - a[:, 2]
    ok = np.isfinite(dh)

    print("\n---- %d images | hole coverage %.1f%% ----" % (len(a), a[:, 4].mean() * 100))
    print("overall PSNR : base %.4f -> corr %.4f   delta %+0.4f  (median %+0.4f)" % (
        a[:, 0].mean(), a[:, 1].mean(), d.mean(), np.median(d)))
    print("hole    PSNR : base %.4f -> corr %.4f   delta %+0.4f  (median %+0.4f)" % (
        a[ok, 2].mean(), a[ok, 3].mean(), dh[ok].mean(), np.median(dh[ok])))
    print("improved on %d/%d images (overall), %d/%d (hole)" % (
        int((d > 0).sum()), len(d), int((dh[ok] > 0).sum()), int(ok.sum())))

    if d.mean() < 0.05:
        print(
            "\n!! the composite is NOT paying off. Before touching the training, "
            "check:\n"
            "   1) mask polarity -- mask must be 1 inside the hole (see "
            "check_env.py, it verifies GT/LQ/mask pairing)\n"
            "   2) the fill: run verify_lfcc.py and print fill vs LQ outside the hole\n"
            "   3) that dataroot_LQ really is the masked image the baseline saw"
        )


if __name__ == "__main__":
    main()
