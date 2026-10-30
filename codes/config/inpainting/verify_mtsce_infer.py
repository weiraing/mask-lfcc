"""Deterministic inference check: baseline vs MTSCE with the *same* checkpoint.

The reverse SDE injects Gaussian noise, so two runs of ``test.py`` are never
bit-identical unless the RNG is re-seeded right before sampling. This script
removes that randomness:

  * both networks load the very same checkpoint (MTSCE keeps its zero-init),
  * the same initial noisy state is used for both,
  * ``torch.manual_seed`` is called immediately before each sampling loop,
  * the two outputs are compared tensor-wise.

A bit-identical result proves that enabling MTSCE does not alter the inference
path of an already trained model, and that the training / inference structures
agree (the module is executed in both, it simply contributes a zero residual
before fine-tuning).

Run from ``codes/config/inpainting``:

    python verify_mtsce_infer.py
    python verify_mtsce_infer.py --ckpt path/to/xxx.pth --data datasets/dark/val_smoke
"""

import argparse
import os
import sys
import time

import cv2  # noqa: F401  (must be imported before torchvision on Windows)
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ``codes`` itself must be importable for ``import utils`` / ``data``
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
)

import utils as util  # noqa: E402
from data.util import read_img  # noqa: E402
from models.base_model import load_state_dict_compatible  # noqa: E402
from models.modules.DenoisingNAFNet_arch import ConditionalNAFNet  # noqa: E402

DEFAULT_CKPT = (
    r"C:\Users\rain\Desktop\任务二\our_model_30000\our_model_30000"
    r"\refusion_mask_dark\models\lastest_EMA.pth"
)

SETTING = dict(
    img_channel=3,
    width=64,
    enc_blk_nums=[1, 1, 1, 28],
    middle_blk_num=1,
    dec_blk_nums=[1, 1, 1, 1],
    use_mask=True,
)

MTSCE_CFG = dict(
    enabled=True,
    insertion_points=["bottleneck"],
    hidden_ratio=0.5,
    structure_dilations=[1, 2, 3],
    texture_kernel_size=3,
    use_structure=True,
    use_texture=True,
    fusion="adaptive",
    use_mask_condition=True,
    residual_scale_init=0.1,
)


def load_triplet(root, name):
    gt = read_img(None, os.path.join(root, "images", name))
    lq = read_img(None, os.path.join(root, "masked_images", name))
    mask_path = os.path.join(root, "masks", name.replace(".png", "_mask.png"))
    if not os.path.exists(mask_path):
        mask_path = os.path.join(root, "masks", name)
    mask = read_img(None, mask_path)

    def to_tensor(x, rgb=True):
        if x.ndim == 3 and x.shape[2] == 3 and rgb:
            x = x[:, :, [2, 1, 0]]
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float().unsqueeze(0)

    gt_t = to_tensor(gt)
    lq_t = to_tensor(lq)
    mask_t = (to_tensor(mask, rgb=False) > 0.5).float()
    return gt_t, lq_t, mask_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--data", default="datasets/dark/val_smoke")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("device:", device)
    print("checkpoint:", args.ckpt)

    torch.manual_seed(0)
    base = ConditionalNAFNet(**SETTING).to(device).eval()
    net = ConditionalNAFNet(**SETTING, mtsce=MTSCE_CFG).to(device).eval()

    state = torch.load(args.ckpt, map_location="cpu")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    base.load_state_dict(state, strict=True)
    report = load_state_dict_compatible(
        net, state, allow_missing_prefixes=("mtsce.",), allow_unexpected_prefixes=("mtsce.",)
    )
    print(
        "compatible load: {} base tensors restored, {} MTSCE tensors freshly initialised".format(
            report["loaded"], len(report["missing_allowed"])
        )
    )
    counts = net.parameter_count()
    print("params: base {:,} + MTSCE {:,} = {:,}".format(counts["base"], counts["mtsce"], counts["total"]))

    sde = util.IRSDE(max_sigma=50, T=100, schedule="cosine", eps=0.005, device=device)

    names = sorted(f for f in os.listdir(os.path.join(args.data, "images")) if f.endswith(".png"))
    print("\n{} image(s) from {}".format(len(names), args.data))

    all_equal = True
    for name in names:
        gt, lq, mask = load_triplet(args.data, name)
        gt, lq, mask = gt.to(device), lq.to(device), mask.to(device)
        hole = float(mask.mean().item())

        sde.set_mu(lq)
        sde.set_mask(mask)
        # identical starting state for both runs
        torch.manual_seed(args.seed)
        noisy = sde.noise_state(lq)

        outs, times = [], []
        for tag, model in (("baseline", base), ("+MTSCE", net)):
            sde.set_model(model)
            torch.manual_seed(args.seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(args.seed)
                torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                out = sde.reverse_sde(noisy.clone(), T=100)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)
            outs.append(out)

        equal = torch.equal(outs[0], outs[1])
        all_equal = all_equal and equal
        diff = (outs[0] - outs[1]).abs().max().item()

        psnr = []
        for out in outs:
            o = out.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            g = gt.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            psnr.append(util.calculate_psnr(o * 255, g * 255))

        print(
            "  {} (hole {:.1%}) | bit-identical: {} (max |diff| {:.1e}) | "
            "PSNR {:.4f} / {:.4f} dB | {:.2f}s / {:.2f}s".format(
                name, hole, equal, diff, psnr[0], psnr[1], times[0], times[1]
            )
        )

    print(
        "\nRESULT: {} -- enabling a zero-initialised MTSCE leaves the inference "
        "output unchanged.".format("PASS" if all_equal else "FAIL")
    )
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
