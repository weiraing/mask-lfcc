# -*- coding: utf-8 -*-
"""Draw the paper framework figure: (a) training with Degradation-Aware Latent,
(b) sampling with the MGDC plug-in. Uses a real mask/masked image from the
dark/val_smoke dataset for the visual-inset strip."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join("..", ".."))
import utils as util  # noqa: E402

C_BOX = "#EAF2FB"
C_BOX2 = "#FDF3E3"
C_NET = "#E8F5E9"
C_EDGE = "#2C3E50"
C_ARROW = "#34495E"
C_TRAIN = "#1F6FB2"
C_MGDC = "#C0392B"

fig = plt.figure(figsize=(15.2, 6.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 152)
ax.set_ylim(0, 66)
ax.axis("off")


def box(x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fs=9, lw=1.2, ls="-", weight="normal", tc="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                       fc=fc, ec=ec, lw=lw, linestyle=ls, mutation_scale=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, weight=weight, linespacing=1.4)


def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.4, ls="-", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5)
    ax.add_patch(a)


# ================================================================ panel (a)
ax.text(1, 63.5, r"$\mathbf{(a)}$  Training: Degradation-Aware Latent (one-time)",
        fontsize=11.5, color=C_TRAIN, weight="bold")

box(2, 49, 13, 9, "")
ax.text(8.5, 55.0, r"noisy state $x_t$", ha="center", fontsize=8.5)
ax.add_patch(plt.Rectangle((4.2, 49.8), 8.6, 3.6, fc="#9aa7b3", ec=C_EDGE, lw=0.8))
box(17, 49, 13, 9, "")
ax.text(23.5, 55.0, r"masked image $\mu$", ha="center", fontsize=8.5)
ax.add_patch(plt.Rectangle((19.2, 49.8), 8.6, 3.6, fc="#cfd8e0", ec=C_EDGE, lw=0.8))
ax.add_patch(plt.Rectangle((24.9, 49.8), 2.9, 3.6, fc="white", ec=C_EDGE, lw=0.8))
box(32, 49, 12, 9, "")
ax.text(38, 55.2, r"mask $M$" + "\n(1 = hole)", ha="center", fontsize=8.5)
ax.add_patch(plt.Rectangle((34.2, 49.8), 7.6, 3.6, fc="black", ec=C_EDGE, lw=0.8))
ax.add_patch(plt.Rectangle((39.4, 49.8), 2.4, 3.6, fc="white", ec=C_EDGE, lw=0.8))

box(7, 35, 8, 5, r"$x_t-\mu$", fc="white", fs=9)
box(19, 35, 18, 5, "concat  (6+1 = 7 channels)", fc="white", fs=9)
arrow(8.5, 49, 10.2, 40.0)
arrow(23.5, 49, 25, 40.0)
arrow(38, 49, 32.5, 40.0)
arrow(15, 37.5, 19, 37.5)

box(12, 21.5, 32, 8.5, r"score network  $\epsilon_\theta(x_t,\ \mu,\ M,\ t)$" + "\nConditionalNAFNet (trainable)",
    fc=C_NET, fs=9, weight="bold")
arrow(28, 35, 28, 30.0)

box(12, 8, 32, 8, "matching loss (max-likelihood)\n" + r"$\mathbb{E}[x_{t-1}|x_t]\ \leftrightarrow\ $optimum step toward $x_0$",
    fc="white", fs=8.5)
arrow(28, 21.5, 28, 16.0)
ax.text(28, 4.6, r"weights updated by back-propagation", ha="center", fontsize=8,
        style="italic", color="#555555")

# ================================================================ panel (b)
ax.text(63, 63.5, r"$\mathbf{(b)}$  Sampling: Mask-Guided Data Consistency (training-free plug-in)",
        fontsize=11.5, color=C_MGDC, weight="bold")

# shared weights link (a)->(b)
arrow(44, 25.5, 66, 48.5, color="#888888", ls=(0, (3, 3)), rad=-0.3)
ax.text(53.5, 39.5, "frozen\nweights", fontsize=7.8, color="#777777", ha="center", style="italic")

box(50, 49, 11, 8, r"$x_T\!\sim\!\mathcal{N}$", fc="white", fs=9.5)
arrow(61, 53, 66, 53)

box(66, 48, 26, 10, r"$\hat\epsilon=\epsilon_\theta(x_t,\mu,M,t)$" + "\n(one call per step, frozen)",
    fc=C_NET, fs=8.5, weight="bold", ls=(0, (4, 2)))

box(97, 48, 22, 10, r"posterior mean $\hat{x}_0$" + "\n" + r"$=(x_t\!-\!\mu\!-\!\bar\sigma_t\hat\epsilon)A_t+\mu$",
    fc="white", fs=8)
arrow(92, 53, 97, 53)

# MGDC module
box(58, 12, 48, 31, "", fc=C_BOX2, ec=C_MGDC, lw=1.6)
ax.text(82, 39.8, "MGDC  module", ha="center", fontsize=9.5, weight="bold", color=C_MGDC)

box(60.5, 29.5, 34, 7, r"$\mathbf{(i)}$  observed pinning" + "\n" + r"$\hat{x}_0\!\leftarrow\!\bar{M}\odot\mu+M\odot\hat{x}_0$",
    fc="white", ec=C_MGDC, fs=7.8)
box(60.5, 20.5, 34, 7, r"$\mathbf{(ii)}$  support-weighted low-frequency" + "\n" + r"$\hat{x}_0\!\leftarrow\!\hat{x}_0+\lambda_t M\odot W\odot(L_{\mathrm{obs}}-G_\sigma\!*\!\hat{x}_0)$",
    fc="white", ec=C_MGDC, fs=7.4)
box(60.5, 13.5, 34, 5.5, r"$\mathbf{(iii)}$  re-derive noise  $\hat\epsilon'=(x_t-\mu-(\hat{x}_0-\mu)/A_t)/\bar\sigma_t$",
    fc="white", ec=C_MGDC, fs=7.0)

arrow(102, 48, 80, 36.8, color=C_ARROW)          # posterior mean -> (i)
arrow(77.5, 29.5, 77.5, 27.5, color=C_MGDC)      # (i) -> (ii)
arrow(77.5, 20.5, 77.5, 19.0, color=C_MGDC)      # (ii) -> (iii)

# inset images (real data) inside MGDC box
def imread_u(p):
    import cv2
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

base = os.path.join("datasets", "dark", "val_smoke")
m_img = imread_u(os.path.join(base, "masks", "0001_mask.png"))
lq_img = imread_u(os.path.join(base, "masked_images", "0001.png"))
m_t = torch.from_numpy(np.max(m_img, 2)[None, None] / 255.0).float()
obs = 1.0 - m_t
with torch.no_grad():
    sup = util.gaussian_blur(obs, 4.0)
    W = sup / sup.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
for k, (cap, im) in enumerate([(r"mask $M$", m_t), (r"support $W$", W)]):
    x0 = 97.0 + k * 4.6
    ax.imshow(im[0, 0].numpy(), cmap="gray", extent=(x0, x0 + 4.0, 15.5, 31.5),
              vmin=0, vmax=1, zorder=5, interpolation="bilinear")
    ax.add_patch(plt.Rectangle((x0, 15.5), 4.0, 16.0, fill=False, ec=C_EDGE, lw=0.7, zorder=6))
    ax.text(x0 + 2.0, 13.6, cap, ha="center", fontsize=6.8, zorder=6)
ax.text(102.8, 33.4, "guidance\nevidence", ha="center", fontsize=7, style="italic", color="#555555")

# reverse SDE step + final composite
box(111, 12.5, 37, 7.5, "reverse SDE step (unchanged law)\n" + r"$x_{t-1}=\mathrm{Step}(x_t,\ -\hat\epsilon'/\bar\sigma_t,\ t)+\sigma_t\sqrt{\Delta t}\,z$",
    fc="white", fs=7.6)
arrow(94.5, 16.2, 111, 16.2, color=C_MGDC)

box(111, 2.5, 37, 6, r"final composite  $x_0=\bar{M}\odot\mu+M\odot x_1$",
    fc="#F5EEF8", ec=C_EDGE, fs=8)
arrow(129.5, 12.5, 129.5, 8.5)

# loop: reverse step -> back to network, routed over the top
ax.plot([129.5, 129.5, 79], [20.0, 59.5, 59.5], color=C_ARROW, lw=1.2, ls=(0, (4, 2)), zorder=1)
arrow(79, 59.5, 79, 58.0, color=C_ARROW, lw=1.2, ls=(0, (4, 2)))
ax.text(106, 61.2, r"$t: T\rightarrow 1$", fontsize=9, color=C_ARROW, style="italic", ha="center")

# legend
handles = [
    Line2D([0], [0], color=C_TRAIN, lw=2, label="training stage"),
    Line2D([0], [0], color=C_MGDC, lw=2, label="MGDC (sampling, no retraining)"),
    Line2D([0], [0], color="#888888", lw=1.4, ls=(0, (3, 3)), label="frozen path"),
]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.005),
          frameon=False, fontsize=8)

out = os.path.join("..", "..", "..", "figs", "mgdc_framework.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, facecolor="white")
print("saved:", os.path.abspath(out))
