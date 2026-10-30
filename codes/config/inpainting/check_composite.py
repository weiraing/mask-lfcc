import os
import cv2
import numpy as np


def imread_u(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


root = os.path.abspath(
    os.path.join("..", "..", "..", "results", "inpainting", "smoke_mgdc", "Val_Dataset")
)
print("result dir:", root)
for img in ["0001", "0002", "0003"]:
    gt = imread_u(os.path.join("datasets/dark/val_smoke/images", img + ".png")).astype(np.float32) / 255.0
    m = imread_u(os.path.join("datasets/dark/val_smoke/masks", img + "_mask.png"))
    m = np.max(m, axis=2, keepdims=True) / 255.0
    obs = m < 0.5
    out = imread_u(os.path.join(root, img + ".png")).astype(np.float32) / 255.0
    obs3 = np.repeat(obs, 3, axis=2)  # mask is 1-channel, images are 3-channel
    err = np.abs(out - gt)[obs3].max()
    hole = (m > 0.5).mean()
    print(f"{img}: max|out-GT| on observed px = {err:.4f}   hole coverage {hole*100:.1f}%")
