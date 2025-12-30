# render/render_frame.py
import numpy as np
import cv2

def render(points_2d, colors, opacity, H, W):
    img = np.zeros((H, W, 3), dtype=np.float32)
    acc_alpha = np.zeros((H, W), dtype=np.float32)

    idx = np.argsort(points_2d[:, 2])  # front-to-back

    for i in idx:
        x, y, z = points_2d[i]
        x, y = int(x), int(y)

        if not (0 <= x < W and 0 <= y < H):
            continue

        # 规范化 opacity（关键）
        a = opacity[i]
        a = np.clip(a, 0.0, 1.0)

        # early termination
        if acc_alpha[y, x] > 0.995:
            continue

        # alpha compositing
        w = a * (1.0 - acc_alpha[y, x])
        img[y, x] += colors[i] * w
        acc_alpha[y, x] += w

    # clamp + 转 uint8
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)
