# render/render_frame.py
import numpy as np
import cv2

def render(points_2d, colors, opacity, H, W):
    img = np.zeros((H, W, 3), dtype=np.float32)

    idx = np.argsort(points_2d[:, 2])[::-1]  # back-to-front
    for i in idx:
        x, y, z = points_2d[i]
        x, y = int(x), int(y)
        if 0 <= x < W and 0 <= y < H:
            a = opacity[i]
            img[y, x] = img[y, x] * (1 - a) + colors[i] * a
    return (img * 255).astype(np.uint8)
