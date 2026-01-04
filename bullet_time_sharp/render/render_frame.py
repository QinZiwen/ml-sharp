# render/render_frame.py
import numpy as np
import cv2

def alpha_blend(points_2d, colors, opacity, H, W):
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

def render_base(
    points_2d,     # (N, 3): u, v, z
    colors,        # (N, 3): RGB in [0,1]
    opacity,       # (N,)
    H, W,
    sigma=1.0,     # 高斯大小（像素单位）
    alpha_thresh=1e-4
):
    """
    Full offline 2D Gaussian splatting renderer (CPU).
    """

    img = np.zeros((H, W, 3), dtype=np.float32)
    acc_alpha = np.zeros((H, W), dtype=np.float32)

    # -------- 深度排序：front-to-back --------
    idx = np.argsort(points_2d[:, 2])

    # 预计算
    two_sigma2 = 2.0 * sigma * sigma
    radius = int(3 * sigma)  # 3σ 截断

    for i in idx:
        u, v, z = points_2d[i]

        if z <= 0:
            continue

        a0 = np.clip(opacity[i], 0.0, 1.0)
        if a0 < alpha_thresh:
            continue

        cx = int(round(u))
        cy = int(round(v))

        # 像素窗口
        x0 = max(cx - radius, 0)
        x1 = min(cx + radius + 1, W)
        y0 = max(cy - radius, 0)
        y1 = min(cy + radius + 1, H)

        if x0 >= x1 or y0 >= y1:
            continue

        # -------- 遍历像素块 --------
        for y in range(y0, y1):
            for x in range(x0, x1):
                if acc_alpha[y, x] > 0.995:
                    continue

                dx = x - u
                dy = y - v

                w = np.exp(-(dx*dx + dy*dy) / two_sigma2)
                a = a0 * w

                if a < alpha_thresh:
                    continue

                contrib = (1.0 - acc_alpha[y, x]) * a
                img[y, x] += colors[i] * contrib
                acc_alpha[y, x] += contrib

    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)

def render(
    points_2d,    # (N, 3): x, y, z  (z > 0, camera space)
    colors,       # (N, 3): RGB in [0, 1]
    opacity,      # (N,): base opacity in [0, 1]
    H, W,
    base_sigma=1.2,
    z_ref=2.0
):
    """
    CPU reference renderer for 3D Gaussian Splatting (isotropic, depth-adaptive).

    Args:
        points_2d : Nx3 array, screen-space (x, y, z)
        colors    : Nx3 array, RGB in [0,1]
        opacity   : Nx1 array, alpha in [0,1]
        H, W      : image height / width
        base_sigma: reference sigma at z_ref
        z_ref     : reference depth for sigma scaling

    Returns:
        uint8 image (H, W, 3)
    """

    # framebuffer
    img = np.zeros((H, W, 3), dtype=np.float32)
    acc_alpha = np.zeros((H, W), dtype=np.float32)

    # front-to-back: small z first
    idx = np.argsort(points_2d[:, 2])

    for i in idx:
        x, y, z = points_2d[i]

        if z <= 0:
            continue

        # depth-adaptive sigma (CRITICAL)
        sigma = base_sigma * (z_ref / z)
        sigma = np.clip(sigma, 0.6, 2.5)

        radius = int(3 * sigma)
        if radius < 1:
            radius = 1

        two_sigma2 = 2.0 * sigma * sigma

        # opacity correction: bigger splat -> more transparent
        a0 = opacity[i] / (sigma + 1e-6)
        a0 = np.clip(a0, 0.0, 1.0)

        cx = int(round(x))
        cy = int(round(y))

        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            continue

        for dy in range(-radius, radius + 1):
            py = cy + dy
            if py < 0 or py >= H:
                continue

            for dx in range(-radius, radius + 1):
                px = cx + dx
                if px < 0 or px >= W:
                    continue

                if acc_alpha[py, px] > 0.995:
                    continue

                d2 = dx * dx + dy * dy
                w = np.exp(-d2 / two_sigma2)

                a = a0 * w
                if a < 1e-4:
                    continue

                # front-to-back alpha compositing
                contrib = a * (1.0 - acc_alpha[py, px])
                img[py, px] += colors[i] * contrib
                acc_alpha[py, px] += contrib

    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)

import numpy as np

def render_elliptical(
    points_2d,      # (N, 3): x, y, z   (pixel x,y + depth z)
    colors,         # (N, 3): RGB in [0,1]
    opacity,        # (N,)
    scales,         # (N, 3): sx, sy, sz (from ply, already exp()'ed)
    rotations,      # (N, 3, 3): rotation matrix
    H, W,
    fx, fy
):
    """
    CPU reference anisotropic 3D Gaussian splatting (stable version)
    """

    img = np.zeros((H, W, 3), dtype=np.float32)
    acc_alpha = np.zeros((H, W), dtype=np.float32)

    # front-to-back
    idx = np.argsort(points_2d[:, 2])

    for i in idx:
        px, py, z = points_2d[i]
        if z <= 0:
            continue

        cx = int(round(px))
        cy = int(round(py))
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            continue

        # ---------- 1. 3D covariance ----------
        S = np.diag(scales[i] ** 2)     # variance, not sigma
        R = rotations[i]
        Sigma3D = R @ S @ R.T           # (3,3)

        # ---------- 2. Jacobian (stable approximation) ----------
        # Ignore depth coupling terms (critical for stability)
        J = np.array([
            [fx / z, 0.0,    0.0],
            [0.0,    fy / z, 0.0]
        ], dtype=np.float32)

        Sigma2D = J @ Sigma3D @ J.T     # (2,2)

        # ---------- 3. numerical stabilization ----------
        Sigma2D += np.eye(2) * 1e-6

        inv_Sigma2D = np.linalg.inv(Sigma2D)

        # ---------- 4. bounding box ----------
        eigvals, _ = np.linalg.eigh(Sigma2D)
        max_sigma = np.sqrt(np.max(eigvals))
        radius = int(np.clip(3.0 * max_sigma, 2, 20))

        # ---------- 5. opacity normalization ----------
        # approximate energy conservation
        det = np.linalg.det(Sigma2D)
        if det <= 0:
            continue

        norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
        a0 = opacity[i] * norm
        a0 = np.clip(a0, 0.0, 1.0)

        # ---------- 6. splatting ----------
        for dy in range(-radius, radius + 1):
            iy = cy + dy
            if iy < 0 or iy >= H:
                continue

            for dx in range(-radius, radius + 1):
                ix = cx + dx
                if ix < 0 or ix >= W:
                    continue

                if acc_alpha[iy, ix] > 0.995:
                    continue

                d = np.array([dx, dy], dtype=np.float32)
                w = np.exp(-0.5 * (d @ inv_Sigma2D @ d))

                if w < 1e-5:
                    continue

                a = a0 * w
                contrib = a * (1.0 - acc_alpha[iy, ix])

                img[iy, ix] += colors[i] * contrib
                acc_alpha[iy, ix] += contrib

    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)
