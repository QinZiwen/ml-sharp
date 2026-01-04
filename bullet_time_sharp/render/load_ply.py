# render/load_ply.py
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def quaternion_to_rotation_matrix(q):
    """
    q: (..., 4) quaternion, assumed format (w, x, y, z)
    return: (..., 3, 3) rotation matrices
    """
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.zeros(q.shape[:-1] + (3, 3), dtype=np.float32)

    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)

    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)

    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def load_gaussians(ply_path):
    """
    Load 3D Gaussians from Sharp / 3DGS style PLY.

    Returns:
        xyz        : (N,3)
        colors     : (N,3) in [0,1]
        opacity    : (N,)
        scales     : (N,3)
        rotations  : (N,3,3)
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    # -------- positions --------
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # -------- scales (exp because stored in log-space in many 3DGS) --------
    scales = np.stack(
        [v["scale_0"], v["scale_1"], v["scale_2"]],
        axis=1
    ).astype(np.float32)

    # Sharp / 3DGS: scale is usually log-space
    scales = np.exp(scales)

    # -------- opacity (sigmoid inverse stored) --------
    opacity = v["opacity"].astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-opacity))

    # -------- color (DC term only) --------
    colors = np.stack(
        [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]],
        axis=1
    ).astype(np.float32)

    # clamp for safety
    colors = 0.5 + colors * 0.282095
    colors = np.clip(colors, 0.0, 1.0)

    # -------- rotation (quaternion) --------
    q = np.stack(
        [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
        axis=1
    ).astype(np.float32)

    rotations = quaternion_to_rotation_matrix(q)

    return xyz, colors, opacity, scales, rotations

def load_gaussians_base(ply_path):
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
    opacity = v['opacity']
    color = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)
    return xyz, color, opacity

def estimate_target_from_ply(vertices, opacity, opacity_thresh=0.01):
    """
    vertices: (N, 3)
    opacity:  (N,)
    """

    mask = opacity > opacity_thresh
    pts = vertices[mask]
    w   = opacity[mask]

    w = w / np.sum(w)

    target = np.sum(pts * w[:, None], axis=0)
    return target

def compute_pca_axes(points):
    center = points.mean(axis=0)
    pts = points - center

    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 从大到小排序
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]   # (3,3)

    return center, eigvals, eigvecs

def visualize_pca_axes(
    points,
    center,
    axes,
    eigvals,
    scale=1
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # -------- 点云（抽样后） --------
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=1, c='gray', alpha=0.2
    )

    # -------- 目标中心 --------
    ax.scatter(
        center[0], center[1], center[2],
        c='k', s=60, label='Center'
    )

    # -------- PCA 三轴 --------
    colors = ['r', 'g', 'b']
    names = [
        'Principal Axis (PCA-0)',
        'Secondary Axis (PCA-1)',
        'Minor Axis (PCA-2)'
    ]

    for i in range(3):
        v = axes[:, i] * scale
        ax.plot(
            [center[0], center[0] + v[0]],
            [center[1], center[1] + v[1]],
            [center[2], center[2] + v[2]],
            color=colors[i],
            linewidth=4,
            label=f'{names[i]}'
        )

        # 轴端文字标注
        ax.text(
            center[0] + v[0],
            center[1] + v[1],
            center[2] + v[2],
            f'{names[i]}',
            color=colors[i],
            fontsize=10
        )

    # -------- 世界坐标轴标签 --------
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_zlabel('World Z')

    # -------- 视角 & 比例 --------
    ax.set_box_aspect([1, 1, 1])  # 等比例
    ax.view_init(elev=20, azim=45)

    ax.legend()
    plt.tight_layout()
    plt.show()

def random_sample(points, max_points=10000):
    N = points.shape[0]
    if N <= max_points:
        return points

    idx = np.random.choice(N, max_points, replace=False)
    return points[idx]

def filter_by_radius(points, keep_ratio=0.7):
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    thresh = np.quantile(dist, keep_ratio)
    return points[dist < thresh]

def filter_by_density(points, k=20, keep_ratio=0.7):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    dists, _ = nbrs.kneighbors(points)
    density = 1.0 / (dists[:, -1] + 1e-6)

    thresh = np.quantile(density, 1 - keep_ratio)
    return points[density > thresh]

if __name__ == "__main__":
    points, _, _ = load_gaussians("data/xiaohei.ply")
    points = filter_by_radius(points, keep_ratio=0.6)
    # points = filter_by_density(points)
    center, eigvals, axes = compute_pca_axes(points)
    points_vis = random_sample(points, max_points=5000)
    visualize_pca_axes(
        points_vis,
        center,
        axes,
        eigvals,
        scale=0.5
    )

