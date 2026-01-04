# render/load_ply.py
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def load_gaussians(ply_path):
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

