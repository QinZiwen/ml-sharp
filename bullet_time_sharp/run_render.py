# run_render.py
from camera.trajectory import generate_bullet_time_poses
from render.load_ply import load_gaussians, filter_by_radius, compute_pca_axes, random_sample, visualize_pca_axes
from render.project_gaussians import project
from render.render_frame import render
from compose_video import compose_video
import cv2

xyz, color, opacity = load_gaussians("data/xiaohei.ply")
points = filter_by_radius(xyz, keep_ratio=0.6)
center, eigvals, axes = compute_pca_axes(points)
points_vis = random_sample(points, max_points=5000)
visualize_pca_axes(
    points_vis,
    center,
    axes,
    eigvals,
    scale=1
)

main_axis = axes[:, 0]   # PCA 主轴
poses = generate_bullet_time_poses(
    target=center,
    main_axis=main_axis,
    radius=2.0,
    n_frames=60
)

H, W = 256, 256
fx = fy = 300
cx, cy = W // 2, H // 2

for i, (R, T) in enumerate(poses):
    pts2d = project(xyz, R, T, fx, fy, cx, cy)
    img = render(pts2d, color, opacity, H, W)
    filename = f"frames/{i:04d}.png"
    print("save to ", filename)
    cv2.imwrite(filename, img)

compose_video()
