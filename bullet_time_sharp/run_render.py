# run_render.py
from camera.trajectory import generate_bullet_time_poses
from render.load_ply import load_gaussians, estimate_target_from_ply
from render.project_gaussians import project
from render.render_frame import render
from compose_video import compose_video
import cv2

xyz, color, opacity = load_gaussians("data/xiaohei.ply")
target = estimate_target_from_ply(xyz, opacity)
poses = generate_bullet_time_poses(
    target=target,
    radius=2.0,
    n_frames=60,
    height=0.2
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
