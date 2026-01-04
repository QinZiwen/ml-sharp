# run_render.py
from camera.trajectory import generate_bullet_time_poses
from render.load_ply import load_gaussians, filter_by_radius, compute_pca_axes, random_sample, visualize_pca_axes
from render.project_gaussians import project
from render.render_frame import render, render_elliptical
from compose_video import compose_video
import cv2
import multiprocessing as mp
import numpy as np

def render_pose_range(
    pose_ids,
    poses,
    xyz,
    colors,
    opacity,
    scales,
    rotations,
    H, W,
    fx, fy, cx, cy,
    render_fn
):
    from render.project_gaussians import project

    for i in pose_ids:
        R, T = poses[i]

        pts2d = project(xyz, R, T, fx, fy, cx, cy)
        img = render_fn(
            pts2d, colors, opacity,
            scales, rotations,
            H, W, fx, fy
        )

        filename = f"frames/{i:04d}.png"
        print(f"[PID {i}] save {filename}")
        cv2.imwrite(filename, img)

def render_all_frames_parallel(
    poses,
    xyz,
    colors,
    opacity,
    scales,
    rotations,
    H, W,
    fx, fy, cx, cy,
    render_fn,
    n_workers=6
):
    ctx = mp.get_context("spawn")

    n_frames = len(poses)
    pose_indices = np.arange(n_frames)
    chunks = np.array_split(pose_indices, n_workers)

    processes = []

    for chunk in chunks:
        if len(chunk) == 0:
            continue

        p = ctx.Process(
            target=render_pose_range,
            args=(
                chunk,
                poses,
                xyz,
                colors,
                opacity,
                scales,
                rotations,
                H, W,
                fx, fy, cx, cy,
                render_fn
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    xyz, colors, opacity, scales, rotations = load_gaussians("data/xiaohei.ply")
    points = filter_by_radius(xyz, keep_ratio=0.6)
    center, eigvals, axes = compute_pca_axes(points)
    points_vis = random_sample(points, max_points=5000)
    # visualize_pca_axes(
    #     points_vis,
    #     center,
    #     axes,
    #     eigvals,
    #     scale=1
    # )

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

    # for i, (R, T) in enumerate(poses):
    #     pts2d = project(xyz, R, T, fx, fy, cx, cy)
    #     img = render_elliptical(pts2d, colors, opacity, scales, rotations, H, W, fx, fy)
    #     filename = f"frames/{i:04d}.png"
    #     print("save to ", filename)
    #     cv2.imwrite(filename, img)
    render_all_frames_parallel(
        poses,
        xyz,
        colors,
        opacity,
        scales,
        rotations,
        H, W,
        fx, fy, cx, cy,
        render_fn=render_elliptical,
        n_workers=6   # M2 Air 推荐 6
    )

    compose_video()
