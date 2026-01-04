# camera/trajectory.py
import numpy as np

def generate_bullet_time_poses(
    target,
    main_axis,
    radius=2.0,
    n_frames=60
):
    poses = []

    # 保证单位化
    main_axis = main_axis / np.linalg.norm(main_axis)

    # -------- 构造与主轴正交的旋转平面 --------
    tmp = np.array([0, 1, 0], dtype=np.float32)
    if abs(np.dot(tmp, main_axis)) > 0.9:
        tmp = np.array([1, 0, 0], dtype=np.float32)

    u = np.cross(main_axis, tmp)
    u = u / np.linalg.norm(u)

    v = np.cross(main_axis, u)  # 已经是单位向量

    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames

        # -------- 相机位置：绕 PCA 主轴画圆 --------
        cam_pos = target + radius * (
            np.cos(theta) * u +
            np.sin(theta) * v
        )

        # -------- 相机朝向目标 --------
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)

        # -------- 使用 PCA 主轴作为 up --------
        right = np.cross(main_axis, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        R = np.stack([right, up, forward], axis=1)
        T = cam_pos

        poses.append((R, T))

    return poses
