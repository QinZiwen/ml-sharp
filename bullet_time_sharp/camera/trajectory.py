# camera/trajectory.py
import numpy as np

def generate_bullet_time_poses(
    target=np.array([0.0, 0.0, 0.0]),
    radius=2.0,
    n_frames=60,
    height=0.0
):
    poses = []

    world_up = np.array([0, 1, 0], dtype=np.float32)

    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames

        offset = np.array([
            radius * np.cos(theta),
            height,
            radius * np.sin(theta)
        ])

        cam_pos = target + offset

        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        R = np.stack([right, up, forward], axis=1)
        T = cam_pos

        poses.append((R, T))

    return poses
