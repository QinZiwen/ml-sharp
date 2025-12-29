# render/project_gaussians.py
import numpy as np

def project(points, R, T, fx, fy, cx, cy):
    pts_cam = (R.T @ (points - T).T).T
    z = pts_cam[:, 2] + 1e-6

    u = fx * pts_cam[:, 0] / z + cx
    v = fy * pts_cam[:, 1] / z + cy
    return np.stack([u, v, z], axis=1)
