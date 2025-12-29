# render/load_ply.py
import numpy as np
from plyfile import PlyData

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

