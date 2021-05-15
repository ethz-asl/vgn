from dataclasses import dataclass

import numpy as np
from scipy import ndimage
import torch

from robot_tools.spatial import Rotation, Transform
from vgn.grasp import *
from vgn.networks import load_network


@dataclass
class Output:
    qual: np.ndarray
    rot: np.ndarray
    width: np.ndarray


class VGN:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)

    def predict(self, tsdf, sigma=1.0):
        assert tsdf.shape == (40, 40, 40)
        tsdf_in = torch.from_numpy(tsdf)[None, None, :].to(self.device)
        with torch.no_grad():
            qual, rot, width = self.net(tsdf_in)
        qual = qual.cpu().squeeze().numpy()
        rot = rot.cpu().squeeze().numpy()
        width = width.cpu().squeeze().numpy()

        # Smooth quality volume with a Gaussian
        qual = ndimage.gaussian_filter(qual, sigma, mode="nearest")

        # Mask out voxels too far away from the surface
        outside_voxels = tsdf > 0.5
        inside_voxels = np.logical_and(1e-3 < tsdf, tsdf < 0.5)
        valid_voxels = ndimage.morphology.binary_dilation(
            outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
        )
        qual[valid_voxels == False] = 0.0

        return Output(qual, rot, width)


def compute_grasps(
    voxel_size,
    out,
    threshold=0.9,
    max_filter_size=4.0,
    score_fn=lambda g: g.q,
):
    # Threshold on grasp quality
    qual = out.qual.copy()
    qual[qual < threshold] = 0.0

    # Non maximum suppression
    max = ndimage.maximum_filter(qual, size=max_filter_size)
    qual = np.where(qual == max, qual, 0.0)
    mask = np.where(qual, 1.0, 0.0)

    # Construct grasp list
    grasps = [select_at(out, i) for i in np.argwhere(mask)]

    # Compute Cartesian coordinates
    grasps = np.asarray([from_voxel_coordinates(g, voxel_size) for g in grasps])

    # Sort according to the score function
    scores = np.asarray([score_fn(g) for g in grasps])
    ind = np.argsort(-scores)

    return grasps[ind]


def select_at(out, index):
    i, j, k = index
    q = out.qual[i, j, k]
    ori = Rotation.from_quat(out.rot[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = out.width[i, j, k]
    return Grasp(Transform(ori, pos), width, q)
