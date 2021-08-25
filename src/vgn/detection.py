from dataclasses import dataclass

import numpy as np
from scipy import ndimage
import torch

from robot_helpers.spatial import Rotation, Transform
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


def select_local_maxima(
    voxel_size,
    out,
    threshold=0.9,
    max_filter_size=3.0,
):
    max = ndimage.maximum_filter(out.qual, size=max_filter_size)
    index_list = np.argwhere(np.logical_and(out.qual == max, out.qual > threshold))
    grasps = [select_at(out, i) for i in index_list]
    return [from_voxel_coordinates(g, voxel_size) for g in grasps]


def select_grid(voxel_size, out, threshold=0.9, step=2):
    grasps = []
    N = out.qual.shape[0]
    for i in range(0, N, step):
        for j in range(0, N, step):
            for k in range(0, N, step):
                if out.qual[i, j, k] > threshold:
                    grasps.append(select_at(out, (i, j, k)))
    return [from_voxel_coordinates(g, voxel_size) for g in grasps]


def select_at(out, index):
    i, j, k = index
    ori = Rotation.from_quat(out.rot[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = out.width[i, j, k]
    quality = out.qual[i, j, k]
    return Grasp(Transform(ori, pos), width, quality)
