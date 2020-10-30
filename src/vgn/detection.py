import time

import numpy as np
from scipy import ndimage
import torch

from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network


class VGN(object):
    def __init__(self, model_path, rviz=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)
        self.rviz = rviz

    def __call__(self, state):
        tsdf_vol = state.tsdf.get_grid()
        voxel_size = state.tsdf.voxel_size

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        grasps, scores = select(qual_vol.copy(), rot_vol, width_vol)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        if len(grasps) > 0:
            p = np.random.permutation(len(grasps))
            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
            scores = scores[p]

        if self.rviz:
            vis.draw_quality(qual_vol, state.tsdf.voxel_size, threshold=0.01)

        return grasps, scores, toc


def predict(tsdf_vol, net, device):
    assert tsdf_vol.shape == (1, 40, 40, 40)

    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > 0.5
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
    # threshold on grasp quality
    qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    return grasps, scores


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score
