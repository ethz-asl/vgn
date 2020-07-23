import time

import numpy as np
from scipy import ndimage
import torch

from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network


class VGN(object):
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)

    def __call__(self, state):
        tsdf_vol = state.tsdf.get_volume()
        voxel_size = state.tsdf.voxel_size

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        grasps, scores = select(qual_vol, rot_vol, width_vol)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        if len(grasps) > 0:
            p = np.random.permutation(len(grasps))
            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
            scores = scores[p]

        vis.draw_quality(qual_vol, voxel_size)

        return grasps, scores, toc


def predict(tsdf_vol, net, device):
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol)
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    threshold=0.90,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth with Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > 0.5
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    # vis.draw_volume(valid_voxels.astype(np.float32), 0.0075)
    qual_vol[valid_voxels == False] = 0.0

    # threshold on grasp quality
    qual_vol[qual_vol < threshold] = 0.0

    # reject widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, max_filter_size=3):

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

