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
        self.vis = vis

    def __call__(self, state):
        tsdf = state.tsdf

        tic = time.time()
        out = predict(tsdf.get_volume(), self.net, self.device)
        out = process(out)
        grasps, scores = select(out)
        if len(grasps) > 0:
            scores, grasps = zip(*sorted(zip(scores, grasps), reverse=True))
            grasps = [from_voxel_coordinates(g, tsdf.voxel_size) for g in grasps]
        toc = time.time() - tic

        vis.quality(out[0], tsdf.voxel_size)

        return grasps, scores, toc


def predict(tsdf_vol, net, device):
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol)
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol


def process(out, threshold=0.90, gaussian_filter_sigma=1.0):
    qual_vol, rot_vol, width_vol = out
    # TODO figure out a more elegant way to handle the borders
    qual_vol[:5, :, :] = 0.0
    qual_vol[-5:, :, :] = 0.0
    qual_vol[:, :5, :] = 0.0
    qual_vol[:, -5:, :] = 0.0
    qual_vol[:, :, :5] = 0.0
    qual_vol[:, :, -5:] = 0.0
    # smooth with Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )
    # threshold
    qual_vol[qual_vol < threshold] = 0.0

    return qual_vol, rot_vol, width_vol


def select(out, max_filter_size=3):
    qual_vol, rot_vol, width_vol = out

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    grasps, scores = [], []
    for index in np.argwhere(mask):
        i, j, k = index
        qual = qual_vol[i, j, k]
        ori = Rotation.from_quat(rot_vol[:, i, j, k])
        pos = np.array([i, j, k], dtype=np.float64)
        width = width_vol[i, j, k]
        grasps.append(Grasp(Transform(ori, pos), width))
        scores.append(qual)

    return grasps, scores
