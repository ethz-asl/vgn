import enum

import numpy as np
from scipy import ndimage
import torch

from vgn.grasp import Grasp
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network


class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed


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
    # smooth with Gaussian
    qual_vol = ndimage.gaussian_filter(qual_vol, sigma=gaussian_filter_sigma)
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
    grasps, scores = np.asarray(grasps), np.asarray(scores)

    if len(grasps) > 0:
        indices = np.argsort(-scores)
        grasps = grasps[indices]
        scores = scores[indices]

    return grasps, scores


def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width)
