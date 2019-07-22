import numpy as np

from vgn.utils.transform import Rotation, Transform


def load_extrinsics(fname):
    data = np.loadtxt(fname, delimiter=',')
    extrinsics = []
    for item in data:
        extrinsics.append(Transform(Rotation.from_quat(item[3:7]), item[:3]))
    return extrinsics


def load_grasps(fname):
    data = np.loadtxt(fname, delimiter=',')
    poses = []
    scores = np.empty((data.shape[0], ))
    for i, item in enumerate(data):
        poses.append(Transform(Rotation.from_quat(item[3:7]), item[:3]))
        scores[i] = item[7]
    return poses, scores
