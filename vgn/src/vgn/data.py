import json
import os

import numpy as np

from vgn import utils
from vgn.perception import camera
from vgn.utils.transform import Transform


def load_scene_data(dirname):
    intrinsic = _load_intrinsic(dirname)
    extrinsics, images = _load_images(dirname)
    poses, scores = _load_grasps(dirname)
    sample = {
        'intrinsic': intrinsic,
        'extrinsics': extrinsics,
        'images': images,
        'poses': poses,
        'scores': scores
    }
    return sample


def _load_intrinsic(dirname):
    fname = os.path.join(dirname, 'intrinsic.json')
    return camera.PinholeCameraIntrinsic.from_json(fname)


def _load_images(dirname):
    with open(os.path.join(dirname, 'viewpoints.json'), 'rb') as fp:
        viewpoints = json.load(fp)

    imgs, extrinsics = [], []
    for viewpoint in viewpoints:
        img = utils.load_image(os.path.join(dirname, viewpoint['image_name']))
        imgs.append(img)
        extrinsics.append(Transform.from_dict(viewpoint['extrinsic']))

    return extrinsics, imgs


def _load_grasps(dirname):
    with open(os.path.join(dirname, 'grasps.json'), 'rb') as fp:
        grasps = json.load(fp)

    poses, scores = [], np.empty((len(grasps), ))
    for i, grasp in enumerate(grasps):
        poses.append(Transform.from_dict(grasp['pose']))
        scores[i] = grasp['score']
    return poses, scores
