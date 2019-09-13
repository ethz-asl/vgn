import json
import os

import numpy as np

import vgn.config as cfg
from vgn import utils
from vgn.perception import camera, integration
from vgn.utils.transform import Transform


def load_scene(dirname):
    intrinsic = _load_intrinsic(dirname)
    extrinsics, images = _load_images(dirname)
    poses, scores = _load_grasps(dirname)
    scene = {
        'intrinsic': intrinsic,
        'extrinsics': extrinsics,
        'images': images,
        'poses': poses,
        'scores': scores
    }
    return scene


def store_scene(dirname, scene):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    _store_intrinsic(dirname, scene['intrinsic'])
    _store_images(dirname, scene['extrinsics'], scene['depth_imgs'])
    _store_grasps(dirname, scene['poses'], scene['scores'])


def _load_intrinsic(dirname):
    fname = os.path.join(dirname, 'intrinsic.json')
    return camera.PinholeCameraIntrinsic.from_json(fname)


def _store_intrinsic(dirname, intrinsic):
    intrinsic.to_json(os.path.join(dirname, 'intrinsic.json'))


def _load_images(dirname):
    with open(os.path.join(dirname, 'viewpoints.json'), 'rb') as fp:
        viewpoints = json.load(fp)
    imgs, extrinsics = [], []
    for viewpoint in viewpoints:
        img = utils.load_image(os.path.join(dirname, viewpoint['img_name']))
        imgs.append(img)
        extrinsics.append(Transform.from_dict(viewpoint['extrinsic']))

    return extrinsics, imgs


def _store_images(dirname, extrinsics, depth_imgs):
    viewpoints = []
    for i in range(len(depth_imgs)):
        img_name = '{0:03d}.png'.format(i)
        utils.save_image(os.path.join(dirname, img_name), depth_imgs[i])
        viewpoints.append({
            'img_name': img_name,
            'extrinsic': extrinsics[i].to_dict(),
        })
    with open(os.path.join(dirname, 'viewpoints.json'), 'wb') as fp:
        json.dump(viewpoints, fp, indent=4)


def _load_grasps(dirname):
    with open(os.path.join(dirname, 'grasps.json'), 'rb') as fp:
        grasps = json.load(fp)
    poses, scores = [], np.empty((len(grasps), ))
    for i, grasp in enumerate(grasps):
        poses.append(Transform.from_dict(grasp['pose']))
        scores[i] = grasp['score']
    return poses, scores


def _store_grasps(dirname, poses, scores):
    grasps = []
    for i in range(len(poses)):
        grasps.append({'pose': poses[i].to_dict(), 'score': scores[i]})
    with open(os.path.join(dirname, 'grasps.json'), 'wb') as fp:
        json.dump(grasps, fp, indent=4)
