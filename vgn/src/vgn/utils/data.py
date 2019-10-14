import json
import os

import numpy as np

import vgn.config as cfg
from vgn import utils
from vgn.perception import camera, integration
from vgn.utils.transform import Transform


def store_scene(dirname, scene):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    store_intrinsic(dirname, scene["intrinsic"])
    store_images(dirname, scene["extrinsics"], scene["depth_imgs"])
    store_grasps(dirname, scene["poses"], scene["outcomes"])


def load_scene(dirname):
    intrinsic = load_intrinsic(dirname)
    extrinsics, depth_imgs = load_images(dirname)
    poses, outcomes = load_grasps(dirname)
    scene = {
        "intrinsic": intrinsic,
        "extrinsics": extrinsics,
        "depth_imgs": depth_imgs,
        "poses": poses,
        "outcomes": outcomes,
    }
    return scene


def store_intrinsic(dirname, intrinsic):
    intrinsic.to_json(os.path.join(dirname, "intrinsic.json"))


def load_intrinsic(dirname):
    fname = os.path.join(dirname, "intrinsic.json")
    return camera.PinholeCameraIntrinsic.from_json(fname)


def store_images(dirname, extrinsics, depth_imgs):
    viewpoints = []
    for i in range(len(depth_imgs)):
        img_name = "{0:03d}.png".format(i)
        utils.save_image(os.path.join(dirname, img_name), depth_imgs[i])
        viewpoints.append({"img_name": img_name, "extrinsic": extrinsics[i].to_dict()})
    with open(os.path.join(dirname, "viewpoints.json"), "w") as fp:
        json.dump(viewpoints, fp, indent=4)


def load_images(dirname):
    with open(os.path.join(dirname, "viewpoints.json"), "r") as fp:
        viewpoints = json.load(fp)
    imgs, extrinsics = [], []
    for viewpoint in viewpoints:
        img = utils.load_image(os.path.join(dirname, viewpoint["img_name"]))
        imgs.append(img)
        extrinsics.append(Transform.from_dict(viewpoint["extrinsic"]))

    return extrinsics, imgs


def store_grasps(dirname, poses, outcomes):
    grasps = []
    for i in range(len(poses)):
        grasps.append({"pose": poses[i].to_dict(), "outcome": outcomes[i]})
    with open(os.path.join(dirname, "grasps.json"), "w") as fp:
        json.dump(grasps, fp, indent=4)


def load_grasps(dirname):
    with open(os.path.join(dirname, "grasps.json"), "r") as fp:
        grasps = json.load(fp)
    poses, outcomes = [], np.empty((len(grasps),))
    for i, grasp in enumerate(grasps):
        poses.append(Transform.from_dict(grasp["pose"]))
        outcomes[i] = grasp["outcome"]
    return poses, outcomes
