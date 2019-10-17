import json
import os

import numpy as np

import vgn.config as cfg
from vgn import utils
from vgn.grasp import Grasp, Label
from vgn.perception import camera, integration
from vgn.utils.transform import Transform


class SceneData(object):
    """Instance of the grasping dataset.

    Attributes:
        intrinsic: The camera intrinsic parameters.
        extrinsics: List of extrinsic parameters associated with each image. 
        depth_imgs: List of images of the scene.
        grasps: List of grasps that were attempted.
        labels: Outcomes of the attempted grasps.
    """

    def __init__(self, intrinsic, extrinsics, depth_imgs, grasps, labels):
        self.intrinsic = intrinsic
        self.extrinsics = extrinsics
        self.depth_imgs = depth_imgs
        self.grasps = grasps
        self.labels = labels

    @classmethod
    def load(cls, dirname):
        intrinsic = load_intrinsic(dirname)
        extrinsics, depth_imgs = load_images(dirname)
        grasps, labels = load_grasps(dirname)
        return cls(intrinsic, extrinsics, depth_imgs, grasps, labels)

    @property
    def n_grasp_attempts(self):
        return len(self.grasps)

    def save(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        save_intrinsic(dirname, self.intrinsic)
        save_images(dirname, self.extrinsics, self.depth_imgs)
        save_grasps(dirname, self.grasps, self.labels)


def load_intrinsic(dirname):
    fname = os.path.join(dirname, "intrinsic.json")
    return camera.PinholeCamera.from_dict(utils.load_dict(fname))


def load_images(dirname):
    images = utils.load_dict(os.path.join(dirname, "images.json"))
    depth_imgs, extrinsics = [], []
    for image in images:
        depth_imgs.append(utils.load_image(os.path.join(dirname, image["name"])))
        extrinsics.append(Transform.from_dict(image["extrinsic"]))
    return extrinsics, depth_imgs


def load_grasps(dirname):
    grasp_attempts = utils.load_dict(os.path.join(dirname, "grasps.json"))
    grasps, labels = [], []
    for attempt in grasp_attempts:
        grasps.append(Grasp.from_dict(attempt["grasp"]))
        labels.append(attempt["label"])
    return grasps, labels


def save_intrinsic(dirname, intrinsic):
    utils.save_dict(os.path.join(dirname, "intrinsic.json"), intrinsic.to_dict())


def save_images(dirname, extrinsics, depth_imgs):
    images = []
    for i in range(len(depth_imgs)):
        name = "{0:03d}.png".format(i)
        utils.save_image(os.path.join(dirname, name), depth_imgs[i])
        images.append({"name": name, "extrinsic": extrinsics[i].to_dict()})
    utils.save_dict(os.path.join(dirname, "images.json"), images)


def save_grasps(dirname, grasps, labels):
    grasp_attempts = []
    for grasp, label in zip(grasps, labels):
        grasp_attempts.append({"grasp": grasp.to_dict(), "label": label})
    utils.save_dict(os.path.join(dirname, "grasps.json"), grasp_attempts)
