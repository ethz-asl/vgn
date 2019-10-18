import json

import numpy as np

import vgn.config as cfg
from vgn import utils
from vgn.grasp import Grasp, Label
from vgn.perception import integration
from vgn.perception.camera import PinholeCameraIntrinsic
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
    def load(cls, scene_dir):
        intrinsic = load_intrinsic(scene_dir)
        extrinsics, depth_imgs = load_images(scene_dir)
        grasps, labels = load_grasps(scene_dir)
        return cls(intrinsic, extrinsics, depth_imgs, grasps, labels)

    @property
    def n_grasp_attempts(self):
        return len(self.grasps)

    def save(self, scene_dir):
        scene_dir.mkdir()
        save_intrinsic(scene_dir, self.intrinsic)
        save_images(scene_dir, self.extrinsics, self.depth_imgs)
        save_grasps(scene_dir, self.grasps, self.labels)


def load_intrinsic(scene_dir):
    return PinholeCameraIntrinsic.from_dict(
        utils.load_dict(scene_dir / "intrinsic.json")
    )


def load_images(scene_dir):
    images = utils.load_dict(scene_dir / "images.json")
    depth_imgs, extrinsics = [], []
    for image in images:
        depth_imgs.append(utils.load_image(scene_dir / image["name"]))
        extrinsics.append(Transform.from_dict(image["extrinsic"]))
    return extrinsics, depth_imgs


def load_grasps(scene_dir):
    grasp_attempts = utils.load_dict(scene_dir / "grasps.json")
    grasps, labels = [], []
    for attempt in grasp_attempts:
        grasps.append(Grasp.from_dict(attempt["grasp"]))
        labels.append(attempt["label"])
    return grasps, labels


def save_intrinsic(scene_dir, intrinsic):
    utils.save_dict(scene_dir / "intrinsic.json", intrinsic.to_dict())


def save_images(scene_dir, extrinsics, depth_imgs):
    images = []
    for i in range(len(depth_imgs)):
        name = "{0:03d}.png".format(i)
        utils.save_image(scene_dir / name, depth_imgs[i])
        images.append({"name": name, "extrinsic": extrinsics[i].to_dict()})
    utils.save_dict(scene_dir / "images.json", images)


def save_grasps(scene_dir, grasps, labels):
    grasp_attempts = []
    for grasp, label in zip(grasps, labels):
        grasp_attempts.append({"grasp": grasp.to_dict(), "label": label})
    utils.save_dict(scene_dir / "grasps.json", grasp_attempts)
