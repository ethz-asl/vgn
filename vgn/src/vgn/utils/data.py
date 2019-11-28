import numpy as np

from vgn import utils
from vgn.grasp import Grasp, Label
from vgn.perception import integration
from vgn.perception.camera import PinholeCameraIntrinsic
from vgn.utils.transform import Transform


class SceneData(object):
    """Instance of the grasping dataset.

    Attributes:
        depth_imgs: List of images of the scene.
        intrinsic: The camera intrinsic parameters.
        extrinsics: List of extrinsic parameters associated with each image. 
        grasps: List of grasps that were attempted.
        labels: Outcomes of the attempted grasps.
    """

    def __init__(self, depth_imgs, intrinsic, extrinsics, grasps, labels):
        self.depth_imgs = depth_imgs
        self.intrinsic = intrinsic
        self.extrinsics = extrinsics
        self.grasps = grasps
        self.labels = labels

    @classmethod
    def load(cls, scene_dir):
        intrinsic = load_intrinsic(scene_dir)
        depth_imgs, extrinsics = load_images(scene_dir)
        grasps, labels = load_grasps(scene_dir)
        return cls(depth_imgs, intrinsic, extrinsics, grasps, labels)

    @property
    def n_grasp_attempts(self):
        return len(self.grasps)

    def save(self, scene_dir):
        scene_dir.mkdir()
        save_intrinsic(scene_dir, self.intrinsic)
        save_images(scene_dir, self.depth_imgs, self.extrinsics)
        save_grasps(scene_dir, self.grasps, self.labels)


def load_intrinsic(scene_dir):
    return PinholeCameraIntrinsic.from_dict(
        utils.io.load_dict(scene_dir / "intrinsic.json")
    )


def load_images(scene_dir):
    images = utils.io.load_dict(scene_dir / "images.json")
    depth_imgs, extrinsics = [], []
    for image in images:
        depth_imgs.append(utils.io.load_image(scene_dir / image["name"]))
        extrinsics.append(Transform.from_dict(image["extrinsic"]))
    return depth_imgs, extrinsics


def load_grasps(scene_dir):
    grasp_attempts = utils.io.load_dict(scene_dir / "grasps.json")
    grasps, labels = [], []
    for attempt in grasp_attempts:
        grasps.append(Grasp.from_dict(attempt["grasp"]))
        labels.append(attempt["label"])
    return grasps, labels


def save_intrinsic(scene_dir, intrinsic):
    utils.io.save_dict(intrinsic.to_dict(), scene_dir / "intrinsic.json")


def save_images(scene_dir, depth_imgs, extrinsics):
    images = []
    for i in range(len(depth_imgs)):
        name = "{0:03d}.png".format(i)
        utils.io.save_image(depth_imgs[i], scene_dir / name)
        images.append({"name": name, "extrinsic": extrinsics[i].to_dict()})
    utils.io.save_dict(images, scene_dir / "images.json")


def save_grasps(scene_dir, grasps, labels):
    grasp_attempts = []
    for grasp, label in zip(grasps, labels):
        grasp_attempts.append({"grasp": grasp.to_dict(), "label": label})
    utils.io.save_dict(grasp_attempts, scene_dir / "grasps.json")
