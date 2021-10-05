import numpy as np
import pandas as pd
import uuid

from robot_helpers.spatial import Transform


class GraspDatabase:
    def __init__(self, root):
        self.root = root
        self.df = pd.read_csv(root / "grasps.csv")

    @property
    def scenes(self):
        return self.df.scene_id.unique()

    def read(self, id):
        imgs, views = self.read_scene(id)
        grasps, qualities = self.read_grasps(id)
        return imgs, views, grasps, qualities

    def read_scene(self, id):
        data = np.load(self.root / (id + ".npz"))
        imgs, views = data["depth_imgs"], data["views"]
        views = [Transform.from_list(view) for view in views]
        return imgs, views

    def read_grasps(self, id):
        pass  # TODO


def write(root, views, imgs, grasps, qualities):
    scene_id = uuid.uuid4().hex
    write_scene(root, scene_id, views, imgs)
    write_grasps(root / "grasps.csv", scene_id, grasps, qualities)


def write_scene(root, id, view_list, img_list):
    count, shape = len(img_list), img_list[0].shape
    views = np.empty((count, 7), np.float32)
    imgs = np.empty((count,) + shape, np.float32)
    for i in range(count):
        views[i] = view_list[i].to_list()
        imgs[i] = img_list[i]
    np.savez_compressed(root / (id + ".npz"), views=views, depth_imgs=imgs)


def write_grasps(path, id, grasps, qualities):
    rows = []
    for g, q in zip(grasps, qualities):
        ori, pos = g.pose.rotation.as_quat(), g.pose.translation
        config = {
            "scene_id": id,
            "qx": ori[0],
            "qy": ori[1],
            "qz": ori[2],
            "qw": ori[3],
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "width": g.width,
            "physics": q,
        }
        rows.append(config)
    df = pd.DataFrame.from_records(rows)
    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        float_format="%.4f",
    )
