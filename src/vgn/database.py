import numpy as np
import pandas as pd
import uuid


class GraspDatabase:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(exist_ok=True)
        self.df_path = root / "grasps.csv"
        self.header = not self.df_path.exists()

    def write(self, views, imgs, grasps, qualities):
        scene_id = uuid.uuid4().hex
        self.write_scene(scene_id, views, imgs)
        self.write_grasps(scene_id, grasps, qualities)

    def read(self, id):
        views, imgs = self.read_scene(id)
        grasps, qualities = self.read_grasps(id)
        return views, imgs, grasps, qualities

    def write_scene(self, id, view_list, img_list):
        count, shape = len(img_list), img_list[0].shape
        views = np.empty((count, 7), np.float32)
        imgs = np.empty((count,) + shape, np.float32)
        for i in range(count):
            views[i] = view_list[i].to_list()
            imgs[i] = img_list[i]
        np.savez_compressed(self.root / (id + ".npz"), views=views, depth_imgs=imgs)

    def write_grasps(self, id, grasps, qualities):
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
        df.to_csv(self.df_path, mode="a", header=self.header, index=False)
