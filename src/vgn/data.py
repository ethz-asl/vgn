import numpy as np
import pandas as pd
import uuid

from robot_helpers.spatial import Transform
from vgn.grasp import ParallelJawGrasp


def write(views, imgs, grasps, scores, root):
    scene_id = uuid.uuid4().hex
    write_sensor_data(views, imgs, root, scene_id)
    write_grasps(grasps, scores, root / "grasps.csv", scene_id)


def read(root, df, id):
    imgs, views = read_sensor_data(root, id)
    grasps, scores = read_grasps(df, id)
    return imgs, views, grasps, scores


def write_sensor_data(views, images, root, id):
    count, shape = len(images), images[0].shape
    views_array = np.empty((count, 7), np.float32)
    imgs_array = np.empty((count,) + shape, np.float32)
    for i in range(count):
        views_array[i] = views[i].to_list()
        imgs_array[i] = images[i]
    np.savez_compressed(root / (id + ".npz"), views=views_array, depth_imgs=imgs_array)


def read_sensor_data(root, id):
    data = np.load(root / (id + ".npz"))
    imgs, views = data["depth_imgs"], data["views"]
    views = [Transform.from_list(view) for view in views]
    return imgs, views


def write_grasps(grasps, scores, path, id):
    rows = []
    for grasp, score in zip(grasps, scores):
        ori, pos = grasp.pose.rotation.as_quat(), grasp.pose.translation
        config = {
            "scene_id": id,
            "qx": ori[0],
            "qy": ori[1],
            "qz": ori[2],
            "qw": ori[3],
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "width": grasp.width,
            "score": score,
        }
        rows.append(config)
    df = pd.DataFrame.from_records(rows)
    df.to_csv(
        path, mode="a", header=not path.exists(), index=False, float_format="%.4f"
    )


def read_grasps(df, id):
    grasps, scores = [], []
    for _, r in df[df.scene_id == id].iterrows():
        pose = Transform.from_list([r.qx, r.qy, r.qz, r.qw, r.x, r.y, r.z])
        grasps.append(ParallelJawGrasp(pose, r.width))
        scores.append(r.score)
    return np.asarray(grasps), np.asarray(scores)


def read_grid(root, scene_id):
    path = root / (scene_id + ".npz")
    return np.load(path)["grid"]


def write_grid(grid, root, scene_id):
    path = root / (scene_id + ".npz")
    np.savez_compressed(path, grid=grid)
