from datetime import datetime
import uuid

import numpy as np
import pandas as pd

from vgn import to_voxel_coordinates


class Logger(object):
    def __init__(self, log_dir, description):
        time_stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        description = "{} {}".format(time_stamp, description).strip()
        self._root = log_dir / description
        self._root.mkdir()

    def add_round(self, round_id, object_count):
        csv_path = self._root / "rounds.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        df = df.append(
            {"round_id": round_id, "object_count": object_count}, ignore_index=True,
        )
        df.to_csv(csv_path, index=False)

    def log_grasp(self, round_id, tsdf, plan_time, grasp, score, label):
        csv_path = self._root / "grasps.csv"
        tsdf_path = self._root / (uuid.uuid4().hex + ".npz")
        np.savez_compressed(str(tsdf_path), tsdf=tsdf.get_volume())
        grasp = to_voxel_coordinates(grasp, tsdf.voxel_size)
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        i, j, k = np.round(grasp.pose.translation).astype(np.int)
        width = grasp.width
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        df = df.append(
            {
                "round_id": round_id,
                "tsdf": tsdf_path.name,
                "plan_time": plan_time,
                "i": i,
                "j": j,
                "k": k,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "width": width,
                "score": score,
                "label": label,
            },
            ignore_index=True,
        )
        df.to_csv(csv_path, index=False)
