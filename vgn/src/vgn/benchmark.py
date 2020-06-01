from datetime import datetime
import uuid

import numpy as np
import pandas as pd

from vgn import to_voxel_coordinates
from vgn.utils import io


class Logger(object):
    def __init__(self, log_dir, description):
        time_stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        description = "{} {}".format(time_stamp, description).strip()
        self._root = log_dir / description
        self._root.mkdir()

    def add_round(self, round_id, object_count):
        csv_path = self._root / "rounds.csv"
        if not csv_path.exists():
            io.create_csv(csv_path, "round_id,object_count")
        io.append_csv(csv_path, round_id, object_count)

    def log_grasp(self, round_id, tsdf, planning_time, grasp, score, label):
        csv_path = self._root / "grasps.csv"

        if not csv_path.exists():
            header = "round_id,tsdf,planning_time,i,j,k,qx,qy,qz,qw,width,score,label"
            io.create_csv(csv_path, header)

        tsdf_path = self._root / (uuid.uuid4().hex + ".npz")
        np.savez_compressed(str(tsdf_path), tsdf=tsdf.get_volume())
        grasp = to_voxel_coordinates(grasp, tsdf.voxel_size)
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        i, j, k = np.round(grasp.pose.translation).astype(np.int)
        width = grasp.width
        label = int(label)

        io.append_csv(
            csv_path,
            round_id,
            tsdf_path.name,
            planning_time,
            i,
            j,
            k,
            qx,
            qy,
            qz,
            qw,
            width,
            score,
            label,
        )
