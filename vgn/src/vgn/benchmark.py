from datetime import datetime

import numpy as np
import pandas as pd


class Logger(object):
    def __init__(self, log_dir, description):
        time_stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        description = "{} {}".format(time_stamp, description).strip()
        self._root = log_dir / description
        self._root.mkdir()

    def add_round(self, round_id, object_count, model_path):
        csv_path = self._root / "rounds.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        df = df.append(
            {
                "round_id": round_id,
                "object_count": object_count,
                "model_path": model_path,
            },
            ignore_index=True,
        )
        df.to_csv(csv_path, index=False)

    def log_grasp(self, round_id, planning_time, score, label):
        csv_path = self._root / "grasps.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        df = df.append(
            {
                "round_id": round_id,
                "planning_time": planning_time,
                "score": score,
                "label": label,
            },
            ignore_index=True,
        )
        df.to_csv(csv_path, index=False)
