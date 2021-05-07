import argparse
from pathlib import Path

import numpy as np
import tqdm

from robot_utils.perception import create_grid_from_map_cloud
from vgn.grasp import *
from vgn.detection import VGN, compute_grasps
from vgn.simulation import ClutterRemovalSim

MAX_CONSECUTIVE_FAILURES = 2


def main(args):
    # Construct VGN
    vgn = VGN(args.model)

    # Construct the test env
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.gui, seed=args.seed)

    # Run experiments
    for _ in tqdm.tqdm(range(args.episode_count)):
        sim.reset(args.object_count)

        consecutive_failures = 1
        last_label = None

        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            # Scan the scene
            tsdf, pc, _ = sim.acquire_tsdf(n=6, N=None)

            if pc.is_empty():
                break  # Empty point cloud, abort this round TODO this should not happen

            # Plan grasps
            map_cloud = tsdf.get_map_cloud()
            points = np.asarray(map_cloud.points)
            distances = np.asarray(map_cloud.colors)[:, 0]
            tsdf_grid = create_grid_from_map_cloud(points, distances, tsdf.voxel_size)
            out = vgn.predict(tsdf_grid)
            grasps = compute_grasps(out, voxel_size=tsdf.voxel_size)

            if len(grasps) == 0:
                break  # No detections found, abort this round

            # Execute grasp
            label, _ = sim.execute_grasp(grasps[0], allow_contact=True)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", type=str, default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--episode-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    main(args)
