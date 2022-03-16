import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from robot_helpers.io import load_yaml
from vgn.detection import VGN, select_local_maxima
from vgn.envs import ClutterRemovalEnv
import vgn.visualizer as vis


def main():
    parser = create_parser()
    args = parser.parse_args()
    cfg = load_yaml(args.cfg)
    rng = np.random.RandomState(args.seed)
    env = ClutterRemovalEnv(cfg, rng)
    vgn = VGN(args.model)

    grasp_count = 0
    cleared_count = 0
    object_count = 0

    def compute_grasp(voxel_size, tsdf_grid):
        out = vgn.predict(tsdf_grid)
        grasps, qualities = select_local_maxima(voxel_size, out, threshold=0.9)
        # vis.grasps(grasps, qualities)
        return grasps[np.argmax(qualities)] if len(grasps) > 0 else None

    for _ in tqdm(range(args.episode_count)):
        info = env.reset()
        object_count += info["object_count"]
        done = False
        while not done:
            voxel_size, tsdf_grid = env.get_observation()
            grasp = compute_grasp(voxel_size, tsdf_grid)
            # vis.show()
            if grasp:
                success, done = env.step(grasp)
                grasp_count += 1
                cleared_count += success
            else:
                break

    print(
        "Grasp count: {}, success rate: {:.2f}, percent cleared: {:.2f}".format(
            grasp_count,
            (cleared_count / grasp_count) * 100,
            (cleared_count / object_count) * 100,
        )
    )


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default="assets/models/vgn_conv.pth")
    parser.add_argument("--cfg", type=Path, default="cfg/sim/blocks.yaml")
    parser.add_argument("--episode-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__":
    main()
