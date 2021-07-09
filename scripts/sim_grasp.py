import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vgn.detection import VGN, compute_grasps
from vgn.envs import ClutterRemovalEnv


def main(args):
    env = ClutterRemovalEnv(args.scene, args.object_count, args.seed, args.gui)
    vgn = VGN(args.model)
    score_fn = lambda g: np.random.uniform(0.0, 1.0)

    object_count = 0
    grasp_count = 0
    cleared_count = 0

    for _ in tqdm(range(args.episode_count)):
        tsdf_grid, voxel_size = env.reset()
        object_count += env.sim.num_objects
        done = False
        while not done:
            out = vgn.predict(tsdf_grid)
            grasps = compute_grasps(voxel_size, out, score_fn, threshold=0.90)
            if len(grasps) == 0:
                break
            (tsdf_grid, voxel_size), score, done, _ = env.step(grasps[0])
            grasp_count += 1
            cleared_count += score

    print(
        "Grasp count: {}, success rate: {:.2f}, percent cleared: {:.2f}".format(
            grasp_count,
            (cleared_count / grasp_count) * 100,
            (cleared_count / object_count) * 100,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default="assets/models/vgn_conv.pth")
    parser.add_argument(
        "--scene",
        type=str,
        choices=["blocks", "pile-train", "pile-test", "packed-train", "packed-test"],
        default="blocks",
    )
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--episode-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    main(args)
