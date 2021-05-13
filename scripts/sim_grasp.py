import argparse
from pathlib import Path

from tqdm import tqdm

from vgn.detection import VGN, compute_grasps
from vgn.envs import ClutterRemovalEnv


def main(args):
    env = ClutterRemovalEnv(args.scene, args.object_count, args.gui)
    vgn = VGN(args.model)

    tsdf_grid, voxel_size = env.reset()
    for _ in tqdm(range(args.grasp_count)):
        out = vgn.predict(tsdf_grid)
        grasps = compute_grasps(out, voxel_size)
        if len(grasps) == 0:
            tsdf_grid, voxel_size = env.reset()
            continue
        (tsdf_grid, voxel_size), score, done, info = env.step(grasps[0])
        print(score, done)

        if done:
            tsdf_grid, voxel_size = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--scene",
        type=str,
        choices=["blocks", "pile-train", "pile-test", "packed-train", "packed-test"],
        default="blocks",
    )
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--grasp-count", type=int, default=100)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    main(args)
