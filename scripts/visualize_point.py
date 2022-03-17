import argparse
import numpy as np
from pathlib import Path

from robot_helpers.spatial import Rotation, Transform
from vgn.dataset import VGNDataset
from vgn.detection import from_voxel_coordinates
from vgn.grasp import ParallelJawGrasp
from vgn.utils import grid_to_map_cloud
import vgn.visualizer as vis


def main():
    parser = create_parser()
    args = parser.parse_args()
    dataset = VGNDataset(args.root, augment=False)

    voxel_size = 0.3 / 40

    while True:
        i = np.random.choice(len(dataset))
        tsdf, (label, rot, width), index = dataset[i]
        points, distances = grid_to_map_cloud(voxel_size, tsdf.squeeze())
        grasp = ParallelJawGrasp(Transform(Rotation.from_quat(rot[0]), index), width)
        vis.map_cloud(voxel_size, points, distances.squeeze())
        vis.grasp(from_voxel_coordinates(voxel_size, grasp), label)
        vis.show()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    return parser


if __name__ == "__main__":
    main()
