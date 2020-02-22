import argparse
from pathlib import Path

import open3d
from mayavi import mlab
import torch

from vgn.dataset import VgnDataset, RandomAffine
from vgn.grasp_detector import GraspDetector


def main(args):
    sample_path = Path(args.sample)

    transforms = [RandomAffine()]
    dataset = VgnDataset(sample_path.parent, transforms=transforms)
    tsdf, (qual, rot, width), mask = dataset[dataset.samples.index(sample_path.name)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_path = Path(args.model)
    detector = GraspDetector(
        device, network_path, show_tsdf=True, show_qual=True, show_detections=True,
    )

    grasps, qualities = detector.detect_grasps(tsdf)
    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="detect grasps in a VGN sample")
    parser.add_argument(
        "--model", type=str, required=True, help="saved model ending with .pth"
    )
    parser.add_argument("--sample", required=True, type=str, help="path to the sample")
    args = parser.parse_args()
    main(args)
