import argparse
import pathlib

from mayavi import mlab
import torch

from vgn.dataset import VgnDataset, RandomAffine
from vgn.detector import GraspDetector


def main(args):
    sample_path = pathlib.Path(args.sample)
    transforms = [RandomAffine()]
    dataset = VgnDataset(sample_path.parent, transforms=transforms)
    tsdf, (qual, rot, width), mask = dataset[dataset.samples.index(sample_path.name)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = GraspDetector(device, pathlib.Path(args.model))
    detector.detect_grasps(tsdf, show_detections=True)

    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sample", type=str, required=True)
    args = parser.parse_args()
    main(args)
