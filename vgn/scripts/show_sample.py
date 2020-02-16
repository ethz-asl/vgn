import argparse
from pathlib import Path

import open3d

from vgn.dataset import VgnDataset
from vgn.utils.vis import show_sample


def main(args):
    sample_path = Path(args.sample_path)

    dataset = VgnDataset(sample_path.parent)
    tsdf, (qual, rot, width), mask = dataset[dataset.samples.index(sample_path.name)]

    show_sample(tsdf, qual, rot, width, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize data from a sample")
    parser.add_argument("sample_path", type=str, help="path to the sample to be showed")
    args = parser.parse_args()
    main(args)
