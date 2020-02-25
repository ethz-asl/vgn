import argparse
from pathlib import Path

from vgn.dataset import VgnDataset


def main(data_dir):
    dataset = VgnDataset(data_dir)
    dataset.stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a trained vgn model")
    parser.add_argument(
        "--data-dir", required=True, type=str, help="root directory of the dataset"
    )

    args = parser.parse_args()

    main(Path(args.data_dir))
