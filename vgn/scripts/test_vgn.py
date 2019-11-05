import argparse
from pathlib import Path

import open3d
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Average
import torch

from vgn.dataset import VGNDataset
from vgn.loss import *
from vgn.metrics import Accuracy
from vgn.networks import get_network
from vgn.utils.train import *


def main(args):
    assert torch.cuda.is_available(), "ERROR: cuda is not available"
    device = torch.device("cuda")
    kwargs = {"pin_memory": True}

    # Load dataset
    test_dataset = VGNDataset(Path(args.root), rebuild_cache=args.rebuild_cache)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )

    # Build network and load weights
    model_path = Path(args.model)
    net = get_network(model_path.name.split("_")[1]).to(device)
    net.load_state_dict(torch.load(model_path))

    # Define test metrics
    metrics = {
        "loss": Average(lambda out: loss_fn(out)),
        "loss_quality": Average(lambda out: quality_loss_fn(out)),
        "loss_quat": Average(lambda out: quat_loss_fn(out)),
        "acc": Accuracy(),
    }

    # Create and run the test engine
    tester = create_evaluator(net, metrics, device)
    ProgressBar(persist=True, ascii=True).attach(tester)
    tester.run(test_loader)

    # Print the results
    metrics = tester.state.metrics
    print("Loss", metrics["loss"])
    print("Quality loss", metrics["loss_quality"])
    print("Quat loss", metrics["loss_quat"])
    print("Accuracy", metrics["acc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", type=str, required=True, help="root directory of the dataset"
    )

    parser.add_argument(
        "--root", type=str, required=True, help="root directory of the test dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="input batch size for testing"
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    main(args)
