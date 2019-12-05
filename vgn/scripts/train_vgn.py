import argparse
from pathlib import Path
from datetime import datetime

import open3d
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average
import torch

from vgn.dataset import VGNDataset
from vgn.loss import *
from vgn.metrics import Accuracy
from vgn.networks import get_network
from vgn.utils.train import *


def train(
    network_name,
    data_dir,
    rebuild_cache,
    log_dir,
    descr,
    batch_size,
    lr,
    epochs,
    val_split,
):
    device = torch.device("cuda")
    kwargs = {"pin_memory": True}

    # Create log directory for the training run
    descr = "{},net={},data={},batch_size={},lr={:.0e},descr={}".format(
        datetime.now().strftime("%b%d_%H-%M-%S"),
        network_name,
        data_dir.name,
        batch_size,
        lr,
        descr,
    )
    log_dir = log_dir / descr
    assert not log_dir.exists(), "log with this setup already exists"

    # Load dataset
    dataset = VGNDataset(data_dir, rebuild_cache=rebuild_cache)

    # Split into train and validation sets
    validation_size = int(val_split * len(dataset))
    train_size = len(dataset) - validation_size

    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size]
    )

    print("Size of training dataset: {}".format(train_size))
    print("Size of validation dataset: {}".format(validation_size))

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    # Build the network
    net = get_network(network_name)

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Define metrics
    metrics = {
        "loss": Average(lambda out: loss_fn(out)),
        "loss_quality": Average(lambda out: quality_loss_fn(out)),
        "loss_quat": Average(lambda out: quat_loss_fn(out)),
        "acc": Accuracy(),
    }

    # Create Ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(net, metrics, device)

    # Add progress bar
    ProgressBar(persist=True, ascii=True).attach(trainer)

    # Logging
    train_writer, val_writer = create_summary_writers(net, device, log_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("loss_quality", metrics["loss_quality"], epoch)
        train_writer.add_scalar("loss_quat", metrics["loss_quat"], epoch)
        train_writer.add_scalar("acc", metrics["acc"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(validation_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("loss_quality", metrics["loss_quality"], epoch)
        val_writer.add_scalar("loss_quat", metrics["loss_quat"], epoch)
        val_writer.add_scalar("acc", metrics["acc"], epoch)

    # Save the model weights every 10 epochs
    checkpoint_handler = ModelCheckpoint(
        log_dir,
        "vgn",
        save_interval=10,
        n_saved=100,
        require_empty=True,
        save_as_state_dict=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, to_save={network_name: net}
    )

    # Run the training loop
    trainer.run(train_loader, max_epochs=epochs)


def main(args):
    assert torch.cuda.is_available(), "ERROR: cuda is not available"

    train(
        network_name=args.net,
        data_dir=Path(args.data_dir),
        rebuild_cache=args.rebuild_cache,
        log_dir=Path(args.log_dir),
        descr=args.descr,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--net", choices=["conv"], default="conv", help="network name")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="root directory of the dataset"
    )
    parser.add_argument(
        "--log-dir", type=str, default="data/runs", help="path to log directory"
    )
    parser.add_argument(
        "--descr", type=str, default="", help="description appended to the run dirname"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="input batch size for training"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="ratio of data used for validation",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    main(args)
