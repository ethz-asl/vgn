import argparse
from pathlib import Path
from datetime import datetime

import open3d
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average
import torch


from vgn.loss import loss_fn
from vgn.metrics import Accuracy
from vgn.networks import get_network
from vgn.utils.training import *


def train(
    network_name,
    dataset_dir,
    augment,
    log_dir,
    descr,
    batch_size,
    lr,
    epochs,
    val_split,
):
    device = torch.device("cuda")
    kwargs = {"num_workers": 4, "pin_memory": True}

    descr = "{},net={},data={},augment={},batch_size={},lr={:.0e},descr={}".format(
        datetime.now().strftime("%b%d_%H-%M-%S"),
        network_name,
        dataset_dir.name,
        augment,
        batch_size,
        lr,
        descr,
    )
    log_dir = log_dir / descr
    assert not log_dir.exists(), "log with this setup already exists"

    train_loader, val_loader = create_train_val_loaders(
        dataset_dir, augment, batch_size, val_split, kwargs
    )

    net = get_network(network_name)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    metrics = {
        "loss": Average(lambda out: out[4][0]),
        "loss_qual": Average(lambda out: out[4][1]),
        "loss_rot": Average(lambda out: out[4][2]),
        "loss_width": Average(lambda out: out[4][3]),
        "acc": Accuracy(),
    }

    # create Ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(net, loss_fn, metrics, device)

    ProgressBar(persist=True, ascii=True).attach(trainer)

    train_writer, val_writer = create_summary_writers(net, device, log_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("loss_qual", metrics["loss_qual"], epoch)
        train_writer.add_scalar("loss_rot", metrics["loss_rot"], epoch)
        train_writer.add_scalar("loss_width", metrics["loss_width"], epoch)
        train_writer.add_scalar("acc", metrics["acc"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("loss_qual", metrics["loss_qual"], epoch)
        val_writer.add_scalar("loss_rot", metrics["loss_rot"], epoch)
        val_writer.add_scalar("loss_width", metrics["loss_width"], epoch)
        val_writer.add_scalar("acc", metrics["acc"], epoch)

    # save the model weights every 10 epochs
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

    # run the training loop
    trainer.run(train_loader, max_epochs=epochs)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--net", default="conv", help="network name")
    parser.add_argument(
        "--dataset-dir", type=str, required=True, help="root directory of the dataset"
    )
    parser.add_argument("--augment", action="store_true", help="augment train dataset")
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
    args = parser.parse_args()

    assert torch.cuda.is_available(), "ERROR: cuda is not available"

    train(
        network_name=args.net,
        dataset_dir=Path(args.dataset_dir),
        augment=args.augment,
        log_dir=Path(args.log_dir),
        descr=args.descr,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
