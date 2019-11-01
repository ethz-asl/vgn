from __future__ import print_function

import argparse
from pathlib import Path
from datetime import datetime

import open3d
import torch
import torch.nn.functional as F
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Metric, Average
from torch.utils import tensorboard

from vgn import networks
from vgn.dataset import VGNDataset


def prepare_batch(batch, device):
    input_tsdf, target_quality, target_quat, mask = batch
    input_tsdf = input_tsdf.to(device)
    target_quality = target_quality.to(device)
    target_quat = target_quat.to(device)
    mask = mask.to(device)
    return input_tsdf, target_quality, target_quat, mask


def quality_loss_fn(pred, target, mask):
    loss = F.binary_cross_entropy(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum()


def quat_loss_fn(pred, target, mask):
    loss = F.l1_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum()


def loss_fn(pred_quality, target_quality, pred_quat, target_quat, mask):
    quality_loss = quality_loss_fn(pred_quality, target_quality, mask)
    quat_loss = quat_loss_fn(pred_quat, target_quat, target_quality)
    loss = quality_loss + quat_loss
    return loss


def create_trainer(net, optimizer, metrics, device):
    net.to(device)

    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # Forward
        input_tsdf, target_quality, target_quat, mask = prepare_batch(batch, device)
        pred_quality, pred_quat = net(input_tsdf)
        loss = loss_fn(pred_quality, target_quality, pred_quat, target_quat, mask)

        # Backward
        loss.backward()
        optimizer.step()

        return pred_quality, target_quality, pred_quat, target_quat, mask

    engine = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_evaluator(net, metrics, device):
    net.to(device)

    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            input_tsdf, target_quality, target_quat, mask = prepare_batch(batch, device)
            pred_quality, pred_quat = net(input_tsdf)

        return pred_quality, target_quality, pred_quat, target_quat, mask

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


class Accuracy(Metric):
    def reset(self):
        self.num_correct = 0
        self.num_examples = 0

    def update(self, out):
        pred_quality, target_quality, mask = out[0], out[1], out[4]
        correct = torch.eq(torch.round(pred_quality), target_quality) * mask

        self.num_correct += torch.sum(correct).item()
        self.num_examples += torch.sum(mask).item()

    def compute(self):
        return self.num_correct / self.num_examples


def create_summary_writers(net, trace, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    # Write the graph to Tensorboard
    train_writer.add_graph(net, trace)

    return train_writer, val_writer


def main(args):
    assert torch.cuda.is_available(), "ERROR: cuda is not available"

    device = torch.device("cuda")
    kwargs = {"pin_memory": True}
    root = Path(args.root)
    log_dir = Path(args.log_dir)

    # Create log directory for the training run
    descr = "{},net={},data={},batch_size={},lr={:.0e}".format(
        datetime.now().strftime("%b%d_%H-%M-%S"),
        args.net,
        root.name,
        args.batch_size,
        args.lr,
    )
    if args.descr != "":
        descr += ",descr={}".format(args.descr)
    log_dir = log_dir / descr
    assert not log_dir.exists(), "log with this setup already exists"

    # Load dataset
    dataset = VGNDataset(root, rebuild_cache=args.rebuild_cache)

    validation_size = int(args.validation_split * len(dataset))
    train_size = len(dataset) - validation_size

    print("Size of training dataset: {}".format(train_size))
    print("Size of validation dataset: {}".format(validation_size))

    # Create train and validation data loaders
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )

    # Build the network
    net = networks.get_network(args.net)

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Define metrics
    metrics = {
        "loss": Average(lambda out: loss_fn(out[0], out[1], out[2], out[3], out[4])),
        "loss_quality": Average(lambda out: quality_loss_fn(out[0], out[1], out[4])),
        "loss_quat": Average(lambda out: quat_loss_fn(out[2], out[3], out[1])),
        "acc": Accuracy(),
    }

    # Create Ignite engines for training and validation
    trainer = create_trainer(net, optimizer, metrics, device)
    evaluator = create_evaluator(net, metrics, device)

    # Add progress bar
    ProgressBar(persist=True, ascii=True).attach(trainer)

    # Logging
    trace, _, _, _ = prepare_batch(next(iter(train_loader)), device)
    train_writer, val_writer = create_summary_writers(net, trace, device, log_dir)

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
        Events.EPOCH_COMPLETED, checkpoint_handler, to_save={args.net: net}
    )

    # Run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--net", choices=["conv"], default="conv", help="network name")
    parser.add_argument(
        "--root", type=str, required=True, help="root directory of the dataset"
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
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="ratio of data used for validation",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    main(args)
