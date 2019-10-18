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
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch.utils import tensorboard

from vgn import networks
from vgn.dataset import VGNDataset


def prepare_batch(batch, device):
    tsdf, idx, quat, quality = batch
    tsdf = tsdf.to(device)
    idx = idx.to(device)
    quality = quality.squeeze().to(device)
    quat = quat.to(device)
    return tsdf, idx, quality, quat


def select_pred(qualities, quats, indices):
    quality_pred = torch.cat(
        [
            s[0, i[:, 0], i[:, 1], i[:, 2]].unsqueeze(0)
            for s, i in zip(qualities, indices)
        ]
    )
    quat_pred = torch.cat(
        [q[:, i[:, 0], i[:, 1], i[:, 2]].unsqueeze(0) for q, i in zip(quats, indices)]
    )
    return quality_pred, quat_pred


def create_trainer(net, optimizer, quality_loss_fn, quat_loss_fn, device):
    net.to(device)

    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # Forward pass
        tsdf, idx, quality, quat = prepare_batch(batch, device)
        quality_out, quat_out = net(tsdf)
        quality_pred, quat_pred = select_pred(quality_out, quat_out, idx)
        loss_quality = quality_loss_fn(quality_pred, quality)
        loss_quat = quat_loss_fn(quat_pred, quat)
        loss = loss_quality + loss_quat

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss, loss_quality, loss_quat, quality_pred, quality

    return Engine(_update)


def create_evaluator(net, device):
    net.to(device)

    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            tsdf, idx, quality, quat = prepare_batch(batch, device)
            quality_out, quat_out = net(tsdf)
            quality_pred, quat_pred = select_pred(quality_out, quat_out, idx)
        return quality_pred, quality, quat_pred, quat

    engine = Engine(_inference)

    return engine


def create_summary_writers(net, data_loader, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    # Write the graph to Tensorboard
    trace, _, _, _ = prepare_batch(next(iter(data_loader)), device)
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

    # Define loss functions
    quality_loss_fn = F.binary_cross_entropy
    quat_loss_fn = torch.nn.L1Loss()

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Create Ignite engines for training and validation
    trainer = create_trainer(net, optimizer, quality_loss_fn, quat_loss_fn, device)
    evaluator = create_evaluator(net, device)

    # Train metrics
    quality_loss_metric = RunningAverage(output_transform=lambda x: x[1])
    quality_loss_metric.attach(trainer, "loss_quality")
    quat_loss_metric = RunningAverage(output_transform=lambda x: x[2])
    quat_loss_metric.attach(trainer, "loss_quat")
    loss_metric = RunningAverage(output_transform=lambda x: x[0])
    loss_metric.attach(trainer, "loss")
    acc_metric = Accuracy(lambda x: (torch.round(x[3]), x[4]))
    acc_metric.attach(trainer, "acc")

    # Validation metrics
    quality_loss_metric = Loss(quality_loss_fn, lambda x: (x[0], x[1]))
    quality_loss_metric.attach(evaluator, "loss_quality")
    quat_loss_metric = Loss(quat_loss_fn, lambda x: (x[2], x[3]))
    quat_loss_metric.attach(evaluator, "loss_quat")
    loss_metric = quality_loss_metric + quat_loss_metric
    loss_metric.attach(evaluator, "loss")
    acc_metric = Accuracy(lambda x: (torch.round(x[0]), x[1]))
    acc_metric.attach(evaluator, "acc")

    # Setup loggers and checkpoints
    ProgressBar(persist=True, ascii=True).attach(trainer, ["loss"])

    train_writer, val_writer = create_summary_writers(
        net, train_loader, device, log_dir
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(_):
        evaluator.run(validation_loader)

        epoch = trainer.state.epoch

        metrics = trainer.state.metrics
        train_writer.add_scalar("loss_quality", metrics["loss_quality"], epoch)
        train_writer.add_scalar("loss_quat", metrics["loss_quat"], epoch)
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("accuracy", metrics["acc"], epoch)

        metrics = evaluator.state.metrics
        val_writer.add_scalar("loss_quality", metrics["loss_quality"], epoch)
        val_writer.add_scalar("loss_quat", metrics["loss_quat"], epoch)
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("accuracy", metrics["acc"], epoch)

    checkpoint_handler = ModelCheckpoint(
        log_dir,
        "vgn",
        save_interval=10,
        n_saved=100,
        require_empty=True,
        save_as_state_dict=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, to_save={"net": net}
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
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="ratio of data used for validation",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    main(args)
