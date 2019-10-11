from __future__ import print_function

import argparse
import os
from datetime import datetime

import open3d
import torch
import torch.nn.functional as F
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch.utils import tensorboard

from vgn import models, utils
from vgn.dataset import VGNDataset


def create_trainer(model, optimizer, quality_loss_fn, quat_loss_fn, device):
    model.to(device)

    def _update(_, batch):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        tsdf, idx, quality, quat = utils.prepare_batch(batch, device)
        quality_out, quat_out = model(tsdf)
        quality_pred, quat_pred = utils.select_pred(quality_out, quat_out, idx)
        loss_quality = quality_loss_fn(quality_pred, quality)
        loss_quat = quat_loss_fn(quat_pred, quat)
        loss = loss_quality + loss_quat

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss, loss_quality, loss_quat, quality_pred, quality

    return Engine(_update)


def create_evaluator(model, device):
    model.to(device)

    def _inference(_, batch):
        model.eval()
        with torch.no_grad():
            tsdf, idx, quality, quat = utils.prepare_batch(batch, device)
            quality_out, quat_out = model(tsdf)
            quality_pred, quat_pred = utils.select_pred(quality_out, quat_out, idx)
        return quality_pred, quality, quat_pred, quat

    engine = Engine(_inference)

    return engine


def create_summary_writers(model, data_loader, device, log_dir):
    train_path = os.path.join(log_dir, "train")
    val_path = os.path.join(log_dir, "validation")

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    # Write the graph to Tensorboard
    trace, _, _, _ = utils.prepare_batch(next(iter(data_loader)), device)
    train_writer.add_graph(model, trace)

    return train_writer, val_writer


def train(args):
    device = torch.device("cuda")
    kwargs = {"pin_memory": True}

    # Create log directory for the current setup
    descr = "{},model={},data={},batch_size={},lr={:.0e}".format(
        datetime.now().strftime("%b%d_%H-%M-%S"),
        args.model,
        args.data,
        args.batch_size,
        args.lr,
    )
    if args.descr != "":
        descr += ",descr={}".format(args.descr)

    log_dir = os.path.join(args.log_dir, descr)

    assert not os.path.exists(log_dir), "log with this setup already exists"

    # Load and inspect data
    path = os.path.join("data", "datasets", args.data)
    dataset = VGNDataset(path, augment=False, rebuild_cache=args.rebuild_cache)

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
        validation_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    # Build the network
    model = models.get_network(args.model)

    # Define loss functions
    quality_loss_fn = F.binary_cross_entropy
    quat_loss_fn = torch.nn.L1Loss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create Ignite engines for training and validation
    trainer = create_trainer(model, optimizer, quality_loss_fn, quat_loss_fn, device)
    evaluator = create_evaluator(model, device)

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
        model, train_loader, device, log_dir
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
        "model",
        save_interval=10,
        n_saved=100,
        require_empty=True,
        save_as_state_dict=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, to_save={"model": model}
    )

    # Run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model")
    parser.add_argument("--data", type=str, required=True, help="name of dataset")
    parser.add_argument(
        "--log-dir", type=str, default="data/runs", help="path to log directory"
    )
    parser.add_argument(
        "--descr", type=str, default="", help="description appended to the run dirname"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="ratio of data used for validation (default: 0.2)",
    )
    parser.add_argument("--rebuild-cache", action="store_true")

    args = parser.parse_args()

    assert torch.cuda.is_available(), "ERROR: cuda is not available"

    train(args)


if __name__ == "__main__":
    main()
