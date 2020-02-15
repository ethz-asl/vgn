from ignite.engine import Engine
import torch
from torch.utils import tensorboard


def prepare_batch(batch, device):
    tsdf, (qual, rot, width), mask = batch

    tsdf = tsdf.to(device)
    qual = qual.to(device)
    rot = rot.to(device)
    width = width.to(device)
    mask = mask.to(device)

    return tsdf, (qual, rot, width), mask


def create_trainer(net, optimizer, loss_fn, metrics, device):
    net.to(device)

    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # Forward
        x, y, mask = prepare_batch(batch, device)
        y_pred = net(x)
        losses = loss_fn(y_pred, y, mask)

        # Backward
        losses[0].backward()
        optimizer.step()

        return x, y_pred, y, mask, losses

    engine = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_evaluator(net, loss_fn, metrics, device):
    net.to(device)

    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, mask = prepare_batch(batch, device)
            y_pred = net(x)
            losses = loss_fn(y_pred, y, mask)

        return x, y_pred, y, mask, losses

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer
