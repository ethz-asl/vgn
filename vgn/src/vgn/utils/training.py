import numpy as np
from ignite.engine import Engine
import torch
from torch.utils import tensorboard

from vgn.dataset import VgnDataset, Rescale, RandomAffine


def create_train_val_loaders(dataset_dir, augment, batch_size, val_split, kwargs):
    if augment:
        train_transforms = [Rescale(width_scale=0.1), RandomAffine()]
    else:
        train_transforms = [Rescale(width_scale=0.1)]

    val_transforms = [Rescale(width_scale=0.1)]

    train_dataset = VgnDataset(dataset_dir, transforms=train_transforms)
    val_dataset = VgnDataset(dataset_dir, transforms=val_transforms)

    num_samples = len(train_dataset)
    indices = list(range(num_samples))
    val_size = int(val_split * num_samples)

    np.random.shuffle(indices)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, **kwargs
    )

    return train_loader, val_loader


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
