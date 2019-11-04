from ignite.engine import Engine
import torch
from torch.utils import tensorboard


def prepare_batch(batch, device):
    input_tsdf, target_quality, target_quat, mask = batch

    input_tsdf = input_tsdf.to(device)
    target_quality = target_quality.to(device)
    target_quat = target_quat.to(device)
    mask = mask.to(device)

    return {
        "input_tsdf": input_tsdf,
        "target_quality": target_quality,
        "target_quat": target_quat,
        "mask": mask,
    }


def create_trainer(net, optimizer, loss_fn, metrics, device):
    net.to(device)

    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # Forward
        batch = prepare_batch(batch, device)
        batch["pred_quality"], batch["pred_quat"] = net(batch["input_tsdf"])
        loss = loss_fn(batch)

        # Backward
        loss.backward()
        optimizer.step()

        return batch

    engine = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_evaluator(net, metrics, device):
    net.to(device)

    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, device)
            batch["pred_quality"], batch["pred_quat"] = net(batch["input_tsdf"])

        return batch

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
