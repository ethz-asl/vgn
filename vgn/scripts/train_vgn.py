import argparse
from pathlib2 import Path
from datetime import datetime

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy
import torch
from torch.utils import tensorboard
import torch.nn.functional as F

from vgn.dataset import Dataset
from vgn.networks import get_network


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    # create log directory
    time_stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    description = "{},net={},dataset={},batch_size={},lr={:.0e},{}".format(
        time_stamp,
        args.net,
        args.dataset_dir.name,
        args.batch_size,
        args.lr,
        args.description,
    )
    log_dir = args.log_dir / description

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset_dir, args.batch_size, args.split, kwargs
    )

    # build the network
    net = get_network(args.net).to(device)

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    metrics = {
        "loss": Average(lambda out: out[3]),
        "accuracy": Accuracy(lambda out: (torch.round(out[1][0]), out[2][0])),
    }

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(net, loss_fn, metrics, device)

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True).attach(trainer)

    train_writer, val_writer = create_summary_writers(net, device, log_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("accuracy", metrics["accuracy"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("accuracy", metrics["accuracy"], epoch)

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        str(log_dir),
        "vgn",
        save_interval=10,
        n_saved=100,
        require_empty=True,
        save_as_state_dict=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, to_save={args.net: net}
    )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def create_train_val_loaders(root, batch_size, split, kwargs):
    # load the dataset
    dataset = Dataset(root)

    # split into train and validation sets
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, **kwargs
    )

    return train_loader, val_loader


def prepare_batch(batch, device):
    tsdf, (label, rot, width), index = batch
    tsdf = tsdf.to(device)
    label = label.float().to(device)
    rot = rot.to(device)
    width = width.to(device)
    index = index.to(device)
    return tsdf, (label, rot, width), index


def select(out, index):
    qual_out, rot_out, width_out = out
    batch_index = torch.arange(qual_out.shape[0])
    label = qual_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]]
    rot = rot_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]]
    width = width_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]]
    return label, rot, width


def loss_fn(y_pred, y):
    label_pred, rot_pred, width_pred = y_pred
    label, rot, width = y

    loss_qual = _qual_loss_fn(label_pred, label)
    # loss_rot = _rot_loss_fn(rot_pred, rot)
    # loss_width = _width_loss_fn(width_pred, width)

    loss = loss_qual  # + label * (loss_rot + 0.1 * loss_width)

    return loss


def _qual_loss_fn(pred, target):
    loss = F.binary_cross_entropy(pred, target)
    return loss


def create_trainer(net, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # forward
        x, y, index = prepare_batch(batch, device)
        y_pred = select(net(x), index)
        loss = loss_fn(y_pred, y)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, index = prepare_batch(batch, device)
            y_pred = select(net(x), index)
            loss = loss_fn(y_pred, y)
        return x, y_pred, y, loss

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default="data/models/logs")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--net", default="conv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()

    main(args)
