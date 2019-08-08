from __future__ import print_function

import argparse
import os

import open3d
import torch
import torch.nn.functional as F
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint

import vgn.utils as utils
from vgn.dataset import VGNDataset
from vgn.models import get_model


def loss_fn(score_pred, score):
    loss = F.binary_cross_entropy(score_pred, score)
    return loss


def train(args):
    device = torch.device('cuda')
    kwargs = {'pin_memory': True}

    dataset = VGNDataset(args.data, args.rebuild_cache)

    train_size = int((1 - args.validation_split) * len(dataset))
    validation_size = int(args.validation_split * len(dataset))

    print('Size of training dataset: {}'.format(train_size))
    print('Size of validation dataset: {}'.format(validation_size))

    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    **kwargs)

    model = get_model(args.model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = utils.create_trainer(model, optimizer, loss_fn, device)

    evaluator = utils.create_evaluator(model, loss_fn, device)

    descr = 'model={},batch_size={},lr={:.0E},seed={}'.format(
        args.model,
        args.batch_size,
        args.lr,
        args.seed,
    )
    log_dir = os.path.join(args.log_dir, descr)

    train_writer, val_writer = utils.create_summary_writers(
        model, train_loader, device, log_dir)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_progress(engine):
        epoch = engine.state.epoch
        iteration = (engine.state.iteration - 1) % len(train_loader) + 1
        loss = engine.state.output[0]

        if iteration % args.log_interval == 0:
            print('Epoch: {:2d}, Iteration: {}/{}, Loss: {:.4f}'.format(
                epoch, iteration, len(train_loader), loss))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(_):
        evaluator.run(validation_loader)

        epoch = trainer.state.epoch
        train_loss = trainer.state.metrics['loss']
        val_loss = evaluator.state.metrics['loss']

        train_writer.add_scalar('epoch_loss', train_loss, epoch)
        val_writer.add_scalar('epoch_loss', val_loss, epoch)

        print(('Validation Results - Epoch: {}, '
               'Avg loss: {:.4f}').format(epoch, val_loss))

    checkpoint_handler = ModelCheckpoint(
        log_dir,
        'best',
        score_function=lambda engine: -engine.state.metrics['loss'],
        n_saved=1,
        require_empty=True,
        save_as_state_dict=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        to_save={'model': model},
    )

    # Run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='voxnet',
        help='model',
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='path to train dataset',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='input batch size for training',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='number of epochs to train',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed',
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='ratio of data used for validation',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        help='path to log directory',
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='number of batches to wait before logging training status',
    )
    parser.add_argument(
        '--rebuild-cache',
        action='store_true',
    )

    args = parser.parse_args()

    assert torch.cuda.is_available(), 'ERROR: cuda is not available'

    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
