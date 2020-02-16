import torch
import torch.nn.functional as F


def loss_fn(y_pred, y, mask):
    loss_qual = qual_loss_fn(y_pred[0], y[0], mask)
    loss_rot = rot_loss_fn(y_pred[1], y[1], mask)
    loss_width = width_loss_fn(y_pred[2], y[2], mask)

    loss = loss_qual + loss_rot + loss_width

    return loss, loss_qual, loss_rot, loss_width


def qual_loss_fn(pred, target, mask):
    loss = F.binary_cross_entropy(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum()


def rot_loss_fn(pred, target, mask):
    loss = 1 - torch.abs(torch.sum(pred * target, dim=1, keepdim=True))
    return (loss * mask).sum() / mask.sum()


def width_loss_fn(pred, target, mask):
    loss = F.mse_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum()
