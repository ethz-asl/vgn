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
    loss0 = _quat_loss_fn(pred, target[:, 0])
    loss1 = _quat_loss_fn(pred, target[:, 1])
    loss = torch.min(loss0, loss1)

    return (loss * mask).sum() / mask.sum()


def width_loss_fn(pred, target, mask):
    loss = F.mse_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum()


def _quat_loss_fn(pred, target):
    """Compute voxel-wise quat loss between pred and target volumes.
    
    Args:
        pred, target: Tensors of shape Bx4x40x40x40 where dim 1 holds a quaternion.
    
    Returns:
        A Bx1x40x40x40 tensor containing the voxel-wise loss.
    """
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1, keepdim=True))
