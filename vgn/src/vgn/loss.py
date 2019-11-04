import torch.nn.functional as F


def quality_loss_fn(out):
    pred, target, mask = out["pred_quality"], out["target_quality"], out["mask"]
    loss = F.binary_cross_entropy(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum()


def quat_loss_fn(out):
    pred, target, mask = out["pred_quat"], out["target_quat"], out["target_quality"]
    loss = F.l1_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum()


def loss_fn(out):
    quality_loss = quality_loss_fn(out)
    quat_loss = quat_loss_fn(out)
    loss = quality_loss + quat_loss
    return loss
