import torch

outcome2quality = lambda outcome: outcome == grasp.Outcome.SUCCESS.value


def prepare_batch(batch, device):
    tsdf, idx, quality, quat = batch
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
