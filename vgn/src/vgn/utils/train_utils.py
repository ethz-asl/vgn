def prepare_batch(batch, device):
    tsdf, idx, score, quat = batch
    tsdf = tsdf.to(device)
    idx = idx.to(device)
    score = score.squeeze().to(device)
    quat = quat.to(device)
    return tsdf, idx, score, quat


def select_pred(scores, quats, indices):
    score_pred = torch.cat(
        [s[0, i[:, 0], i[:, 1], i[:, 2]].unsqueeze(0) for s, i in zip(scores, indices)]
    )
    quat_pred = torch.cat(
        [q[:, i[:, 0], i[:, 1], i[:, 2]].unsqueeze(0) for q, i in zip(quats, indices)]
    )
    return score_pred, quat_pred


def compute_loss():
    pass
