import cv2
import ignite.engine
import ignite.metrics
import numpy as np
import torch


def save_image(fname, img):
    """Save image as a PNG file."""
    img = (1000. * img).astype(np.uint16)
    cv2.imwrite(fname, img)


def load_image(fname):
    """Load image from a PNG file."""
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) * 0.001
    return img


def _prepare_batch(batch, device):
    tsdf, idx, score = batch
    tsdf = tsdf.to(device)
    idx = idx.to(device)
    score = score.squeeze().to(device)
    return tsdf, idx, score


def _select_pred(out, idx):
    score_pred = torch.cat(
        [t[0, i, j, k].unsqueeze(0) for t, (i, j, k) in zip(out, idx)])
    return score_pred


def create_trainer(model, optimizer, loss_fn, device):
    model.to(device)

    def _update(_, batch):
        model.train()
        optimizer.zero_grad()
        tsdf, idx, score = _prepare_batch(batch, device)
        out = model(tsdf)
        score_pred = _select_pred(out, idx)
        loss = loss_fn(score_pred, score)
        loss.backward()
        optimizer.step()
        return loss.item()

    return ignite.engine.Engine(_update)


def create_evaluator(model, loss_fn, device):
    model.to(device)

    def _inference(_, batch):
        model.eval()
        with torch.no_grad():
            tsdf, idx, score = _prepare_batch(batch, device)
            out = model(tsdf)
            score_pred = _select_pred(out, idx)
            return score_pred, score

    engine = ignite.engine.Engine(_inference)

    loss_metric = ignite.metrics.Loss(loss_fn)
    loss_metric.attach(engine, 'loss')

    return engine
