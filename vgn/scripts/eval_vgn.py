import argparse

import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from torch.utils.data.dataloader import DataLoader

from vgn.dataset import VGNDataset
from vgn.models import get_model
from vgn.utils import vis


def _prepare_batch(batch, device):
    tsdf, idx, score = batch
    tsdf = tsdf.to(device)
    idx = idx.to(device)
    score = score.squeeze().to(device)
    return tsdf, idx, score


def main(args):
    device = torch.device('cuda')
    model = get_model('conv').to(device)
    model.load_state_dict(torch.load(args.model))

    test_dataset = VGNDataset('data/datasets/cube/train')

    with torch.no_grad():
        idx = np.random.randint(len(test_dataset))
        tsdf, idx, score = test_dataset[idx]
        tsdf = torch.from_numpy(tsdf).unsqueeze(0).to(device)
        out = model(tsdf)

        vis.plot_tsdf(tsdf.squeeze().cpu().numpy())
        vis.plot_vgn(out.squeeze().cpu().numpy())
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
    )
    args = parser.parse_args()

    main(args)
