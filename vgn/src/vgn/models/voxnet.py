import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchviz


class VoxNet(nn.Module):
    def __init__(self):
        super(VoxNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, 7, padding=3)
        self.conv2 = nn.Conv3d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv3d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        score_out = torch.sigmoid(self.conv3(x))

        return score_out


if __name__ == '__main__':
    device = torch.device('cuda')

    net = VoxNet().to(device)
    voxel_grid = torch.randn(32, 1, 60, 60, 60).to(device)
    out = net(voxel_grid)

    assert out.shape[0] == voxel_grid.shape[0], 'batch size is not the same'
    assert out.shape[1] == 1, 'number of output channels is wrong'
    assert out.shape[2:] == voxel_grid.shape[2:], 'voxel dimensions are wrong'

    raw_input('Press any key to continue')
