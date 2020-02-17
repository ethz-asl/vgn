import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def get_network(name):
    models = {"conv": ConvNet()}
    return models[name.lower()]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        input_channels = 1
        filter_sizes = [16, 32, 64, 64, 32, 16]

        self.conv1 = nn.Conv3d(
            input_channels, filter_sizes[0], kernel_size=5, stride=2, padding=2,
        )
        self.conv2 = nn.Conv3d(
            filter_sizes[0], filter_sizes[1], kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv3d(
            filter_sizes[1], filter_sizes[2], kernel_size=3, stride=2, padding=1
        )
        self.conv4 = nn.Conv3d(
            filter_sizes[2], filter_sizes[3], kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv3d(
            filter_sizes[3], filter_sizes[4], kernel_size=3, padding=1
        )
        self.conv6 = nn.Conv3d(
            filter_sizes[4], filter_sizes[5], kernel_size=3, padding=1
        )

        self.conv_qual = nn.Conv3d(filter_sizes[5], 1, kernel_size=5, padding=2)
        self.conv_rot = nn.Conv3d(filter_sizes[5], 4, kernel_size=5, padding=2)
        self.conv_width = nn.Conv3d(filter_sizes[5], 1, kernel_size=5, padding=2)

    def forward(self, x):
        # 1 x 40 x 40 x 40
        x = self.conv1(x)
        x = F.relu(x)

        # 16 x 20 x 20 x 20
        x = self.conv2(x)
        x = F.relu(x)

        # 32 x 10 x 10 x 10
        x = self.conv3(x)
        x = F.relu(x)

        # 32 x 5 x 5 x 5
        x = self.conv4(x)
        x = F.relu(x)

        # 32 x 5 x 5 x 5
        x = F.interpolate(x, 10)
        x = self.conv5(x)
        x = F.relu(x)

        # 32 x 10 x 10 x 10
        x = F.interpolate(x, 20)
        x = F.relu(self.conv6(x))

        # 16 x 20 x 20 x 20
        x = F.interpolate(x, 40)

        qual_out = torch.sigmoid(self.conv_qual(x))
        rot_out = F.normalize(self.conv_rot(x), dim=1)
        width_out = self.conv_width(x)

        return qual_out, rot_out, width_out

