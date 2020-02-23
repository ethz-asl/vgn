import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def get_network(name):
    models = {"conv": ConvNet()}
    return models[name.lower()]


class BaseNet(nn.Module):
    def count_num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConvNet(BaseNet):
    def __init__(
        self, filters=[16, 32, 64, 64, 32, 16], kernel_sizes=[5, 3, 3, 3, 3, 5]
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            1, filters[0], kernel_sizes[0], stride=2, padding=kernel_sizes[0] // 2
        )
        self.conv2 = nn.Conv3d(
            filters[0],
            filters[1],
            kernel_sizes[1],
            stride=2,
            padding=kernel_sizes[1] // 2,
        )
        self.conv3 = nn.Conv3d(
            filters[1],
            filters[2],
            kernel_sizes[2],
            stride=2,
            padding=kernel_sizes[2] // 2,
        )
        self.conv4 = nn.Conv3d(
            filters[2], filters[3], kernel_sizes[3], padding=kernel_sizes[3] // 2
        )
        self.conv5 = nn.Conv3d(
            filters[3], filters[4], kernel_sizes[4], padding=kernel_sizes[4] // 2
        )
        self.conv6 = nn.Conv3d(
            filters[4], filters[5], kernel_sizes[5], padding=kernel_sizes[5] // 2
        )

        self.conv_qual = nn.Conv3d(filters[5], 1, kernel_size=5, padding=2)
        self.conv_rot = nn.Conv3d(filters[5], 4, kernel_size=5, padding=2)
        self.conv_width = nn.Conv3d(filters[5], 1, kernel_size=5, padding=2)

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

