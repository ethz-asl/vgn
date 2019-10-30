import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def get_network(name):
    models = {"conv": ConvNet()}
    return models[name.lower()]


def predict(tsdf_vol, net, device):
    # Move data to the PyTorch device
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).unsqueeze(0).to(device)

    # Predict grasp qualities and orientations
    with torch.no_grad():
        quality_vol, quat_vol = net(tsdf_vol)

    # Move data back to the CPU
    quality_vol = quality_vol.squeeze().cpu().numpy()
    quat_vol = quat_vol.cpu().numpy()  # TODO swap axes

    return quality_vol, quat_vol


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        input_channels = 1
        filter_sizes = [16, 32, 32, 32, 32, 16]

        self.conv1 = nn.Conv3d(
            input_channels, filter_sizes[0], kernel_size=5, padding=2
        )
        self.conv2 = nn.Conv3d(
            filter_sizes[0], filter_sizes[1], kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv3d(
            filter_sizes[1], filter_sizes[2], kernel_size=3, padding=1
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

        self.conv_quality = nn.Conv3d(filter_sizes[5], 1, kernel_size=5, padding=2)

        self.conv_quat = nn.Conv3d(filter_sizes[5], 4, kernel_size=5, padding=2)

    def forward(self, x):
        # 1 x 40 x 40 x 40
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)

        # 16 x 20 x 20 x 20
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)

        # 32 x 10 x 10 x 10
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool3d(x, 2)

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

        quality = self.conv_quality(x)
        quality_out = torch.sigmoid(quality)

        quat = self.conv_quat(x)
        quat_out = F.normalize(quat, dim=1)

        return quality_out, quat_out
