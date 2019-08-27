import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(name):
    models = {
        'conv': ConvNet(),
    }
    return models[name.lower()]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        input_channels = 1
        filter_sizes = [16, 32, 32, 32, 32, 16]

        self.conv1 = nn.Conv3d(input_channels,
                               filter_sizes[0],
                               kernel_size=5,
                               padding=2)
        self.conv2 = nn.Conv3d(filter_sizes[0],
                               filter_sizes[1],
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv3d(filter_sizes[1],
                               filter_sizes[2],
                               kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv3d(filter_sizes[2],
                               filter_sizes[3],
                               kernel_size=3,
                               padding=1)
        self.conv5 = nn.Conv3d(filter_sizes[3],
                               filter_sizes[4],
                               kernel_size=3,
                               padding=1)
        self.conv6 = nn.Conv3d(filter_sizes[4],
                               filter_sizes[5],
                               kernel_size=3,
                               padding=1)

        self.conv_score = nn.Conv3d(filter_sizes[5],
                                    1,
                                    kernel_size=5,
                                    padding=2)

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
        x = self.conv_score(x)
        score_out = torch.sigmoid(x)

        # 1 x 40 x 40 x 40
        return score_out


if __name__ == '__main__':
    device = torch.device('cuda')
    model = get_model('conv').to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters:', n_params)

    trace = torch.randn(32, 1, 40, 40, 40).to(device)

    out = model(trace)

    assert out.shape[0] == trace.shape[0], 'batch size is not the same'
    assert out.shape[1] == 1, 'number of output channels is wrong'
    assert out.shape[2:] == trace.shape[2:], 'voxel dimensions are wrong'

    raw_input('Press any key to continue')
