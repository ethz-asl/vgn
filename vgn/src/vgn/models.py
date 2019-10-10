import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def get_network(name):
    models = {"conv": ConvNet()}
    return models[name.lower()]

    # def predict(self, tsdf):
    #     tsdf_in = torch.from_numpy(tsdf).unsqueeze(0).to(self.device)
    #     with torch.no_grad():
    #         score_out, quat_out = self.model(tsdf_in)
    #     score_out = score_out.squeeze().cpu().numpy()
    #     quat_out = quat_out.squeeze().cpu().numpy()
    #     quat_out = np.swapaxes(quat_out, 0, 1)
    #     return score_out, quat_out


#  process_input()

# def process_output(score_out, quat_out):
#     grasp_map = out.copy()
#     print(out.shape)
#     print(tsdf.shape)
#     grasp_map[tsdf == 0.0] = 0.0
#     grasp_map = ndimage.gaussian_filter(grasp_map, sigma=4.0)
#     grasp_map[grasp_map < threshold] = 0.0
#     return grasp_map


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

        self.conv_score = nn.Conv3d(filter_sizes[5], 1, kernel_size=5, padding=2)

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

        score = self.conv_score(x)
        score_out = torch.sigmoid(score)

        quat = self.conv_quat(x)
        quat_out = F.normalize(quat, dim=1)

        return score_out, quat_out
