import numpy as np
from scipy import ndimage
import torch

from vgn.constants import vgn_res
from vgn.grasp import Grasp
from vgn.utils.transform import Transform, Rotation
from vgn.networks import get_network, predict


class GraspDetector(object):
    def __init__(self, model_path, vol_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = get_network(model_path.name.split("_")[1]).to(self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))

        self.voxel_size = vol_size / vgn_res

    def detect_grasps(self, tsdf_vol, threshold=0.9):
        """Returns a list of detected grasps.
        
        Args:
            tsdf_vol
            threshold

        Return:
            List of grasp candidates, predicted qualities, and an info dict containing the intermediate steps.
        """
        grasps, qualities, info = [], [], {}

        # Predict grasp quality map
        quality_vol, quat_vol = predict(tsdf_vol, self.net, self.device)
        info["output"] = quality_vol.copy()

        # Filter grasp quality map
        quality_vol[quality_vol < threshold] = 0.0
        info["filtered"] = quality_vol.copy()

        # Non-maxima suppression
        max_vol = ndimage.maximum_filter(quality_vol, size=5)
        quality_vol = np.where(quality_vol == max_vol, quality_vol, 0.0)
        info["non_maximum_suppression"] = quality_vol.copy()

        # Sort by their scores
        for (i, j, k) in np.argwhere(quality_vol):
            position = self.voxel_size * np.r_[i, j, k]
            orientation = Rotation.from_quat(quat_vol[i, j, k])
            grasps.append(Grasp(Transform(orientation, position)))
            qualities.append(quality_vol[i, j, k])

        return grasps, qualities, info

    def sample_grasps(self, tsdf, n):
        """Importance sample grasps from the predicted grasp quality map.
        """
        grasps = []
        return grasps
