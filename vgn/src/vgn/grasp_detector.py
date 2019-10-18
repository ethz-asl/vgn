import numpy as np
import torch

from vgn.networks import get_network, predict


class GraspDetector(object):
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = get_network(model_path.name.split("_")[1]).to(self.device)
        self.net.load_state_dict(torch.load(model_path))

    def detect_grasps(self, tsdf):
        """Returns a list of grasps.
        
        Args:
            tsdf (np.ndarray)

        Return:
            List of grasp candidates, and an info dict containing the intermediate steps.
        """
        grasps, info = [], {}

        # Predict grasp quality map
        quality_grid, quat_grid = predict(self.net, self.device, tsdf)
        info["quality_out"] = quality_grid.copy()

        # Filter grasp quality map

        # Cluster

        # Sort by their scores
        return grasps, info

    def sample_grasps(self, tsdf, n):
        """Importance sample grasps from the predicted grasp quality map.
        """
        grasps = []
        return grasps
