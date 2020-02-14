import numpy as np
import torch.utils.data


class VgnDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        """Dataset for the volumetric grasping network.

        Args:
            root: Root directory of the dataset.
        """
        self.root_dir = root

        self._detect_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.root_dir / self.samples[idx]
        sample = np.load(path)

        return (
            sample["tsdf_vol"],
            sample["qual_vol"],
            sample["rot_vol"],
            sample["width_vol"],
            sample["mask"],
        )

    def _detect_samples(self):
        self.samples = [f.name for f in self.root_dir.iterdir() if f.suffix == ".npz"]
