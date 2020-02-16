import numpy as np
import torch.utils.data


class VgnDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=[]):
        """Dataset for the volumetric grasping network.

        Args:
            root: Root directory of the dataset.
        """
        self.root_dir = root_dir
        self.transforms = transforms

        self._detect_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.root_dir / self.samples[idx]
        sample = np.load(path)
        x = sample["tsdf_vol"]
        y = (sample["qual_vol"], sample["rot_vol"], sample["width_vol"])
        mask = sample["mask"]

        for transform in self.transforms:
            x, y, mask = transform(x, y, mask)

        return x, y, mask

    def _detect_samples(self):
        self.samples = [f.name for f in self.root_dir.iterdir() if f.suffix == ".npz"]


class Rescale(object):
    def __init__(self, width_scale):
        self.width_scale = width_scale

    def __call__(self, x, y, mask):
        qual, rot, width = y
        width *= self.width_scale
        return x, (qual, rot, width), mask
