import time

import numpy as np
from scipy import ndimage
import torch.utils.data

from vgn.utils.transform import Rotation, Transform


class VgnDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=[]):
        """Dataset for the volumetric grasping network.

        Args:
            data_dir: Root directory of the dataset.
        """
        self.data_dir = data_dir
        self.transforms = transforms

        self._detect_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.data_dir / self.samples[idx]
        sample = np.load(path)
        x = sample["tsdf_vol"]
        y = (sample["qual_vol"], sample["rot_vol"], sample["width_vol"])
        mask = sample["mask"]

        for transform in self.transforms:
            x, y, mask = transform(x, y, mask)

        return x, y, mask

    def _detect_samples(self):
        self.samples = [f.name for f in self.data_dir.iterdir() if f.suffix == ".npz"]


class Rescale(object):
    def __init__(self, width_scale):
        self.width_scale = width_scale

    def __call__(self, x, y, mask):
        qual, rot, width = y
        width *= self.width_scale
        return x, (qual, rot, width), mask


class RandomAffine(object):
    """Augment samples by a random translation and rotation about the gravity vector."""

    def __call__(self, x, y, mask):
        T = self._compute_transform(x)
        x = self._transform_input(x, T)
        y, mask = self._transform_targets(y, mask, T)
        return x, y, mask

    def _compute_transform(self, tsdf):
        center = np.argwhere(tsdf[0] > 0.0).mean(axis=0)

        T_center = Transform(Rotation.identity(), center)

        angle = np.random.uniform(0.0, 2.0 * np.pi)
        rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
        translation = (20.0 - center) + np.random.uniform(-15.0, 15.0, size=(3,))
        T_augment = Transform(rotation, translation)

        return T_center * T_augment * T_center.inverse()

    def _transform_input(self, x, T):
        T_inv = T.inverse()
        matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
        x[0] = ndimage.affine_transform(x[0], matrix, offset, order=1)
        return x

    def _transform_targets(self, y, mask, T):
        qual, rot, width = y

        qual_t = np.zeros_like(qual, dtype=np.float32)
        rot_t = np.zeros_like(rot, dtype=np.float32)
        width_t = np.zeros_like(width, dtype=np.float32)
        mask_t = np.zeros_like(mask, dtype=np.float32)

        for (i, j, k) in np.argwhere(mask[0] == 1.0):
            index_t = np.round(T.transform_point(np.r_[i, j, k])).astype(np.int)
            if np.any(index_t < 0) or np.any(index_t > 40 - 1):
                continue

            i_t, j_t, k_t = index_t
            rot0 = T.rotation * Rotation.from_quat(rot[0][:, i, j, k])
            rot1 = T.rotation * Rotation.from_quat(rot[1][:, i, j, k])

            qual_t[0, i_t, j_t, k_t] = qual[0, i, j, k]
            rot_t[0, :, i_t, j_t, k_t] = rot0.as_quat()
            rot_t[1, :, i_t, j_t, k_t] = rot1.as_quat()
            width_t[0, i_t, j_t, k_t] = width[0, i, j, k]
            mask_t[0, i_t, j_t, k_t] = 1.0

        return (qual_t, rot_t, width_t), mask_t
