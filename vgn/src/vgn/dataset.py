import numpy as np
import pandas
from scipy import ndimage
import torch.utils.data

from vgn.utils.transform import Rotation, Transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, augment=False):
        self.root = root
        self.df = pandas.read_csv(self.root / "grasps.csv")
        self._augment = augment

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        path = self.df.iloc[i, 0]
        tsdf = np.load(str(self.root / path))["tsdf"]
        index = self.df.iloc[i, 1:4].to_numpy(dtype=np.long)
        rotation = Rotation.from_quat(self.df.iloc[i, 4:8].to_numpy())
        width = self.df.iloc[i, 8]
        label = self.df.iloc[i, 9]

        if self._augment:
            tsdf, index, rotation = self._apply_random_transform(tsdf, index, rotation)

        rotations = np.empty((2, 4), dtype=np.float32)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = rotation.as_quat()
        rotations[1] = (rotation * R).as_quat()

        x, y, index = tsdf, (label, rotations, width), index

        return x, y, index

    def _apply_random_transform(self, tsdf, index, rotation):
        # center sample at grasp point
        T_center = Transform(Rotation.identity(), index)
        # sample random transform
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])
        t_augment = 20 - index + np.random.uniform(-14, 14, size=(3,))
        T_augment = Transform(R_augment, t_augment)
        T = T_center * T_augment * T_center.inverse()
        # transform tsdf
        T_inv = T.inverse()
        matrix, offset = T_inv.rotation.as_dcm(), T_inv.translation
        tsdf[0] = ndimage.affine_transform(tsdf[0], matrix, offset, order=1)
        # transform grasp pose
        index = np.round(T.transform_point(index)).astype(np.long)
        rotation = T.rotation * rotation
        return tsdf, index, rotation
