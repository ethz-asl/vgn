import numpy as np
import pandas
import torch.utils.data

from vgn.utils.transform import Rotation, Transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=[]):
        self.root = root
        self.transforms = transforms
        self.df = pandas.read_csv(self.root / "grasps.csv")

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        path = self.df.iloc[i, 0]
        tsdf = np.load(str(self.root / path))["tsdf"]
        index = self.df.iloc[i, 1:4].to_numpy(dtype=np.long)
        rotation = Rotation.from_quat(self.df.iloc[i, 4:8].to_numpy())
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations = np.empty((2, 4), dtype=np.float32)
        rotations[0] = rotation.as_quat()
        rotations[1] = (rotation * R).as_quat()
        width = self.df.iloc[i, 8]
        label = self.df.iloc[i, 9]

        x, y, index = tsdf, (label, rotations, width), index

        for transform in self.transforms:
            x, y, index = transform(x, y, index)

        return x, y, index
