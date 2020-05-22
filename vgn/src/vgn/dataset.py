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
        index = self.df.iloc[i, 1:4].to_numpy(dtype=np.int32)
        rotation = self.df.iloc[i, 4:8].to_numpy(dtype=np.float32)
        width = self.df.iloc[i, 8]
        label = self.df.iloc[i, 9]

        x, y, index = tsdf, (rotation, width, label), index

        for transform in self.transforms:
            x, y, index = transform(x, y, index)

        return x, y, index
