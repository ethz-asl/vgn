from pathlib2 import Path

import numpy as np
import pandas
from scipy import ndimage
import torch.utils.data

from vgn.grasp import Grasp
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, root, size=0.3, resolution=40, augment=False, reconstruction="partial"
    ):
        self.root = root
        csv_path = self.root / "grasps.csv"
        assert csv_path.exists()

        self.df = pandas.read_csv(csv_path)
        self.size = size
        self.resolution = resolution
        self.augment = augment
        self.reconstruction_mode = reconstruction

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id, ori, pos, width, label = self.lookup_sample(i)
        tsdf = self.read_tsdf(scene_id)

        if self.augment:
            tsdf, ori, pos = self._apply_random_transform(tsdf, ori, pos)

        index = np.round(pos).astype(np.long)
        rotations = np.empty((2, 4), dtype=np.float32)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        x, y, index = tsdf, (label, rotations, width), index

        return x, y, index

    def lookup_sample(self, i):
        voxel_size = self.size / self.resolution
        scene_id = self.df.loc[i, "scene_id"]
        orientation = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.double))
        position = self.df.loc[i, "x":"z"].to_numpy(np.double) / voxel_size
        width = self.df.loc[i, "width"] / voxel_size
        label = self.df.loc[i, "label"]

        return scene_id, orientation, position, width, label

    def read_tsdf(self, scene_id):
        tsdf_path = self.root / "tsdfs" / (scene_id + ".npz")
        return np.load(str(tsdf_path))[self.reconstruction_mode]

    def read_pc(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        tsdf = TSDFVolume(self.size, 120)
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)  # TODO
        raw = np.load(self.root / "raw" / (str(scene_id) + ".npz"))

        N = raw["extrinsics"].shape[0]
        n = N if self.reconstruction_mode == "complete" else raw["n"]

        for i in range(n):
            extrinsic = Transform.from_list(raw["extrinsics"][i])
            depth_img = raw["depth_imgs"][i]
            tsdf.integrate(depth_img, intrinsic, extrinsic)
        return tsdf.extract_point_cloud()

    def _apply_random_transform(self, tsdf, orientation, position):
        angle = np.pi / 2.0 * np.random.choice(4)
        R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

        z_offset = np.random.uniform(6, 34) - position[2]

        t_augment = np.r_[0.0, 0.0, z_offset]
        T_augment = Transform(R_augment, t_augment)

        T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
        T = T_center * T_augment * T_center.inverse()

        # transform tsdf
        T_inv = T.inverse()
        matrix, offset = T_inv.rotation.as_dcm(), T_inv.translation
        tsdf[0] = ndimage.affine_transform(tsdf[0], matrix, offset, order=0)

        # transform grasp pose
        position = T.transform_point(position)
        orientation = T.rotation * orientation

        return tsdf, orientation, position
