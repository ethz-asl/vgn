from __future__ import division, print_function

import json
import os

import numpy as np
import torch.utils.data
from scipy import ndimage
from tqdm import tqdm

import vgn.config as cfg
from vgn import data, utils
from vgn.utils.transform import Rotation, Transform


class VGNDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, augment=True, rebuild_cache=False):
        """Dataset for the volumetric grasping network.

        Args:
            root_dir: Path to the synthetic grasp dataset.
            augment: Augment data by translations, rotations, and flipping.
            rebuild_cache: Discard cached volumes.
        """
        self.root_dir = root_dir
        self.augment = augment
        self.rebuild_cache = rebuild_cache

        self.detect_scenes()
        self.build_cache()

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        data = np.load(os.path.join(self.cache_dir, scene) + '.npz')

        tsdf = data['tsdf']
        indices = data['indices']
        scores = data['scores']
        quats = np.swapaxes(data['quats'], 0, 1)

        if self.augment:
            # TODO fix me
            center = np.mean(indices, 0)
            spread = np.max(indices, 0) - np.min(indices, 0)
            T_center = Transform(Rotation.identity(), center)

            while True:
                rotation = Rotation.random()
                translation = cfg.resolution / 2. - center
                translation += np.random.uniform(-spread / 2., spread / 2.)
                T_augment = Transform(rotation, translation)

                T = T_center * T_augment * T_center.inverse()

                indices = [T.apply_to_point(index) for index in indices]
                indices = np.round(indices).astype(np.long)

                if np.all(indices >= 0) and np.all(indices < cfg.resolution):
                    break

            T_inv = T.inverse()
            matrix, offset = T_inv.rotation.as_dcm(), T_inv.translation
            tsdf = ndimage.affine_transform(tsdf, matrix, offset, order=2)

        return np.expand_dims(tsdf, 0), indices, scores, quats

    @property
    def cache_dir(self):
        return os.path.join(self.root_dir, 'cache')

    def detect_scenes(self):
        self.scenes = []
        for d in sorted(os.listdir(self.root_dir)):
            path = os.path.join(self.root_dir, d)
            if os.path.isdir(path) and path != self.cache_dir:
                self.scenes.append(d)

    def build_cache(self):
        print('Verifying cache:')

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        for dirname in tqdm(self.scenes):

            path = os.path.join(self.cache_dir, dirname) + '.npz'
            if not os.path.exists(path) or self.rebuild_cache:
                scene = data.load_scene(os.path.join(self.root_dir, dirname))
                _, voxel_grid = data.reconstruct_volume(scene)
                tsdf = utils.voxel_grid_to_array(voxel_grid, cfg.resolution)

                indices = np.empty((len(scene['poses']), 3), dtype=np.long)
                scores = np.asarray(scene['scores'], dtype=np.float32)
                quats = np.empty((len(scene['poses']), 4), dtype=np.float32)
                for i, pose in enumerate(scene['poses']):
                    index = voxel_grid.get_voxel(pose.translation)
                    indices[i] = np.clip(index, [0, 0, 0],
                                         [cfg.resolution - 1] * 3)
                    quats[i] = pose.rotation.as_quat()

                np.savez_compressed(
                    path,
                    tsdf=tsdf,
                    indices=indices,
                    scores=scores,
                    quats=quats,
                )
