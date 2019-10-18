from __future__ import division, print_function

import json

import numpy as np
import torch.utils.data
from scipy import ndimage
from tqdm import tqdm

import vgn.config as cfg
from vgn.grasp import Label
from vgn import utils
from vgn.utils import data
from vgn.perception import integration
from vgn.utils.transform import Rotation, Transform


class VGNDataset(torch.utils.data.Dataset):
    def __init__(self, root, rebuild_cache=False):
        """Dataset for the volumetric grasping network.

        The mapping between grasp label and target grasp quality is defined
        by the `label2quality` method.

        Args:
            root: Root directory of the dataset.
            rebuild_cache: Discard cached volumes.
        """
        self.root_dir = root
        self.rebuild_cache = rebuild_cache
        self.cache_dir = self.root_dir / "cache"

        self.detect_scenes()
        self.build_cache()

    @staticmethod
    def label2quality(label):
        quality = 1.0 if label == Label.SUCCESS else 0.0
        return quality

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_dir = self.scenes[idx]
        data = np.load(self.cache_dir / (scene_dir.name + ".npz"))

        tsdf = data["tsdf"]
        indices = data["indices"]
        quats = np.swapaxes(data["quats"], 0, 1)
        qualities = data["qualities"]

        return np.expand_dims(tsdf, 0), indices, quats, qualities

    def detect_scenes(self):
        self.scenes = [
            d for d in self.root_dir.iterdir() if d.is_dir() and d != self.cache_dir
        ]

    def build_cache(self):
        print("Verifying cache:")
        self.cache_dir.mkdir(exist_ok=True)

        for scene_dir in tqdm(self.scenes, ascii=True):
            p = self.cache_dir / (scene_dir.name + ".npz")
            if not p.exists() or self.rebuild_cache:
                # Load the scene data and reconstruct the TSDF
                scene = data.SceneData.load(scene_dir)
                _, voxel_grid = integration.reconstruct_scene(
                    scene.intrinsic,
                    scene.extrinsics,
                    scene.depth_imgs,
                    resolution=cfg.resolution,
                )

                # Store the input TSDF and targets as tensors
                tsdf = utils.voxel_grid_to_array(voxel_grid, cfg.resolution)
                indices = np.empty((scene.n_grasp_attempts, 3), dtype=np.long)
                quats = np.empty((scene.n_grasp_attempts, 4), dtype=np.float32)
                for i, grasp in enumerate(scene.grasps):
                    index = voxel_grid.get_voxel(grasp.pose.translation)
                    indices[i] = np.clip(index, [0, 0, 0], [cfg.resolution - 1] * 3)
                    quats[i] = grasp.pose.rotation.as_quat()
                qualities = np.asarray(
                    [VGNDataset.label2quality(l) for l in scene.labels],
                    dtype=np.float32,
                )

                np.savez_compressed(
                    p, tsdf=tsdf, indices=indices, quats=quats, qualities=qualities
                )
