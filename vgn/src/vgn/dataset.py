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
from vgn.perception.integration import TSDFVolume
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

        tsdf_vol = data["tsdf_vol"]
        indices = data["indices"]
        quats = np.swapaxes(data["quats"], 0, 1)
        qualities = data["qualities"]

        return tsdf_vol, indices, quats, qualities

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
                tsdf = TSDFVolume(cfg.size, cfg.resolution)
                for depth_img, extrinsic in zip(scene.depth_imgs, scene.extrinsics):
                    tsdf.integrate(depth_img, scene.intrinsic, extrinsic)
                tsdf_vol = tsdf.get_volume()

                # Store the input TSDF and targets as tensors
                indices = np.empty((scene.n_grasp_attempts, 3), dtype=np.long)
                quats = np.empty((scene.n_grasp_attempts, 4), dtype=np.float32)
                for i, grasp in enumerate(scene.grasps):
                    index = tsdf.get_index(grasp.pose.translation)
                    indices[i] = np.clip(index, [0, 0, 0], [tsdf.resolution - 1] * 3)
                    quats[i] = grasp.pose.rotation.as_quat()
                qualities = np.asarray(
                    [VGNDataset.label2quality(l) for l in scene.labels],
                    dtype=np.float32,
                )

                np.savez_compressed(
                    p,
                    tsdf_vol=np.expand_dims(tsdf_vol, 0),  # add channel dimension
                    indices=indices,
                    quats=quats,
                    qualities=qualities,
                )
