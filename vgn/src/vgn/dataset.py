from pathlib import Path

import open3d
import numpy as np
import torch.utils.data
from tqdm import tqdm

import vgn.config as cfg
from vgn.grasp import Label
from vgn.perception.integration import TSDFVolume
from vgn.utils.data import SceneData
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

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_dir = self.scenes[idx]
        data = np.load(self.cache_dir / (scene_dir.name + ".npz"))
        return (
            data["input_tsdf_vol"],
            data["target_quality_vol"],
            data["target_quat_vol"],
            data["mask"],
        )

    def detect_scenes(self):
        self.scenes = [
            d for d in self.root_dir.iterdir() if d.is_dir() and d != self.cache_dir
        ]

    def build_cache(self):
        print("Building cache:")
        self.cache_dir.mkdir(exist_ok=True)

        # Iterate through all scenes and verify whether it needs to be processed
        for scene_dir in tqdm(self.scenes, ascii=True):
            p = self.cache_dir / (scene_dir.name + ".npz")
            if not p.exists() or self.rebuild_cache:
                # Load the data and build the TSDF
                scene = SceneData.load(scene_dir)
                tsdf = TSDFVolume(cfg.size, cfg.resolution)
                for depth_img, extrinsic in zip(scene.depth_imgs, scene.extrinsics):
                    tsdf.integrate(depth_img, scene.intrinsic, extrinsic)
                tsdf_vol = tsdf.get_volume()

                # Construct input
                input_tsdf_vol = np.expand_dims(tsdf_vol, 0)

                # Construct targets
                vol_shape = (tsdf.resolution, tsdf.resolution, tsdf.resolution)

                mask = np.zeros((1,) + vol_shape, dtype=np.float32)
                target_quality_vol = np.zeros((1,) + vol_shape, dtype=np.float32)
                target_quat_vol = np.zeros((4,) + vol_shape, dtype=np.float32)

                for grasp, label in zip(scene.grasps, scene.labels):
                    i, j, k = tsdf.get_index(grasp.pose.translation)
                    mask[0, i, j, k] = 1.0
                    target_quality_vol[0, i, j, k] = VGNDataset.label2quality(label)
                    target_quat_vol[:, i, j, k] = grasp.pose.rotation.as_quat()

                np.savez_compressed(
                    p,
                    input_tsdf_vol=input_tsdf_vol,
                    target_quality_vol=target_quality_vol,
                    target_quat_vol=target_quat_vol,
                    mask=mask,
                )

    @staticmethod
    def label2quality(label):
        quality = 1.0 if label == Label.SUCCESS else 0.0
        return quality


if __name__ == "__main__":
    # Use for debugging
    scene_dir = Path("data/datasets/debug/foo")
    print(scene_dir.absolute())
    dataset = VGNDataset(scene_dir.parent, rebuild_cache=True)

    index = dataset.scenes.index(scene_dir)
    input_tsdf_vol, target_quality_vol, target_quat_vol, mask = dataset[index]
