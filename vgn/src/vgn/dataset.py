import json
import logging
import os

import numpy as np
import torch.utils.data
from send2trash import send2trash
from tqdm import tqdm

from vgn import data, utils
from vgn.utils.transform import Rotation, Transform


class VGNDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, rebuild_cache=False):
        """Dataset for the volumetric grasping network.

        Instances consist of an input voxelized tsdf, and the position, score
        and orientation of one target grasp. For efficiency, the voxel grids
        are precomputed and cached in a compressed format.

        Args:
            root_dir: Path to the synthetic grasp dataset.
            rebuild_cache: Discard cached volumes.
        """
        self.root_dir = root_dir
        self.rebuild_cache = rebuild_cache

        self.build_cache()

        with open(os.path.join(self.cache_dir, 'grasps.json'), 'rb') as fp:
            self.grasps = json.load(fp)

    def __len__(self):
        return len(self.grasps)

    def __getitem__(self, idx):
        """
        Returns:
            The input TSDF, voxel index of the grasp position, and grasp score.
        """
        point = self.grasps[idx]
        tsdf = np.load(os.path.join(self.cache_dir, point['tsdf']))['tsdf']
        idx = np.asarray(point['idx'], dtype=np.int32)
        score = np.asarray([point['score']], dtype=np.float32)
        return tsdf, idx, score

    @property
    def cache_dir(self):
        return os.path.join(self.root_dir, '_cache')

    def build_cache(self):
        if os.path.exists(self.cache_dir):
            if self.rebuild_cache:
                logging.info('Moving existing cache to trash')
                send2trash(self.cache_dir)
            else:
                logging.info('Using existing cache')
                return

        logging.info('Building cache for scene')

        # Detect all scenes in the synthetic grasp dataset
        scene_directories = [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]

        # Cache reconstructions
        os.makedirs(self.cache_dir)

        grasps = []
        for dirname in tqdm(scene_directories):
            scene = data.load_scene(os.path.join(self.root_dir, dirname))
            _, voxel_grid = data.reconstruct_volume(scene)
            tsdf = utils.voxel_grid_to_array(voxel_grid, resolution=40)
            tsdf = np.expand_dims(tsdf, 0)

            fname = os.path.join(self.cache_dir, dirname) + '.npz'
            np.savez_compressed(fname, tsdf=tsdf)

            for pose, score in zip(scene['poses'], scene['scores']):
                i, j, k = voxel_grid.get_voxel(pose.translation).tolist()
                grasps.append({
                    'tsdf': dirname + '.npz',
                    'idx': [i, j, k],
                    'score': score,
                })

        with open(os.path.join(self.cache_dir, 'grasps.json'), 'wb') as fp:
            json.dump(grasps, fp)
