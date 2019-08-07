import json
import logging
import os

import numpy as np
import torch.utils.data
from send2trash import send2trash
from tqdm import tqdm

from vgn import utils
from vgn.perception import integration
from vgn.utils import camera
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

        # Build and cache dataset
        os.makedirs(self.cache_dir)

        grasps = []
        for dirname in tqdm(scene_directories):
            data = load_scene_data(os.path.join(self.root_dir, dirname))

            # Build TSDF
            size, resolution = 0.2, 60
            intrinsic = data['intrinsic']
            volume = integration.TSDFVolume(size=size, resolution=resolution)
            for extrinsic, img in zip(data['extrinsics'], data['images']):
                volume.integrate(img, intrinsic, extrinsic)
            voxel_grid = volume.get_voxel_grid()
            shape = (1, resolution, resolution, resolution)
            tsdf = np.zeros(shape, dtype=np.float32)
            for voxel in voxel_grid.voxels:
                i, j, k = voxel.grid_index
                tsdf[0, i, j, k] = voxel.color[0]

            # Write cached TSDF to disk and add grasps to dataset
            cached_tsdf = os.path.join(self.cache_dir, dirname) + '.npz'
            np.savez_compressed(cached_tsdf, tsdf=tsdf)

            for pose, score in zip(data['poses'], data['scores']):
                i, j, k = voxel_grid.get_voxel(pose.translation).tolist()
                grasps.append({
                    'tsdf': dirname + '.npz',
                    'idx': [i, j, k],
                    'score': score,
                })

        with open(os.path.join(self.cache_dir, 'grasps.json'), 'wb') as fp:
            json.dump(grasps, fp)


def load_scene_data(dirname):
    intrinsic = _load_intrinsic(dirname)
    extrinsics, images = _load_images(dirname)
    poses, scores = _load_grasps(dirname)
    sample = {
        'intrinsic': intrinsic,
        'extrinsics': extrinsics,
        'images': images,
        'poses': poses,
        'scores': scores
    }
    return sample


def _load_intrinsic(dirname):
    fname = os.path.join(dirname, 'intrinsic.json')
    return camera.PinholeCameraIntrinsic.from_json(fname)


def _load_images(dirname):
    with open(os.path.join(dirname, 'viewpoints.json'), 'rb') as fp:
        viewpoints = json.load(fp)

    imgs, extrinsics = [], []
    for viewpoint in viewpoints:
        img = utils.load_image(os.path.join(dirname, viewpoint['image_name']))
        imgs.append(img)
        extrinsics.append(Transform.from_dict(viewpoint['extrinsic']))

    return extrinsics, imgs


def _load_grasps(dirname):
    with open(os.path.join(dirname, 'grasps.json'), 'rb') as fp:
        grasps = json.load(fp)

    poses, scores = [], np.empty((len(grasps), ))
    for i, grasp in enumerate(grasps):
        poses.append(Transform.from_dict(grasp['pose']))
        scores[i] = grasp['score']
    return poses, scores
