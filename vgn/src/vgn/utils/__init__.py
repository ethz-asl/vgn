import os

import cv2
import numpy as np
import torch


def save_image(fname, img):
    """Save image as a PNG file."""
    img = (1000.0 * img).astype(np.uint16)
    cv2.imwrite(fname, img)


def load_image(fname):
    """Load image from a PNG file."""
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) * 0.001
    return img


def voxel_grid_to_array(voxel_grid, resolution):
    """Convert an Open3D voxel grid to a contiguous 3D numpy array."""
    v = np.zeros(shape=(resolution, resolution, resolution), dtype=np.float32)
    for voxel in voxel_grid.voxels:
        i, j, k = voxel.grid_index
        v[i, j, k] = voxel.color[0]
    return v
