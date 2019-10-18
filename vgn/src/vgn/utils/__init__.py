import json
import os

import cv2
import numpy as np
import torch


def save_dict(path, data):
    """Serialize dict object as JSON file."""
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def load_dict(path):
    """Load dict object from JSON file."""
    with path.open("r") as f:
        data = json.load(f)
    return data


def save_image(path, img):
    """Save image as a PNG file."""
    img = (1000.0 * img).astype(np.uint16)
    cv2.imwrite(str(path), img)


def load_image(path):
    """Load image from a PNG file."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) * 0.001
    return img


def voxel_grid_to_array(voxel_grid, resolution):
    """Convert an Open3D voxel grid to a contiguous 3D numpy array."""
    v = np.zeros(shape=(resolution, resolution, resolution), dtype=np.float32)
    for voxel in voxel_grid.voxels:
        i, j, k = voxel.grid_index
        v[i, j, k] = voxel.color[0]
    return v
