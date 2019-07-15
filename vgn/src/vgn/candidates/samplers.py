from __future__ import division

import numpy as np
import open3d

from vgn.utils.transform import Rotation, Transform


def uniform(point_cloud, n):
    """Sample grasp candidates uniformly from a point cloud."""
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    selection = np.random.choice(len(points), size=n)

    candidates = []
    for idx in selection:
        # Add random offset to the finger tip point along the surface normal
        z_offset = np.random.uniform(0., 0.04)
        position = points[idx] - z_offset * normals[idx]

        # Sample an orientation of the grasp frame
        orientation = sample_orientation(normals[idx])

        candidates.append(Transform(orientation, position))

    return candidates


def sample_orientation(normal):
    z = -normal

    x = np.array([1., 0., 0.])
    if np.isclose(np.abs(np.dot(x, z)), 1., 1e-4):
        x = np.array([0., 1., 0.])

    y = np.cross(z, x)
    x = np.cross(y, z)
    r = Rotation.from_dcm(np.vstack((x, y, z)).T)

    # Randomly perturbe around yaw, pitch and roll
    x_rot = 0.  # truncated_normal(stddev=np.pi/18.)
    y_rot = 0.  # truncated_normal(stddev=np.pi/18.)
    z_rot = np.random.uniform(-np.pi / 2., np.pi / 2.)
    r = r * Rotation.from_euler("xyz", [x_rot, y_rot, z_rot])

    # TODO Flip  if y is pointing downards, to be consistent

    return r


def _truncated_normal(mean=0., stddev=1.):
    """Values higher than 2 standard deviations are dropped and re-picked."""
    while True:
        sample = np.random.normal(mean, stddev)
        if np.abs(sample) < 3 * stddev:
            return sample
