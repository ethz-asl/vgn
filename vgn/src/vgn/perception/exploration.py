from math import cos, pi, sin

import numpy as np

import vgn.config as cfg
from vgn.utils.transform import Transform


def sample_hemisphere(n):
    """Generate random viewpoints on a task space cenetered half-sphere.

    Args:
        n: The number of viewpoints.

    Returns:
        List of extrinsics, i.e. T_camera_task.
    """
    extrinsics = []
    for _ in range(n):
        half_size = 0.5 * cfg.size

        phi = np.random.uniform(0.0, 2.0 * pi)
        theta = np.random.uniform(pi / 6.0, 5.0 * pi / 12.0)
        r = 3 * half_size

        eye = np.array(
            [
                r * sin(theta) * cos(phi) + half_size,
                r * sin(theta) * sin(phi) + half_size,
                r * cos(theta),
            ]
        )
        target = np.array([half_size, half_size, 0.0])
        up = np.array([0.0, 0.0, 1.0])

        extrinsics.append(Transform.look_at(eye, target, up))

    return extrinsics
