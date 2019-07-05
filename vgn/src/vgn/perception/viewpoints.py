from math import pi, cos, sin
import numpy as np

from vgn.utils.transform import Transform


def sample_hemisphere(n, length):
    """Generate random viewpoints on a half-sphere.

    Args:
        n: The number of viewpoints.
        length: The length of the workspace.
    """
    for _ in range(n):
        half_length = 0.5 * length

        phi = np.random.uniform(0., 2. * pi)
        theta = np.random.uniform(pi/6., 5.*pi/12.)
        r = 3 * half_length

        eye = np.array([r * sin(theta) * cos(phi) + half_length,
                        r * sin(theta) * sin(phi) + half_length,
                        r * cos(theta)])
        target = np.array([half_length, half_length, 0.])
        up = [0., 0., 1.]

        yield Transform.look_at(eye, target, up)
