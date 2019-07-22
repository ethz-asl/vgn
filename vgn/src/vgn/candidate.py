import numpy as np

from vgn import grasp
from vgn.utils.transform import Rotation, Transform


def evaluate(s, g, point, normal):
    """Evaluate the quality of the given grasp point.

    We generate several grasp poses by rotating the hand around the surface
    normal and, if succesfull, report the mean orientation.

    Args:
        s: The simulation used for evaluating the grasp point.
        g: A Grasper object.
        point: The grasp point to be evaluated.
        normal: The surface normal at the grasp point.
    """
    # Define a frame where the z-axis corresponds to -surface normal
    z = -normal
    x = np.array([1., 0., 0.])
    if np.isclose(np.abs(np.dot(x, z)), 1., 1e-4):
        x = np.array([0., 1., 0.])
    y = np.cross(z, x)
    x = np.cross(y, z)
    R = Rotation.from_dcm(np.vstack((x, y, z)).T)

    good_yaws = []
    for yaw in np.linspace(-0.5 * np.pi, 0.5 * np.pi, 16):
        orientation = R * Rotation.from_euler('z', yaw)
        s.restore_state()
        outcome = g.grasp(Transform(orientation, point))
        if outcome == grasp.Outcome.SUCCESS:
            good_yaws.append(yaw)

    if good_yaws:
        orientation = R * Rotation.from_euler('z', np.mean(good_yaws))
        return 1., _ensure_consistent_orientation(orientation)
    else:
        return 0., _ensure_consistent_orientation(R)


def _ensure_consistent_orientation(orientation):
    """Due to the symmetric geometry of a parallel-jaw gripper, make sure
    the y-axis always points upwards.
    """
    y = orientation.as_dcm()[:, 1]
    if np.dot(y, np.array([0., 0., 1.])) < 0.:
        orientation *= Rotation.from_euler('z', np.pi)
    return orientation
