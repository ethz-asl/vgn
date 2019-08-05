import numpy as np
import scipy.signal as signal

from vgn import grasp
from vgn.utils.transform import Rotation, Transform


def evaluate(s, g, point, normal):
    """Evaluate the quality of the given grasp point.

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

    yaws = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 16)
    scores = []

    for yaw in yaws:
        orientation = R * Rotation.from_euler('z', yaw)
        s.restore_state()
        outcome = g.grasp(Transform(orientation, point))
        scores.append(outcome == grasp.Outcome.SUCCESS)

    if np.sum(scores):
        # Detect the peak over yaw orientations
        peaks, properties = signal.find_peaks(x=np.r_[0, scores, 0],
                                              height=1,
                                              width=1)
        idx_of_widest_peak = peaks[np.argmax(properties['widths'])] - 1
        yaw = yaws[idx_of_widest_peak]

        ori = _ensure_consistent_orientation(R * Rotation.from_euler('z', yaw))
        return Transform(ori, point), 1.
    else:
        ori = _ensure_consistent_orientation(R)
        return Transform(ori, point), 0.


def _ensure_consistent_orientation(orientation):
    """Due to the symmetric geometry of a parallel-jaw gripper, make sure
    the y-axis always points upwards.
    """
    y = orientation.as_dcm()[:, 1]
    if np.dot(y, np.array([0., 0., 1.])) < 0.:
        orientation *= Rotation.from_euler('z', np.pi)
    return orientation
