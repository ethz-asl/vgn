import enum

import numpy as np
import scipy.signal as signal

from vgn.utils.transform import Rotation, Transform


class Outcome(enum.Enum):
    """Possible outcomes of a grasp attempt."""
    SUCCESS = 0
    COLLISION = 1
    EMPTY = 2
    SLIPPED = 3


class Grasper(object):
    """Open-loop grasp execution.

    First, the TCP is positioned to a pre-grasp pose, from which the grasp pose
    is approached linearly. If the grasp pose is reached without any collisions,
    the gripper is closed and the object retrieved.
    """
    def __init__(self, robot):
        self.robot = robot
        self.T_grasp_pregrasp = Transform(Rotation.identity(), [0., 0., -0.05])

    def grasp(self, T_body_grasp):
        """Execute the given grasp and report the outcome."""
        threshold = 0.2
        T_body_pregrasp = T_body_grasp * self.T_grasp_pregrasp

        if not self.robot.set_tcp_pose(T_body_pregrasp):
            return Outcome.COLLISION

        self.robot.open_gripper()

        if not self.robot.move_tcp_xyz(T_body_grasp, check_collisions=True):
            return Outcome.COLLISION

        self.robot.close_gripper()
        if self.robot.get_gripper_opening_width() < threshold:
            return Outcome.EMPTY

        self.robot.move_tcp_xyz(T_body_pregrasp)

        if self.robot.get_gripper_opening_width() < threshold:
            return Outcome.SLIPPED

        return Outcome.SUCCESS


def sample_uniform(point_cloud, min_z_offset, max_z_offset):
    """Uniformly sample a grasp point from a point cloud.

    A random offset is applied to the point along the negative surface normal.

    Args:
        point_cloud: The point cloud from which the point is sampled
        min_offset: The minimum offset along the surface normal.
        max_offset: The maximum offset along the surface normal.
    """
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    selection = np.random.randint(len(points))
    point, normal = points[selection], normals[selection]
    z_offset = np.random.uniform(min_z_offset, max_z_offset)
    points = point - normal * z_offset

    return point, normal


def evaluate(sim, grasper, point, normal):
    """Evaluate the quality of the given grasp point.

    Args:
        sim: The simulation used for evaluating the grasp point.
        grasper: A Grasper object.
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
        sim.restore_state()
        outcome = grasper.grasp(Transform(orientation, point))
        scores.append(outcome == Outcome.SUCCESS)

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
