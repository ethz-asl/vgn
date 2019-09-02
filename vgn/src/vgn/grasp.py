from __future__ import print_function

import enum

import numpy as np
import scipy.signal as signal

import vgn.config as cfg
from vgn.utils.transform import Rotation, Transform


class Outcome(enum.Enum):
    """Possible outcomes of a grasp attempt."""
    SUCCESS = 0
    COLLISION = 1
    EMPTY = 2
    SLIPPED = 3


def sample_uniform(point_cloud):
    """Uniformly sample a grasp point from a point cloud.

    A random offset is applied to the point along the negative surface normal.

    Args:
        point_cloud: The point cloud from which the point is sampled
    """
    gripper_depth = 0.5 * cfg.max_width

    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    selection = np.random.randint(len(points))
    point, normal = points[selection], normals[selection]
    z_offset = np.random.uniform(-0.2 * gripper_depth, 1.2 * gripper_depth)
    point = point - normal * (z_offset - gripper_depth)

    return point, normal


def execute(robot, T_base_grasp):
    """Open-loop grasp execution.

    First, the TCP is positioned to a pre-grasp pose, from which the grasp pose
    is approached linearly. If the grasp pose is reached without any collisions,
    the gripper is closed and the object retrieved.

    Args:
        robot: Reference to the manipulator which will execute the grasp.
        T_base_grasp: The pose of the grasp w.r.t. manipulator base frame.
    """
    grasp_detection_threshold = 0.2
    T_grasp_pregrasp = Transform(Rotation.identity(), [0., 0., -0.05])

    T_base_pregrasp = T_base_grasp * T_grasp_pregrasp

    robot.set_tcp_pose(T_base_pregrasp, override_dynamics=True)
    robot.open_gripper()

    if not robot.move_tcp_xyz(T_base_grasp, check_collisions=True):
        return Outcome.COLLISION

    robot.close_gripper()

    if robot.get_gripper_opening_width() < grasp_detection_threshold:
        return Outcome.EMPTY

    robot.move_tcp_xyz(T_base_pregrasp)

    if robot.get_gripper_opening_width() < grasp_detection_threshold:
        return Outcome.SLIPPED

    return Outcome.SUCCESS


def evaluate(sim, point, normal, n_rotations=9):
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

    yaws = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_rotations)
    scores = []

    for yaw in yaws:
        orientation = R * Rotation.from_euler('z', yaw)
        sim.restore_state()
        outcome = execute(sim.hand, Transform(orientation, point))
        scores.append(outcome == Outcome.SUCCESS)

    if np.sum(scores):
        # Detect the peak over yaw orientations
        peaks, properties = signal.find_peaks(x=np.r_[0, scores, 0],
                                              height=1,
                                              width=1)
        idx_of_widest_peak = peaks[np.argmax(properties['widths'])] - 1
        yaw = yaws[idx_of_widest_peak]
        ori, score = R * Rotation.from_euler('z', yaw), 1.0
    else:
        ori, score = R, 0.0

    ori = _ensure_consistent_orientation(ori)
    return Transform(ori, point), score


def _ensure_consistent_orientation(orientation):
    """Due to the symmetric geometry of a parallel-jaw gripper, make sure
    the y-axis always points upwards.
    """
    y = orientation.as_dcm()[:, 1]
    if np.dot(y, np.array([0., 0., 1.])) < 0.:
        orientation *= Rotation.from_euler('z', np.pi)
    return orientation
