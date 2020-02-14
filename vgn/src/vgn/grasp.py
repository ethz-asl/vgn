import enum

from vgn.utils.transform import Transform


class Label(enum.IntEnum):
    """Outcome of a grasping attempt."""

    COLLISION = 1
    EMPTY = 2
    SLIPPED = 3
    SUCCESS = 4
    ROBUST = 5


def to_voxel_coordinates(grasp, T_base_task, voxel_size):
    pose = T_base_task.inverse() * grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, T_base_task, voxel_size):
    pose = T_base_task * grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width)


class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    TODO(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width
