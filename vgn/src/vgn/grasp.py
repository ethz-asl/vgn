import enum

from vgn.utils.transform import Transform


class Label(enum.IntEnum):
    """Outcome of a grasping attempt."""

    COLLISION = 1
    EMPTY = 2
    SLIPPED = 3
    SUCCESS = 4
    ROBUST = 5


class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    TODO(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose):
        self.pose = pose

    @classmethod
    def from_dict(cls, data):
        pose = Transform.from_dict(data["pose"])
        return cls(pose)

    def to_dict(self):
        data = {"pose": self.pose.to_dict()}
        return data
