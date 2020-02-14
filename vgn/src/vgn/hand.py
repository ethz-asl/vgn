from vgn.utils.transform import Transform


class Hand(object):
    def __init__(
        self, max_gripper_width, finger_depth, T_tool0_tcp, urdf_path,
    ):
        self.max_gripper_width = max_gripper_width
        self.finger_depth = finger_depth

        self.T_tool0_tcp = T_tool0_tcp
        self.urdf_path = urdf_path

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["max_gripper_width"],
            data["finger_depth"],
            Transform.from_dict(data["T_tool0_tcp"]),
            data["urdf_path"],
        )
