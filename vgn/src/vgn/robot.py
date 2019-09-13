class RobotArm(object):
    """Abstract base class for a robot arm."""
    def get_tool_pose(self):
        """Return the pose of the tool in base frame."""
        raise NotImplementedError

    def set_tool_pose(self, target_pose):
        """Move the tool to the given pose.

        Args:
            target_pose: The target pose of the tool in base frame, T_base_tool.

        Returns:
            False if the target pose is in collision.
        """
        raise NotImplementedError

    def move_tool_xyz(self, target_pose, eef_step, check_collisions):
        """Linearly move the EE in cartesian space to the given pose.

        Args:
            target_pose: The target pose of the tool in base frame.
            eef_step: Path interpolation resolution [m].
            check_collisions: Abort the movement if a collision is detected.

        Returns:
            False if a collision was detected.
        """
        raise NotImplementedError

    def get_gripper_opening_width(self):
        """Return the gripper opening width, normalized to [0, 1]."""
        raise NotImplementedError

    def open_gripper(self):
        """Open the gripper."""
        raise NotImplementedError

    def close_gripper(self):
        """Close the gripper."""
        raise NotImplementedError
