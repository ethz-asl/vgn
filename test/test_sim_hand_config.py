import numpy as np

from vgn.simulation import Gripper
from vgn.utils import io, btsim
from vgn.utils.transform import Transform, Rotation


def main():
    world = btsim.BtWorld(gui=True)
    world.reset()
    gripper = Gripper(world)

    # check that the palm is located at the origin
    pose = Transform(Rotation.from_euler("y", np.pi), [0, 0, 0])
    gripper.reset(pose)
    gripper.body.joints["panda_finger_joint1"].set_position(0.0, kinematics=True)
    gripper.body.joints["panda_finger_joint2"].set_position(0.0, kinematics=True)

    # check that the finger depth is correct
    pose = Transform(Rotation.from_euler("y", np.pi), [0, 0, gripper.finger_depth])
    gripper.set_tcp(pose)
    pass


if __name__ == "__main__":
    main()

