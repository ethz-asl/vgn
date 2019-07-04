"""Small script to test the simulated grasp primitive."""

import numpy as np

from vgn.grasper import Grasper
from vgn.simulation import Simulation
from vgn.utils.transform import Rotation, Transform


def main():
    s = Simulation(gui=True)
    g = Grasper(robot=s)

    s.reset()
    s.spawn_plane()
    s.spawn_cuboid()
    s.spawn_robot()
    s.save_state()

    rotation = np.array([[1., 0., 0.],
                         [0., -1., 0.],
                         [0., 0., -1.]])
    position = np.array([0.1, 0.1, 0.01])
    T_world_tcp = Transform(Rotation.from_dcm(rotation), position)

    while True:
        s.restore_state()
        s.sleep(1.0)
        result = g.grasp(T_world_tcp)
        print(result)
        s.sleep(1.0)


if __name__ == '__main__':
    main()
