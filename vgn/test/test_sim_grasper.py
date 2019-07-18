"""Small script to test the simulated grasp primitive."""

import numpy as np

from vgn import grasp, simulation
from vgn.utils.transform import Rotation, Transform


def test_sim_grasper():
    s = simulation.Simulation(gui=True, real_time=True)
    g = grasp.Grasper(robot=s)

    s.reset()
    s.spawn_plane()
    s.spawn_debug_cuboid()
    s.spawn_robot()
    s.save_state()

    rotation = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    position = np.array([0.1, 0.1, 0.01])
    T_world_tcp = Transform(Rotation.from_dcm(rotation), position)

    s.restore_state()
    result = g.grasp(T_world_tcp)
    s.sleep(1.0)

    assert result == grasp.Outcome.SUCCESS


if __name__ == "__main__":
    test_sim_grasper()
