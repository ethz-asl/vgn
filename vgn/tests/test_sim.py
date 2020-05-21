import numpy as np

from vgn.simulation import GraspSimulation
from vgn.utils.transform import Transform, Rotation


def test_grasp():
    sim = GraspSimulation("blocks", "config/default.yaml", gui=True)
    sim.reset(0)

    urdf = "data/urdfs/blocks/cuboid/cuboid.urdf"
    pose = Transform(Rotation.identity(), [0.5 * sim.size, 0.5 * sim.size, 0.02])
    sim.world.load_urdf(urdf, pose)

    R_world_grasp = Rotation.from_euler("xyz", [np.pi, 0.0, 0.5 * np.pi])
    t_world_grasp = np.r_[0.5 * sim.size, 0.5 * sim.size, 0.072]
    T_world_grasp = Transform(R_world_grasp, t_world_grasp)
    res = sim.execute_grasp(T_world_grasp)


if __name__ == "__main__":
    test_grasp()
