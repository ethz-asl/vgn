import pathlib2

import numpy as np
import open3d as o3d

from vgn.simulation import GraspSimulation
from vgn.utils import io
from vgn.utils.transform import Transform, Rotation

SHOW_VIS = False


def test_sim():
    object_set = "cuboid"
    config = io.load_dict(pathlib2.Path("config/default.yaml"))

    sim = GraspSimulation(object_set, config, gui=True)
    sim.reset()
    sim.world.pause()

    tsdf, pc = sim.acquire_tsdf()
    if SHOW_VIS:
        o3d.visualization.draw_geometries([tsdf.extract_point_cloud()])

    sim.world.unpause()
    R_world_grasp = Rotation.from_euler("xyz", [np.pi, 0.0, 0.5 * np.pi])
    t_world_grasp = np.r_[0.5 * sim.size, 0.5 * sim.size, 0.06]
    res = sim.execute_grasp(Transform(R_world_grasp, t_world_grasp))


if __name__ == "__main__":
    test_sim()
