"""Small script to test the simulated grasp primitive."""

import numpy as np

from vgn.perception import integration
from vgn.grasper import Grasper
from vgn.simulation import Simulation
from vgn.utils.transform import Rotation, Transform
from vgn.candidates.samplers import sample_orientation


def main():
    s = Simulation(gui=True)
    g = Grasper(robot=s)

    s.reset()
    s.spawn_plane()
    s.spawn_debug_cuboid()
    s.spawn_robot()
    s.save_state()

    T_eye_world = Transform.look_at(eye=[0.1, 0.1, 0.5],
                                    center=[0.1, 0.1, 0.],
                                    up=[1., 0., 0.])

    volume = integration.TSDFVolume(length=0.3, resolution=100)
    rgb, depth = s.camera.get_rgb_depth(T_eye_world)
    volume.integrate(rgb, depth, s.camera.intrinsic, T_eye_world)

    point_cloud = volume.extract_point_cloud()

    idx = 150
    point = np.asarray(point_cloud.points)[idx]
    normal = np.asarray(point_cloud.normals)[idx]

    while True:
        orientation = sample_orientation(normal)
        T_world_tcp = Transform(orientation, point)

        s.restore_state()
        s.set_tcp_pose(T_world_tcp)
        s.sleep(1.0)


if __name__ == "__main__":
    main()
