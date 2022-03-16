import numpy as np

from robot_helpers.spatial import Transform, Rotation
from vgn.simulation import GraspSim, load_urdf


def main():
    cfg = {"gui": True, "lateral_friction": 1.0}
    sim = GraspSim(cfg, np.random)

    ori = Rotation.from_euler("y", np.pi)
    pos = [0.0, 0.0, 0.0]
    sim.robot.reset(Transform(ori, pos), 0.08)

    load_urdf("plane/model.urdf", Transform.t_[0.0, 0.0, -0.05])

    [sim.step() for _ in range(120)]

    while True:
        sim.step()


if __name__ == "__main__":
    main()
