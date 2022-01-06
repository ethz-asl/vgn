import numpy as np

from robot_helpers.spatial import Transform, Rotation
from vgn.simulation import GraspSim


def main():
    cfg = {"gui": True, "sleep": True, "scene": "pile"}
    sim = GraspSim(cfg, np.random)

    ori = Rotation.from_euler("y", np.pi)
    pos = [0.0, 0.0, 0.0]
    sim.gripper.reset(Transform(ori, pos), 0.08)

    [sim.step() for _ in range(120)]

    while True:
        sim.step()


if __name__ == "__main__":
    main()
