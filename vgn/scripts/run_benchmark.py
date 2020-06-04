"""Clutter removal benchmark.

Each round, N objects are randomly placed in a tray. Then, the system is run until
(a) no objects remain, (b) VGN failed to find a grasp hypothesis, or (c) three
consecutive failed grasp attempts.
"""

import argparse
from pathlib2 import Path


from vgn.benchmark import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simulated clutter removal benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--object-set", type=str, default="adversarial")
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    args = parser.parse_args()

    main(args)
