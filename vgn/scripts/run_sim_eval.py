import argparse
from pathlib import Path
import time

import open3d
from mayavi import mlab
import numpy as np
import torch
from tqdm import tqdm

from vgn.grasp import Label
from vgn.hand import Hand
from vgn.grasp import from_voxel_coordinates
from vgn.grasp_detector import GraspDetector
from vgn.data_generation import reconstruct_scene
from vgn.simulation import GraspExperiment
from vgn.utils.io import load_dict
from vgn.utils.transform import Transform
from vgn.utils import vis


def main(args):
    config = load_dict(Path(args.config))
    urdf_root = Path(config["urdf_root"])
    hand_config = load_dict(Path(config["hand_config"]))
    object_set = config["object_set"]
    network_path = Path(config["model_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_trials = args.num_trials

    hand = Hand.from_dict(hand_config)
    size = 4 * hand.max_gripper_width

    sim = GraspExperiment(urdf_root, object_set, hand, size, args.sim_gui, args.rtf)
    detector = GraspDetector(device, network_path, show_qual=args.show_vis)

    outcomes = np.empty(num_trials, dtype=np.int)
    for i in tqdm(range(num_trials)):
        outcomes[i] = run_trial(sim, detector, args.sim_gui, args.show_vis)

    print_results(outcomes)


def run_trial(sim, detector, sim_gui, show_vis):
    sim.setup()
    sim.pause()
    tsdf, high_res_tsdf = reconstruct_scene(sim)
    point_cloud = high_res_tsdf.extract_point_cloud()

    grasps, qualities = detector.detect_grasps(tsdf.get_volume())

    if grasps.size == 0:
        return 0

    T = Transform.identity()
    for i, grasp in enumerate(grasps):
        grasps[i] = from_voxel_coordinates(grasp, T, tsdf.voxel_size)

    if show_vis:
        mlab.figure()
        vis.draw_detections(point_cloud, grasps, qualities)
        mlab.show()

    i = np.argmax(qualities)
    grasp = grasps[i]
    sim.resume()
    outcome, width = sim.test_grasp(grasp.pose)

    if outcome is not Label.SUCCESS and sim_gui:
        time.sleep(2.0)

    return outcome


def print_results(outcomes):
    num_trials = len(outcomes)
    num_no_detections = np.sum(outcomes == 0)
    num_collisions = np.sum(outcomes == Label.COLLISION)
    num_slipped = np.sum(outcomes == Label.SLIPPED)
    num_successes = np.sum(outcomes == Label.SUCCESS)

    print("No detections: {}/{}".format(num_no_detections, num_trials))
    print("Collisions:   {}/{}".format(num_collisions, num_trials))
    print("Slipped:   {}/{}".format(num_slipped, num_trials))
    print("Successes:   {}/{}".format(num_successes, num_trials))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate vgn in simulation")
    parser.add_argument(
        "--config", type=str, required=True, help="experiment configuration file",
    )
    parser.add_argument(
        "--num-trials", type=int, default=100, help="number of trials to run"
    )
    parser.add_argument("--sim-gui", action="store_true", help="disable headless mode")
    parser.add_argument(
        "--rtf", type=float, default=-1.0, help="real time factor of the simulation"
    )
    parser.add_argument("--show-vis", action="store_true")
    args = parser.parse_args()
    main(args)
