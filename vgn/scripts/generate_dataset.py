from __future__ import print_function, division

import argparse
from pathlib2 import Path
import uuid

from mpi4py import MPI
import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm

from vgn import Label, to_voxel_coordinates
from vgn.grasp import Grasp
from vgn.simulation import GraspSimulation
from vgn.utils.transform import Rotation, Transform

MAX_OBJECT_COUNT = 4
VIEWPOINT_COUNT = 3


def main(args):
    workers, rank = setup_mpi()
    create_dataset_dir(args.dataset_dir, rank)

    sim = GraspSimulation(args.object_set, "config/default.yaml", args.sim_gui)
    finger_depth = sim.config["finger_depth"]

    for _ in tqdm(range(args.grasps // workers), disable=rank is not 0):
        # generate heap
        object_count = np.random.randint(1, MAX_OBJECT_COUNT + 1)
        sim.reset(object_count)
        sim.save_state()

        # reconstruct and crop surface from point cloud
        tsdf, pc = sim.acquire_tsdf(num_viewpoints=VIEWPOINT_COUNT)
        l, u = 1.2 * finger_depth, sim.size - 1.2 * finger_depth
        z = sim.world.bodies[0].get_pose().translation[2] + 0.005
        pc = pc.crop(np.r_[l, l, z], np.r_[u, u, sim.size])

        # sample grasp point
        point, normal = sample_grasp_point(pc, finger_depth)

        # test grasp point
        grasp, label = evaluate_grasp_point(sim, point, normal)

        # store sample
        store_sample(args.dataset_dir, tsdf, grasp, label)


def setup_mpi():
    workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return workers, rank


def create_dataset_dir(dataset_dir, rank):
    if rank != 0:
        return
    dataset_dir.mkdir(exist_ok=True)
    csv_path = dataset_dir / "grasps.csv"
    if not csv_path.exists():
        with open(str(csv_path), "w") as f:
            f.write("tsdf,i,j,k,qx,qy,qz,qw,width,label\n")


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    idx = np.random.randint(len(points))
    point, normal = points[idx], normals[idx]
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_dcm(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        outcome, width = sim.execute_grasp(Transform(ori, pos), remove=False)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


def store_sample(dataset_dir, tsdf, grasp, label):
    # convert to voxel coordinates
    tsdf_vol = tsdf.get_volume()
    grasp = to_voxel_coordinates(grasp, tsdf.voxel_size)
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    i, j, k = np.round(grasp.pose.translation).astype(np.int)
    width = grasp.width

    # store TSDF volume in compressed .npz format
    path = dataset_dir / (uuid.uuid4().hex + ".npz")
    np.savez_compressed(str(path), tsdf=tsdf_vol)
    # add a row to the table (TODO concurrent writes could be an issue)
    values = [str(v) for v in [path.name, i, j, k, qx, qy, qz, qw, width, label]]
    with open(str(dataset_dir / "grasps.csv"), "a") as f:
        f.write(",".join(values))
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--object-set", type=str, required=True)
    parser.add_argument("--grasps", type=int, default=1000)
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    main(args)
