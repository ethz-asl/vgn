import argparse
from pathlib import Path

from mpi4py import MPI
import numpy as np
import open3d as o3d
from tqdm import tqdm

from robot_helpers.spatial import Rotation, Transform
from vgn.database import GraspDatabase
from vgn.grasp import UniformPointCloudSampler
from vgn.perception import UniformTSDFVolume
from vgn.simulation import GraspSim
from vgn.utils import load_cfg, find_urdfs, camera_on_sphere


def main():
    worker_count, rank = setup_mpi()

    parser = create_parser()
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    rng = np.random.RandomState(args.seed + 100 * rank)
    urdfs = find_urdfs(Path(cfg["urdf_root"]))
    origin = Transform.t([0.0, 0.0, 0.05])

    sim = GraspSim(cfg["sim"], rng)
    grasp_sampler = UniformPointCloudSampler(sim.gripper, rng)
    database = GraspDatabase(args.root)

    grasp_count = args.count // worker_count
    scene_count = grasp_count // cfg["scene_grasp_count"]

    for _ in tqdm(range(scene_count), disable=rank != 0):
        # Generate a new scene
        sim.scene.clear()
        object_count = np.random.poisson(cfg["object_count_lambda"]) + 1
        urdfs = rng.choice(urdfs, object_count)
        sim.gripper.reset(Transform.t(np.full(3, 100)), sim.gripper.max_width)
        sim.scene.generate(origin, urdfs, scaling=1.6)
        sim.save_state()

        # Sample camera views
        view_count = np.random.randint(cfg["max_view_count"]) + 1
        views = sample_views(sim, view_count, rng)

        # Render images
        imgs = render_imgs(sim, views)

        # Reconstruct point cloud
        pc = create_pc(sim, imgs, views)

        # Sample grasps
        grasps = grasp_sampler(pc, cfg["scene_grasp_count"])

        # Evaluate grasps
        qualities = [evaluate_grasp_point(sim, grasp) for grasp in grasps]

        # Write
        database.write(views, imgs, grasps, qualities)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default="data/grasps")
    parser.add_argument("--cfg", type=Path, default="cfg/grasp_database.yaml")
    parser.add_argument("--count", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def setup_mpi():
    worker_count = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return worker_count, rank


def sample_views(sim, view_count, rng):
    views = []
    for _ in range(view_count):
        r = rng.uniform(1.6, 2.4) * sim.scene.size
        theta = rng.uniform(0.0, np.pi / 4.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        views.append(camera_on_sphere(sim.scene.center, r, theta, phi))
    return views


def render_imgs(sim, views):
    return [sim.camera.get_image(view)[1] for view in views]


def create_pc(sim, imgs, views):
    tsdf = UniformTSDFVolume(sim.scene.size, 120)
    for img, view in zip(imgs, views):
        tsdf.integrate(img, sim.camera.intrinsic, view.inv())
    pc = tsdf.get_scene_cloud()

    lower = np.r_[0.02, 0.02, sim.scene.center.translation[2] + 0.01]
    upper = np.r_[sim.scene.size - 0.02, sim.scene.size - 0.02, sim.scene.size]
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
    pc = pc.crop(bounding_box)
    # o3d.visualization.draw_geometries([pc])
    return pc


def evaluate_grasp_point(sim, grasp, rot_count=6):
    # If a stable configuration is found, changes the rotation of the grasp in place
    angles = np.linspace(0.0, np.pi, rot_count)
    R = grasp.pose.rotation
    for angle in angles:
        grasp.pose.rotation = R * Rotation.from_rotvec([0, 0, angle])
        sim.restore_state()
        quality, _ = sim.quality(grasp)
        if quality:
            return 1.0
    return 0.0


if __name__ == "__main__":
    main()
