import argparse
from pathlib import Path
from mpi4py import MPI
import numpy as np
import open3d as o3d
from tqdm import tqdm

from robot_helpers.io import load_yaml
from robot_helpers.spatial import Rotation, Transform
from vgn.data import write
from vgn.grasp import ParallelJawGrasp
from vgn.perception import create_tsdf
from vgn.simulation import GraspSim, get_metric, generate_pile
from vgn.utils import find_urdfs, view_on_sphere


def main():
    worker_count, rank = setup_mpi()

    parser = create_parser()
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed + 100 * rank)
    cfg = load_yaml(args.cfg)
    object_urdfs = find_urdfs(Path(cfg["object_urdfs"]))

    origin = Transform.t_[0.0, 0.0, 0.05]
    size = 0.3
    center = origin * Transform.t_[0.5 * size, 0.5 * size, 0.0]

    sim = GraspSim(cfg["sim"], rng)
    score_fn = get_metric(cfg["metric"])(sim)

    grasp_count = args.count // worker_count
    scene_count = grasp_count // cfg["scene_grasp_count"]

    for _ in tqdm(range(scene_count), disable=rank != 0):
        object_count = rng.poisson(cfg["object_count_lambda"]) + 1
        urdfs = rng.choice(object_urdfs, object_count)
        scales = rng.uniform(cfg["scaling"]["low"], cfg["scaling"]["high"], len(urdfs))
        sim.clear()
        sim.robot.reset(Transform.t_[np.full(3, 10)], sim.robot.max_width)
        generate_pile(sim, origin, size, urdfs, scales)
        state_id = sim.save_state()

        view_count = rng.randint(cfg["max_view_count"]) + 1
        views = sample_views(center, size, view_count, rng)
        imgs = render_imgs(views, sim.camera)
        pc = create_pc(center, size, imgs, sim.camera.intrinsic, views)
        if pc.is_empty():
            continue

        grasps = sample_grasps(cfg["scene_grasp_count"], pc, sim.robot, rng)
        scores = [evaluate_grasp_point(g, sim, state_id, score_fn) for g in grasps]

        write(views, imgs, grasps, scores, args.root)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def setup_mpi():
    worker_count = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return worker_count, rank


def sample_views(center, size, view_count, rng):
    views = []
    for _ in range(view_count):
        r = rng.uniform(1.6, 2.4) * size
        theta = rng.uniform(np.pi / 4.0)
        phi = rng.uniform(2.0 * np.pi)
        views.append(view_on_sphere(center, r, theta, phi))
    return views


def render_imgs(views, camera):
    return [camera.get_image(view)[1] for view in views]


def create_pc(center, size, imgs, intrinsic, views):
    tsdf = create_tsdf(size, 120, imgs, intrinsic, views)
    pc = tsdf.get_scene_cloud()
    lower = np.r_[0.02, 0.02, center.translation[2] + 0.01]
    upper = np.r_[size - 0.02, size - 0.02, size]
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
    return pc.crop(bounding_box)


def sample_grasps(count, pc, robot, rng, eps=0.1):
    points, normals = np.asarray(pc.points), np.asarray(pc.normals)
    grasps = []
    for _ in range(count):
        ok = False
        while not ok:  # This could result in an infinite loop, though unlikely.
            i = rng.randint(len(points))
            point, normal = points[i], normals[i]
            ok = normal[2] > -0.1  # Ensure that the normal is pointing upwards
        depth = rng.uniform(-eps * robot.max_depth, (1 + eps) * robot.max_depth)
        pose = construct_grasp_frame(point, normal) * Transform.t_[0, 0, -depth]
        grasps.append(ParallelJawGrasp(pose, robot.max_width))
    return grasps


def construct_grasp_frame(point, normal):
    z = -normal
    y = np.r_[z[1] - z[2], -z[0] + z[2], z[0] - z[1]]
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return Transform(Rotation.from_matrix(np.vstack((x, y, z))), point)


def evaluate_grasp_point(grasp, sim, state_id, quality_fn, rot_count=6):
    # Changes the rotation of the grasp in place
    angles = np.linspace(0.0, np.pi, rot_count)
    R = grasp.pose.rotation
    for angle in angles:
        grasp.pose.rotation = R * Rotation.from_rotvec([0, 0, angle])
        sim.restore_state(state_id)
        quality, _ = quality_fn(grasp)
        if quality:
            return 1.0
    return 0.0


if __name__ == "__main__":
    main()
