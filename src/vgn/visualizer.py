import numpy as np
from mayavi import mlab

from vgn.utils import grid_to_map_cloud

cm = lambda s: tuple([float(1 - s), float(s), float(0)])


def clear():
    mlab.clf()


def scene_cloud(voxel_size, points):
    mlab.points3d(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        scale_factor=0.8 * voxel_size,
        scale_mode="none",
    )


def map_cloud(voxel_size, points, distances):
    mlab.points3d(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        distances,
        mode="cube",
        scale_factor=voxel_size,
        scale_mode="none",
    )


def quality(voxel_size, grid, vmin=0.9):
    points, values = grid_to_map_cloud(voxel_size, grid, vmin)
    mlab.points3d(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        values.squeeze(),
        mode="cube",
        scale_factor=voxel_size,
        scale_mode="none",
        vmin=vmin,
        vmax=1.0,
        opacity=0.2,
    )


def grasp(grasp, score, radius=0.002):
    pose, w, d = grasp.pose, 0.08, 0.05
    color = cm(score)

    points = [[0, -w / 2, d], [0, -w / 2, 0], [0, w / 2, 0], [0, w / 2, d]]
    points = np.vstack([pose.apply(p) for p in points])
    mlab.plot3d(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=color,
        tube_radius=radius,
    )

    points = [[0, 0, 0], [0, 0, -d]]
    points = np.vstack([pose.apply(p) for p in points])
    mlab.plot3d(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=color,
        tube_radius=radius,
    )


def grasps(grasps, scores, max_grasps=None):
    if max_grasps and max_grasps < len(grasps):
        i = np.random.randint(len(grasps), size=max_grasps)
        grasps, scores = grasps[i], scores[i]
    for grasp_config, quality in zip(grasps, scores):
        grasp(grasp_config, quality)


def show():
    mlab.show()
