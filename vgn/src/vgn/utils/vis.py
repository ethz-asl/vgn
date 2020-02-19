import numpy as np
from mayavi import mlab

from vgn.grasp import Grasp
from vgn.utils.transform import Rotation, Transform


def draw_sample(tsdf, qual, rot, width, mask):
    tsdf = tsdf.squeeze()
    qual = qual.squeeze()
    rot = rot.squeeze()
    width = width.squeeze()
    mask = mask.squeeze()

    draw_volume(tsdf.squeeze())

    grasps, qualities = [], []
    for (i, j, k) in np.argwhere(mask == 1.0):
        t = Transform(Rotation.from_quat(rot[:, i, j, k]), np.r_[i, j, k])
        q = qual[i, j, k]
        w = width[i, j, k]
        grasps.append(Grasp(t, w))
        qualities.append(q)
    draw_grasps(grasps, qualities, 6)


def draw_detections(point_cloud, grasps, qualities):
    draw_point_cloud(point_cloud)
    draw_grasps(grasps, qualities, 0.05)


def draw_volume(vol, tol=0.001):
    (x, y, z), scalars = np.where(vol > tol), vol[vol > tol]

    mlab.points3d(
        x,
        y,
        z,
        scalars,
        vmin=0.0,
        vmax=1.0,
        mode="cube",
        scale_mode="none",
        scale_factor=1,
        opacity=0.05,
    )

    # draw a slice through the volume
    res = vol.shape[0]
    x, y, z = np.mgrid[0:res, 0:res, 0:res]
    mlab.volume_slice(
        x, y, z, vol, vmin=0.0, vmax=1.0, plane_orientation="x_axes", transparent=True,
    )

    mlab.xlabel("x")
    mlab.ylabel("y")
    mlab.zlabel("z")
    mlab.colorbar(nb_labels=6, orientation="vertical")


def draw_point_cloud(point_cloud):
    points = np.asarray(point_cloud.points)
    colors = np.array(point_cloud.colors)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mlab.points3d(x, y, z)


def draw_grasps(grasps, qualities, finger_depth):
    for grasp, quality in zip(grasps, qualities):
        draw_grasp(grasp, quality, finger_depth)


def draw_grasp(grasp, q, d):
    if q == 0.0:
        x, y, z = grasp.pose.translation
        mlab.points3d(x, y, z, q, color=(0, 0, 1), scale_mode="none", scale_factor=1)
        return

    w = grasp.width
    lines = [
        [
            [0.0, -0.5 * w, d],
            [0.0, -0.5 * w, 0.0],
            [0.0, 0.5 * w, 0.0],
            [0.0, 0.5 * w, d],
        ],
        [[0.0, 0.0, 0.0], [0.0, 0.0, -0.5 * d]],
    ]
    radius = 0.033 * d

    for line in lines:
        points = [grasp.pose.transform_point(p) for p in line]
        draw_line_strip(points, q, radius)


def draw_line_strip(points, s, radius):
    """Draw a line between every two consecutive points.

    Args:
        points:  A list of points.
    """
    points = np.vstack(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    s = s * np.ones_like(x)

    mlab.plot3d(
        x, y, z, s, vmin=0.0, vmax=1.0, tube_radius=radius,
    )
