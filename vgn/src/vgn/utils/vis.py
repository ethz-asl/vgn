import numpy as np
from mayavi import mlab

from vgn.utils.transform import Rotation


def draw_volume(vol, voxel_size, tol=0.001):
    (x, y, z), scalars = np.where(vol > tol), vol[vol > tol]
    mlab.points3d(
        x * voxel_size,
        y * voxel_size,
        z * voxel_size,
        scalars,
        vmin=0.0,
        vmax=1.0,
        mode="cube",
        scale_mode="none",
        scale_factor=voxel_size,
        opacity=0.05,
    )

    x, y, z = np.mgrid[0:40, 0:40, 0:40]
    mlab.volume_slice(
        x * voxel_size,
        y * voxel_size,
        z * voxel_size,
        vol,
        vmin=0.0,
        vmax=1.0,
        plane_orientation="x_axes",
        transparent=True,
    )

    mlab.xlabel("x")
    mlab.ylabel("y")
    mlab.zlabel("z")
    mlab.colorbar(nb_labels=6, orientation="vertical")


def draw_point_cloud(point_cloud):
    points = np.asarray(point_cloud.points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mlab.points3d(x, y, z, color=(0.4, 0.4, 0.4), scale_mode="none", scale_factor=0.002)


def draw_grasps(grasps, qualities, draw_frames=True):
    for grasp, quality in zip(grasps, qualities):
        x, y, z = grasp.pose.translation
        mlab.points3d(
            x, y, z, quality, vmin=0.0, vmax=1.0, scale_mode="none", scale_factor=0.004
        )

        if draw_frames:
            x, y, z = np.split(np.repeat([x, y, z], 3), 3)
            u, v, w = np.split(grasp.pose.rotation.as_dcm().flatten(), 3)
            axes = mlab.quiver3d(
                x,
                y,
                z,
                u,
                v,
                w,
                scalars=[1.0, 0.5, 0.0],
                colormap="blue-red",
                mode="arrow",
                scale_mode="none",
                scale_factor=0.01,
            )
            axes.glyph.color_mode = "color_by_scalar"
