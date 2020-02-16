import numpy as np
from mayavi import mlab

from vgn.utils.transform import Rotation, Transform


def show_sample(tsdf, qual, rot, width, mask):

    tsdf = tsdf.squeeze()
    qual = qual.squeeze()
    rot = rot.squeeze()
    width = width.squeeze()
    mask = mask.squeeze()

    mlab.figure()

    draw_volume(tsdf.squeeze())
    draw_grasps(mask, qual, rot, width)

    mlab.show()


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


def draw_grasps(mask, qual, rot, width):
    indices = np.where(mask == 1.0)

    for i, j, k in zip(*indices):

        q = qual[i, j, k]
        t = Transform(Rotation.from_quat(rot[:, i, j, k]), np.r_[i, j, k])
        w = width[i, j, k]
        d = 6  # TODO(mbreyer): use actual finger depth

        if q == 0.0:
            mlab.points3d(
                i, j, k, q, color=(0, 0, 1), scale_mode="none", scale_factor=1
            )
            continue

        lines = [
            (
                [0.0, -0.5 * w, d],
                [0.0, -0.5 * w, 0.0],
                [0.0, 0.5 * w, 0.0],
                [0.0, 0.5 * w, d],
            ),
            ([0.0, 0.0, 0.0], [0.0, 0.0, -0.5 * d]),
        ]

        for line in lines:
            points = np.array([])
            for p in line:
                points = np.concatenate([points, t.transform_point(p)])
            points = np.reshape(points, (3, -1), order="F")

            mlab.plot3d(
                points[0], points[1], points[2], color=(1, 0, 0), tube_radius=0.2
            )
