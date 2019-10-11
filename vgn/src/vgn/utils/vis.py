import numpy as np
from mayavi import mlab

from vgn.utils.transform import Rotation


def draw_voxels(voxels, tol=0.001):
    voxels = voxels.squeeze()

    x, y, z = np.where(voxels > tol)
    scalars = voxels[voxels > tol]

    mlab.points3d(
        x,
        y,
        z,
        scalars,
        vmin=0.0,
        vmax=1.0,
        mode="cube",
        scale_mode="none",
        scale_factor=1.0,
        opacity=0.05,
    )
    mlab.volume_slice(
        voxels, vmin=0.0, vmax=1.0, plane_orientation="x_axes", transparent=True
    )

    mlab.xlabel("x")
    mlab.ylabel("y")
    mlab.zlabel("z")
    mlab.colorbar(nb_labels=6, orientation="vertical")


def draw_frame(index, quat, scale=1.0):
    x, y, z = np.split(np.repeat(index, 3), 3)
    u, v, w = np.split(scale * Rotation.from_quat(quat).as_dcm().flatten(), 3)
    c = [1.0, 0.5, 0.0]

    axes = mlab.quiver3d(
        x,
        y,
        z,
        u,
        v,
        w,
        scalars=c,
        colormap="blue-red",
        mode="arrow",
        scale_mode="none",
        scale_factor=scale,
    )
    axes.glyph.color_mode = "color_by_scalar"


def draw_candidates(indices, quats, qualities, draw_frames=False):
    x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]
    mlab.points3d(
        x, y, z, qualities, vmin=0.0, vmax=1.0, scale_mode="none", scale_factor=0.5
    )

    if draw_frames:
        for index, quat in zip(indices, quats):
            draw_frame(index, quat)
