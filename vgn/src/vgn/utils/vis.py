import numpy as np
from mayavi import mlab


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
        mode='cube',
        scale_mode='none',
        scale_factor=1.0,
        opacity=0.05,
    )
    mlab.volume_slice(
        voxels,
        vmin=0.0,
        vmax=1.0,
        plane_orientation='x_axes',
        transparent=True,
    )

    mlab.xlabel('x')
    mlab.ylabel('y')
    mlab.zlabel('z')
    mlab.colorbar(nb_labels=6, orientation='vertical')


def draw_candidates(indices, scores):
    x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]
    mlab.points3d(
        x,
        y,
        z,
        scores,
        vmin=0.0,
        vmax=1.0,
        scale_mode='none',
        scale_factor=0.5,
    )
