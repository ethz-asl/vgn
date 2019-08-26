import numpy as np
from mayavi import mlab


def draw_voxels(voxels, name=''):
    x, y, z = np.where(voxels > 0.001)
    scalars = voxels[voxels > 0.001]

    mlab.figure(name)
    mlab.points3d(
        x,
        y,
        z,
        scalars,
        vmin=0.,
        vmax=1.,
        mode='cube',
        opacity=0.01,
    )
    mlab.volume_slice(
        voxels,
        vmin=0.,
        vmax=1.,
        plane_orientation='x_axes',
        transparent=True,
    )
    mlab.colorbar(nb_labels=6, orientation='vertical')
