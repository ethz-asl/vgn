from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_tsdf(tsdf):
    n_slices = 6
    res = tsdf.shape[0]
    skip = res // n_slices

    fig = plt.figure()
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(1, n_slices),
                     share_all=True,
                     axes_pad=0.05,
                     cbar_mode='single',
                     cbar_location='right',
                     cbar_size='5%',
                     cbar_pad=None)

    for i in range(n_slices):
        ax = grid[i]
        ax.axis('off')
        img = ax.imshow(tsdf[i * skip, :, :], vmin=0.0, vmax=1.0)
        ax.cax.colorbar(img)


def plot_vgn(g):
    n_slices = 6
    res = g.shape[0]
    skip = res // n_slices

    fig = plt.figure()
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(1, n_slices),
                     share_all=True,
                     axes_pad=0.05,
                     cbar_mode='single',
                     cbar_location='right',
                     cbar_size='5%',
                     cbar_pad=None)

    for i in range(n_slices):
        ax = grid[i]
        ax.axis('off')
        img = ax.imshow(g[i * skip, :, :], vmin=0.0, vmax=1.0)
        ax.cax.colorbar(img)
