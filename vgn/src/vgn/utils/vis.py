import numpy as np
from mayavi import mlab

from vgn.grasp import Label
from vgn.perception.integration import TSDFVolume
from vgn.utils.transform import Rotation


def display_scene(scene_data, vol_size, vol_res):

    tsdf = TSDFVolume(vol_size, vol_res)
    tsdf.integrate_images(
        scene_data.depth_imgs, scene_data.intrinsic, scene_data.extrinsics
    )
    tsdf_vol = tsdf.get_volume()
    point_cloud = tsdf.extract_point_cloud()

    mlab.figure()

    draw_volume(tsdf_vol.squeeze(), tsdf.voxel_size)
    draw_point_cloud(point_cloud)
    draw_grasps(scene_data.grasps, scene_data.labels, draw_frames=True)

    mlab.show()


def draw_volume(vol, voxel_size, tol=0.001):
    (x, y, z), scalars = np.where(vol > tol), vol[vol > tol]

    # Draw the volume
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

    # Draw a slice through the volume
    res = vol.shape[0]
    x, y, z = np.mgrid[0:res, 0:res, 0:res]
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


def draw_grasps(grasps, labels, draw_frames=True):

    for grasp, label in zip(grasps, labels):
        x, y, z = grasp.pose.translation
        s = 0.0 if label < Label.SUCCESS else 1.0

        mlab.points3d(
            x, y, z, s, vmin=0.0, vmax=1.0, scale_mode="none", scale_factor=0.004
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
