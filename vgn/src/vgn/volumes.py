from mayavi import mlab
import numpy as np
import open3d

from vgn.utils import ros_conversions, rviz_tools


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF.

    The volume is scaled to the unit cube.

    TODO
        * Handle scaling properly
    """

    def __init__(self, length, resolution):
        self._resolution = resolution
        self._voxel_length = length / self._resolution

        self._volume = open3d.integration.UniformTSDFVolume(
            length=length,
            resolution=self._resolution,
            sdf_trunc=4*self._voxel_length,
            color_type=open3d.integration.TSDFVolumeColorType.RGB8)

    def integrate(self, rgb, depth, intrinsic, extrinsic):
        """
        Args:
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from world to camera coordinages, T_eye_world.
        """
        rgbd = open3d.create_rgbd_image_from_color_and_depth(
            open3d.Image(rgb),
            open3d.Image(depth),
            depth_scale=1.0,
            depth_trunc=1.0,
            convert_rgb_to_intensity=False)

        intrinsic = open3d.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy)

        extrinsic = extrinsic.as_matrix()

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def draw_point_cloud(self):
        point_cloud = self._volume.extract_point_cloud()
        open3d.draw_geometries([point_cloud])
