import numpy as np
import open3d

import vgn.config as cfg


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.resolution = resolution
        self.voxel_size = size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = open3d.integration.UniformTSDFVolume(
            length=size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=getattr(open3d.integration.TSDFVolumeColorType, "None"),
        )

    def integrate(self, depth, intrinsic, extrinsic):
        """
        Args:
            depth: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from world to camera coordinages, T_eye_world.
        """
        rgbd = open3d.geometry.create_rgbd_image_from_color_and_depth(
            open3d.geometry.Image(np.empty_like(depth)),
            open3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = open3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic.as_matrix()

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def integrate_images(self, depth_imgs, intrinsic, extrinsics):
        for depth_img, extrinsic in zip(depth_imgs, extrinsics):
            self.integrate(depth_img, intrinsic, extrinsic)

    def get_index(self, position):
        return np.round(position / self.voxel_size).astype(np.int)

    def get_volume(self):
        """Return voxel volume with truncated signed distances."""
        shape = (self.resolution, self.resolution, self.resolution)
        tsdf_vol = np.zeros(shape, dtype=np.float32)
        voxels = self._volume.extract_voxel_grid().voxels
        for voxel in voxels:
            i, j, k = voxel.grid_index
            tsdf_vol[i, j, k] = voxel.color[0]
        return tsdf_vol

    def extract_point_cloud(self):
        """Return extracted point cloud as open3d.PointCloud object."""
        return self._volume.extract_point_cloud()
