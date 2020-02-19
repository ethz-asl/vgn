import numpy as np
import open3d as o3d


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=getattr(o3d.integration.TSDFVolumeColorType, "None"),
        )

    def integrate(self, depth_img, intrinsic, extrinsic):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic.as_matrix()

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_volume(self):
        """Return voxel volume with truncated signed distances."""
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_vol = np.zeros(shape, dtype=np.float32)
        voxels = self._volume.extract_voxel_grid().voxels
        for voxel in voxels:
            i, j, k = voxel.grid_index
            tsdf_vol[0, i, j, k] = voxel.color[0]
        return tsdf_vol

    def extract_point_cloud(self):
        """Return extracted point cloud as o3d.PointCloud object."""
        return self._volume.extract_point_cloud()
