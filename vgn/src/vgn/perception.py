from math import pi, cos, sin
import numpy as np
import open3d

from vgn.utils.transform import Transform


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF.

    The volume is scaled to the unit cube.

    TODO
        * Handle scaling properly
    """

    def __init__(self, length, resolution):
        self._resolution = resolution
        self._voxel_length = length / self._resolution
        self._sdf_trunc = 4 * self._voxel_length

        self._volume = open3d.integration.UniformTSDFVolume(
            length=length,
            resolution=self._resolution,
            sdf_trunc=self._sdf_trunc,
            color_type=open3d.integration.TSDFVolumeColorType.None)

    def integrate(self, rgb, depth, intrinsic, extrinsic):
        """
        Args:
            rgb
            depth
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

    def get_point_cloud(self):
        point_cloud = self._volume.extract_point_cloud()
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        normals = np.asarray(point_cloud.normals)
        return points, colors, normals

    def sample_surface_points(self, n=1):
        point_cloud = self._volume.extract_point_cloud()
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        # Sample random points from the point cloud
        indices = np.random.choice(len(points), size=n)

        # Estimate curvature at these points
        tree = open3d.KDTreeFlann(point_cloud)
        axes_of_principal_curvature = np.zeros((n, 3))
        for i, idx in enumerate(indices):
            [_, nn_idx, _] = tree.search_radius_vector_3d(points[idx], 0.02)
            cov = np.cov(np.asarray(points[nn_idx].T))
            w, v = np.linalg.eig(cov)
            axes_of_principal_curvature[i] = v[:, np.argmax(w)]

        return points[indices], normals[indices], axes_of_principal_curvature

    def draw_point_cloud(self):
        point_cloud = self._volume.extract_point_cloud()
        open3d.draw_geometries([point_cloud])


def random_viewpoints_on_hemisphere(n, length):
    """Generate random viewpoints on a half-sphere.

    Args:
        n: The number of viewpoints.
        length: The length of the workspace.
    """
    for _ in range(n):
        half_length = 0.5 * length

        phi = np.random.uniform(0., 2. * pi)
        theta = np.random.uniform(pi/6., 5.*pi/12.)
        r = 3 * half_length

        eye = np.array([r * sin(theta) * cos(phi) + half_length,
                        r * sin(theta) * sin(phi) + half_length,
                        r * cos(theta)])
        target = np.array([half_length, half_length, 0.])
        up = [0., 0., 1.]

        yield Transform.look_at(eye, target, up)
