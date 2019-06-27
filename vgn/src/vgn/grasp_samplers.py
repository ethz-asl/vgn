
import numpy as np


from vgn.utils.transform import Rotation, Transform


def uniform_grasps_on_surface(n, volume):
    """Sample grasp candidates uniformly from a point cloud.

    The approach vector and yaw are deduced from the local geometry.
    """
    points, normals, curvatures = volume.sample_surface_points(n)
    grasp_candidates = []

    for point, normal, axis_of_principal_curvature in zip(points, normals, curvatures):

        R = np.empty((3, 3))
        R[:, 0] = axis_of_principal_curvature
        R[:, 1] = np.cross(axis_of_principal_curvature, normal)
        R[:, 2] = -normal

        grasp_pose = Transform(Rotation.from_dcm(R), point)
        grasp_candidates.append(grasp_pose)

    return grasp_candidates
