"""Various samplers of grasp candidates."""

import numpy as np

from vgn.utils.transform import Rotation, Transform


def uniform(n, volume):
    """Sample grasp candidates uniformly from a point cloud."""
    points, normals, curvatures = volume.sample_surface_points(n)
    grasp_candidates = []

    for point, normal, axis_of_principal_curvature in zip(points, normals, curvatures):

        # TODO First, define a frame on the object surface
        # normal and principal curvature might not be orthogonal
        R = np.empty((3, 3))
        R[:, 0] = axis_of_principal_curvature
        R[:, 1] = np.cross(axis_of_principal_curvature, normal)
        R[:, 2] = -normal
        T_world_surface = Transform(Rotation.from_dcm(R), point)

        # Next, add a random offset along the approach vector
        z_offset = np.random.uniform(0., 0.05)
        T_surface_grasp = Transform(Rotation.identity(),
                                    np.r_[0., 0., z_offset])

        T_world_grasp = T_world_surface * T_surface_grasp
        grasp_candidates.append(T_world_grasp)

    return grasp_candidates
