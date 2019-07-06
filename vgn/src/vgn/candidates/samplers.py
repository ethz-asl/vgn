import numpy as np
import open3d

from vgn.utils.transform import Rotation, Transform


def uniform(point_cloud, n):
    """Sample grasp candidates uniformly from a point cloud."""
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    # Build a kd-tree for efficient lookup
    kdtree = open3d.KDTreeFlann(point_cloud)
    radius = 0.025

    selection = np.random.choice(len(points), size=n, replace=False)

    candidates = []
    for point in points[selection]:
        candidate = estimate_frame(normals, point, radius, kdtree)

        # Randonly shift frame along the approach vector
        offset = Transform(Rotation.identity(),
                           np.r_[0., 0., np.random.uniform(0., 0.05)])

        candidates.append(candidate * offset)

    return candidates


def estimate_frame(normals, query, radius, kdtree):
    """Estimate grasp frame on the surface using a PCA of the local normals.

    TODO
        * Ensure normal is pointing in the same direction as existing ones
    """

    [_, nn_idx, _] = kdtree.search_radius_vector_3d(query, radius)

    m = np.dot(normals[nn_idx].T, normals[nn_idx])
    w, v = np.linalg.eigh(m)

    normal = v[:, np.argmax(w)]
    curvature = v[:, np.argmin(w)]
    binormal = np.cross(curvature, normal)

    r = np.vstack((curvature, binormal, -normal)).T
    return Transform(Rotation.from_dcm(r), query)
