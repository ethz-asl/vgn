import numpy as np


def uniform(n, volume, min_z_offset, max_z_offset):
    """Uniformly sample grasp points from the reconstructed surface.
    
    Also, a random offset along the negative surface normal is applied to the
    fingertip position.
    """
    point_cloud = volume.get_point_cloud()
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    selection = np.random.choice(len(points), size=n)
    points, normals = points[selection], normals[selection]

    z_offsets = np.random.uniform(min_z_offset, max_z_offset, size=(n, 1))
    points = points - normals * z_offsets

    return points, normals
