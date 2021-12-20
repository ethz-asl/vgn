import numpy as np

from robot_helpers.spatial import Transform, Rotation
from vgn.grasp import ParallelJawGrasp


class UniformPointCloudSampler:
    def __init__(self, gripper, rng):
        self.max_width = gripper.max_width
        self.max_depth = gripper.max_depth
        self.rng = rng

    def __call__(self, pc, count, eps=0.1):
        # TODO erode point cloud to avoid sampling at boundaries
        points, normals = np.asarray(pc.points), np.asarray(pc.normals)
        grasps = []
        for _ in range(count):
            i = self.rng.choice(len(points))
            point, normal = points[i], normals[i]
            pose = self.construct_grasp_frame(pc, point, normal)
            depth = self.rng.uniform(-eps * self.max_depth, (1 + eps) * self.max_depth)
            pose *= Transform.t([0, 0, -depth])
            grasps.append(ParallelJawGrasp(pose, self.max_width))
        return grasps

    def construct_grasp_frame(self, pc, point, normal):
        # TODO use curvature to construct frame
        z = -normal
        y = np.r_[z[1] - z[2], -z[0] + z[2], z[0] - z[1]]
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        return Transform(Rotation.from_matrix(np.vstack((x, y, z))), point)
