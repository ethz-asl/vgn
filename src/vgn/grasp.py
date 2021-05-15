class Grasp(object):
    def __init__(self, pose, width, quality=None):
        self.pose = pose
        self.width = width
        self.quality = quality


def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width, grasp.quality)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width, grasp.quality)
