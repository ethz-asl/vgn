import numpy as np


class PinholeCameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def project(self, point_cloud):
        raise NotImplementedError

    def deproject(self, depth_img):
        raise NotImplementedError
