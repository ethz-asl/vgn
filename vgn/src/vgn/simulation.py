import time

import numpy as np
import pybullet
from pybullet_utils import bullet_client

from vgn import robot
from vgn.utils.transform import Rotation, Transform
from vgn.utils.camera_intrinsics import PinholeCameraIntrinsic


class Simulation(robot.Robot):
    """
    In this simulation, world, task and body frames are identical.

    Attributes:
        camera: A virtual camera.
        robot: A simulated robot hand.
        sim_time: The current virtual time.
    """

    def __init__(self, gui):
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self._p = bullet_client.BulletClient(connection_mode=connection_mode)

        self.hz = 240
        self.dt = 1. / self.hz
        self.solver_steps = 150
        self.real_time = gui
        self.sim_time = 0.

        # Initialize a virtual camera
        intrinsic = PinholeCameraIntrinsic(640, 480, 540., 540., 320., 240.)
        self.camera = Camera(intrinsic, 0.1, 2.0, self._p)

        # Static transform between tool0 and tcp
        self.T_tool0_tcp = Transform(Rotation.identity(), [0., 0., 0.08])
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()

        # Default initializations
        self._state_id = None

    def reset(self):
        """Reset the state of the physics simulation and create a new scene."""
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(fixedTimeStep=self.dt,
                                          numSolverIterations=self.solver_steps)
        self._p.setGravity(0., 0., -9.81)

        self.sim_time = 0.
        self._start_time = time.time()

    def spawn_plane(self):
        self._p.loadURDF('data/urdfs/plane/plane.urdf', [0., 0., 0.])

    def spawn_cuboid(self):
        """Spawn a cuboid at fixed pose for debugging purpose."""
        position = np.r_[0.1, 0.1, 0.2]
        self._p.loadURDF('data/urdfs/wooden_blocks/cuboid0.urdf', position)

        # wait for the object to rest
        while self.sim_time < 1.0:
            self._step()

    def spawn_objects(self):
        # position = np.r_[np.random.uniform(0.2*self.length,
        #                                    0.8*self.length, size=(2,)), 0.2]
        pass

    def spawn_robot(self):
        self._robot_uid = self._p.loadURDF('data/urdfs/hand/hand.urdf',
                                           basePosition=[0., 0., 1.])

    def save_state(self):
        """Save a snapshot of the current configuration."""
        self._state_id = self._p.saveState()

    def restore_state(self):
        """Restore the state of the simulation from the latest snapshot."""
        assert self._state_id is not None, 'save_state must be called first'
        self._p.restoreState(stateId=self._state_id)

    def set_tcp_pose(self, pose):
        T_body_tool0 = pose * self.T_tcp_tool0

        position = T_body_tool0.translation
        orientation = T_body_tool0.rotation.as_quat()

        self._p.resetBasePositionAndOrientation(self._robot_uid,
                                                position, orientation)

        self._cuid = self._p.createConstraint(
            parentBodyUniqueId=self._robot_uid,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0., 0., 0.],
            parentFramePosition=[0., 0., 0.],
            childFramePosition=position,
            childFrameOrientation=orientation)

    def move_tcp_xyz(self, displacement, eef_step):
        """
        Args:
            eef_step: Path interpolation resolution[m].
        """
        pass

    def open_gripper(self):
        self._p.setJointMotorControlArray(self._robot_uid,
                                          jointIndices=[0, 1],
                                          controlMode=pybullet.POSITION_CONTROL,
                                          targetPositions=[0.025, 0.025])
        for _ in range(self.hz):
            self._step()

    def close_gripper(self):
        self._p.setJointMotorControlArray(self._robot_uid,
                                          jointIndices=[0, 1],
                                          controlMode=pybullet.POSITION_CONTROL,
                                          targetPositions=[0.0, 0.0],
                                          forces=[15, 15])
        for _ in range(self.hz):
            self._step()

    def _step(self):
        self._p.stepSimulation()
        self.sim_time += self.dt
        if self.real_time:
            real_time = time.time()
            time.sleep(max(0., self.sim_time - real_time + self._start_time))


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic (PinholeCameraIntrinsic): The camera intrinsic parameters.
        near (float): The near plane of the virtual camera.
        far (float): The far plane of the virtual camera.
    """

    def __init__(self, intrinsic, near, far, physics_client):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self._proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self._p = physics_client

    def get_rgb_depth(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_eye_ref.
        """
        assert self._p.isNumpyEnabled(), 'Pybullet needs to be built with NumPy support'

        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order='F')
        gl_proj_matrix = self._proj_matrix.flatten(order='F')

        result = self._p.getCameraImage(width=self.intrinsic.width,
                                        height=self.intrinsic.height,
                                        viewMatrix=gl_view_matrix,
                                        projectionMatrix=gl_proj_matrix,
                                        renderer=pybullet.ER_TINY_RENDERER)

        rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = 1.*self.far*self.near/(self.far-(self.far-self.near)*z_buffer)
        return rgb, depth


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array([[intrinsic.fx, 0., -intrinsic.cx, 0.],
                            [0., intrinsic.fy, -intrinsic.cy, 0.],
                            [0., 0., near + far, near * far],
                            [0., 0., -1., 0.]])
    ortho = _gl_ortho(0., intrinsic.width, intrinsic.height, 0., near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag([2./(right-left), 2./(top-bottom), - 2./(far-near), 1.])
    ortho[0, 3] = - (right + left) / (right - left)
    ortho[1, 3] = - (top + bottom) / (top - bottom)
    ortho[2, 3] = - (far + near) / (far - near)
    return ortho
