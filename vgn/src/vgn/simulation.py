import time

import numpy as np
import pybullet
from pybullet_utils import bullet_client

from vgn import robot
from vgn.utils.camera import PinholeCameraIntrinsic
from vgn.utils.transform import Rotation, Transform


class Simulation(robot.Robot):
    """
    In this simulation, world, task and body frames are identical.

    Attributes:
        camera: A virtual camera.
        robot: A simulated robot hand.
        sim_time: The current virtual time.
    """
    def __init__(self, gui, real_time=False):
        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self._p = bullet_client.BulletClient(connection_mode=connection_mode)

        self.hz = 240
        self.dt = 1.0 / self.hz
        self.solver_steps = 150
        self.real_time = real_time
        self.sim_time = 0.0

        # Initialize a virtual camera
        intrinsic = PinholeCameraIntrinsic(640, 480, 540., 540., 320., 240.)
        self.camera = Camera(intrinsic, 0.1, 2.0, self._p)

        # Static transform between tool0 and tcp
        self.T_tool0_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.08])
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()

        # Default initializations
        self._state_id = None

    def step(self):
        """Perform one forward step of the physics simulation."""
        self._p.stepSimulation()
        self.sim_time += self.dt
        if self.real_time:
            time.sleep(max(0.0, self.sim_time - time.time() + self.start_time))

    def sleep(self, duration):
        """Pause the simulation for the given duration [s]."""
        time.sleep(duration)
        self.start_time += duration

    def reset(self):
        """Reset the state of the physics simulation and create a new scene."""
        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=self.solver_steps)
        self._p.setGravity(0.0, 0.0, -9.81)

        self.sim_time = 0.0
        self.start_time = time.time()

    def save_state(self):
        """Save a snapshot of the current configuration."""
        self._state_id = self._p.saveState()

    def restore_state(self):
        """Restore the state of the simulation from the latest snapshot."""
        assert self._state_id is not None, "save_state must be called first"
        self._p.restoreState(stateId=self._state_id)

    def spawn_plane(self):
        self._p.loadURDF("data/urdfs/plane/plane.urdf", [0.0, 0.0, 0.0])

    def spawn_debug_cylinder(self):
        position = np.r_[0.1, 0.1, 0.2]
        self._p.loadURDF("data/urdfs/wooden_blocks/cylinder.urdf", position)
        for _ in range(self.hz):
            self.step()

    def spawn_debug_cuboid(self):
        position = np.r_[0.1, 0.1, 0.2]
        self._p.loadURDF("data/urdfs/wooden_blocks/cuboid0.urdf", position)
        for _ in range(self.hz):
            self.step()

    def spawn_objects(self):
        # position = np.r_[np.random.uniform(0.2*self.length,
        #                                    0.8*self.length, size=(2,)), 0.2]
        pass

    def spawn_robot(self):
        position = [0.0, 0.0, 1.0]
        orientation = [0.0, 0.0, 0.0, 1.0]

        self.robot_uid = self._p.loadURDF(
            "data/urdfs/hand/hand.urdf",
            basePosition=position,
            baseOrientation=orientation,
        )
        self.cuid = self._p.createConstraint(
            parentBodyUniqueId=self.robot_uid,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=position,
            childFrameOrientation=orientation,
        )

    def get_tcp_pose(self):
        pos, ori = self._p.getBasePositionAndOrientation(self.robot_uid)
        T_world_tool0 = Transform(Rotation.from_quat(ori), list(pos))
        return T_world_tool0 * self.T_tcp_tool0.inverse()

    def get_gripper_opening_width(self):
        states = self._p.getJointStates(self.robot_uid, [0, 1])
        return 20.0 * (states[0][0] + states[1][0])

    def detect_collisions(self):
        return self._p.getContactPoints(self.robot_uid)

    def set_tcp_pose(self, pose, override_dynamics=True):
        T_world_tool0 = pose * self.T_tcp_tool0

        position = T_world_tool0.translation
        orientation = T_world_tool0.rotation.as_quat()

        if override_dynamics:
            self._p.resetBasePositionAndOrientation(self.robot_uid, position,
                                                    orientation)

        self._p.changeConstraint(
            self.cuid,
            jointChildPivot=position,
            jointChildFrameOrientation=orientation,
            maxForce=300,
        )

        self.step()
        return len(self.detect_collisions()) == 0

    def move_tcp_xyz(self,
                     target_pose,
                     eef_step=0.002,
                     check_collisions=False,
                     vel=0.10):
        pose = self.get_tcp_pose()
        pos_diff = target_pose.translation - pose.translation
        n_steps = int(np.linalg.norm(pos_diff) / eef_step)

        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            pose.translation += dist_step
            self.set_tcp_pose(pose, override_dynamics=False)
            for _ in range(int(dur_step * self.hz)):
                self.step()
            if check_collisions and self.detect_collisions():
                return False

        return True

    def open_gripper(self):
        self._p.setJointMotorControlArray(
            self.robot_uid,
            jointIndices=[0, 1],
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=[0.025, 0.025],
        )
        for _ in range(self.hz // 2):
            self.step()

    def close_gripper(self):
        self._p.setJointMotorControlArray(
            self.robot_uid,
            jointIndices=[0, 1],
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=[0.0, 0.0],
            forces=[10, 10],
        )
        for _ in range(self.hz // 2):
            self.step()


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
        assert (self._p.isNumpyEnabled()
                ), "Pybullet needs to be built with NumPy support"

        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self._proj_matrix.flatten(order="F")

        result = self._p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = (1.0 * self.far * self.near /
                 (self.far - (self.far - self.near) * z_buffer))
        return rgb, depth


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array([
        [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
        [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
        [0.0, 0.0, near + far, near * far],
        [0.0, 0.0, -1.0, 0.0],
    ])
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0])
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho
