from pathlib2 import Path
import time

import numpy as np
import pybullet
import scipy.stats as stats

from vgn import Label
from vgn.perception import *
from vgn.utils import btsim, io, vis
from vgn.utils.transform import Rotation, Transform


class GraspSimulation(object):
    def __init__(self, object_set, config_path, random_state=None, gui=True):
        self.config = io.read_yaml(Path(config_path))

        self._urdf_root = Path(self.config["urdf_root"])
        self._object_set = object_set
        self._discover_object_urdfs()
        self._rng = random_state if random_state else np.random
        self._test = False if object_set in ["train"] else True
        self._global_scaling = {"blocks": 1.67}.get(object_set, 1.0)
        self._gui = gui

        self.world = btsim.BtWorld(self._gui)
        self.size = 6 * self.config["finger_depth"]

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self._setup_table()
        self._setup_camera()
        self._draw_task_space()
        self._drop_objects(object_count)

    def acquire_tsdf(self, num_viewpoints):
        tsdf = TSDFVolume(self.size, 40)
        high_res_tsdf = TSDFVolume(self.size, 120)

        t_world_center = np.r_[0.5 * self.size, 0.5 * self.size, 0.0]
        T_world_center = Transform(Rotation.identity(), t_world_center)

        for i in range(num_viewpoints):
            phi = 2.0 * np.pi * i / num_viewpoints
            theta = np.pi / 4.0
            r = 1.5 * self.size
            extrinsic = compute_viewpoint_on_hemisphere(T_world_center, phi, theta, r)
            depth_img = self.camera.render(extrinsic)[1]
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
        pc = high_res_tsdf.extract_point_cloud()

        return tsdf, pc

    def execute_grasp(self, T_world_grasp, remove=True):
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
        T_world_retreat = T_world_grasp * T_grasp_retreat

        gripper = Gripper(self.world, self.config)
        gripper.set_tcp(T_world_pregrasp)

        if gripper.detect_collision(threshold=0.0):
            result = Label.FAILURE, gripper.max_opening_width
        else:
            gripper.move_tcp_xyz(T_world_grasp)
            if gripper.detect_collision():
                result = Label.FAILURE, gripper.max_opening_width
            else:
                gripper.move(0.0)
                gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self._check_success(gripper):
                    result = Label.SUCCESS, gripper.read()
                    if remove:
                        contacts = self.world.get_contacts(gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, gripper.max_opening_width
        del gripper

        if remove:
            self._remove_and_wait()

        return result

    def _discover_object_urdfs(self):
        root = self._urdf_root / self._object_set
        self._urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

    def _setup_table(self):
        urdf = self._urdf_root / "table" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, 1.0 / 6.0 * self.size])
        self.world.load_urdf(urdf, pose, scale=0.6)

    def _setup_camera(self):
        intrinsic = PinholeCameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

    def _draw_task_space(self):
        points = vis.workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def _drop_objects(self, object_count):
        table_height = self.world.bodies[0].get_pose().translation[2]

        # place a box
        urdf = self._urdf_root / "table" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self._rng.choice(self._urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self._rng)
            xy = self._rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = 1.0 if self._test else self._rng.uniform(0.8, 1.0)
            body = self.world.load_urdf(urdf, pose, scale=self._global_scaling * scale)
            self._wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self._remove_and_wait()

    def _wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def _remove_objects_outside_workspace(self):
        removed_object = False
        for _, body in self.world.bodies.items():
            xy = body.get_pose().translation[:2]
            if np.any(xy < 0.0) or np.any(xy > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def _remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self._wait_for_objects_to_rest()
            removed_object = self._remove_objects_outside_workspace()

    def _check_success(self, gripper):
        # TODO this could be improved
        return gripper.read() > 0.1 * gripper.max_opening_width


class Gripper(object):
    def __init__(self, world, config):
        self.world = world
        self.body = None
        self.max_opening_width = config["max_opening_width"]

        self.T_body_tcp = Transform.from_dict(config["T_tool0_tcp"])
        self.T_tcp_body = self.T_body_tcp.inverse()

        self._urdf_path = Path(config["urdf_root"]) / "panda/hand.urdf"

    def __del__(self):
        self.world.remove_body(self.body)

    def _load(self, T_world_body):
        self.body = self.world.load_urdf(self._urdf_path, T_world_body)
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.world.add_constraint(  # constraint to keep fingers centered
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)

        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def _change_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        if self.body is None:  # spawn robot if necessary
            self._load(T_word_body)
        self.body.set_pose(T_word_body)
        self._change_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self._change_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_collision():
                return

    def detect_collision(self, threshold=10):
        contacts = self.world.get_contacts(self.body)
        for contact in contacts:
            if contact.force > threshold:
                return True
        return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
