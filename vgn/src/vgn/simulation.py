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
    def __init__(self, object_set, config_path, gui=True):
        assert object_set in ["debug", "train", "test", "adversarial"]
        config = io.load_dict(Path(config_path))

        self._urdf_root = Path(config["urdf_root"])
        self._object_set = object_set
        self._gripper_config = config["gripper"]
        self._test = False if object_set in ["train"] else True
        self._gui = gui

        self.size = 4 * self._gripper_config["max_opening_width"]
        self.world = btsim.BtWorld(self._gui)

        if object_set == "debug":
            self._place_objects = self._place_cuboid
        else:
            self._discover_object_models()
            self._place_objects = self._generate_heap

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self._setup_table()
        self._setup_camera()
        self._draw_task_space()
        self._place_objects()

    def acquire_tsdf(self, num_viewpoints=6):
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

        return tsdf, high_res_tsdf.extract_point_cloud()

    def execute_grasp(self, grasp, remove=False):
        T_world_grasp = grasp.pose

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
        T_world_retreat = T_world_grasp * T_grasp_retreat

        robot = Robot(self.world, self._gripper_config, T_world_pregrasp)
        if robot.detect_collision(threshold=0.0):
            result = Label.FAILURE, 0.0
        else:
            robot.move_tcp_xyz(T_world_grasp)
            if robot.detect_collision():
                result = Label.FAILURE, 0.00
            else:
                robot.gripper.move(0.0)
                robot.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self._check_success(robot):
                    result = Label.SUCCESS, robot.gripper.read()
                    if remove:
                        contacts = self.world.check_contacts(robot._body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, 0.0
        del robot

        return result

    def _discover_object_models(self):
        root = self._urdf_root / self._object_set
        self._model_paths = [
            d / (d.name + ".urdf") for d in root.iterdir() if d.is_dir()
        ]

    def _setup_table(self):
        plane = self.world.load_urdf(self._urdf_root / "plane/plane.urdf")
        plane.set_pose(Transform(Rotation.identity(), [0.0, 0.0, 0.0]))

    def _setup_camera(self):
        intrinsic = PinholeCameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

    def _place_cuboid(self):
        model_path = self._urdf_root / "cuboid/cuboid.urdf"
        position = np.r_[0.5 * self.size, 0.5 * self.size, 0.03]
        orientation = Rotation.identity()
        self._drop_object(model_path, Transform(orientation, position))

    def _generate_heap(self):
        object_count = np.random.poisson(4 - 1) + 1
        model_paths = np.random.choice(self._model_paths, size=object_count)
        for model_path in model_paths:
            planar_position = self._sample_planar_position()
            pose = Transform(Rotation.random(), np.r_[planar_position, 0.15])
            scale = 1.0 if self._test else np.random.uniform(0.8, 1.0)
            self._drop_object(model_path, pose, scale)

    def _sample_planar_position(self):
        l, u = 0.0, self.size
        mu, sigma = self.size / 2.0, self.size / 4.0
        X = stats.truncnorm((l - mu) / sigma, (u - mu) / sigma, loc=mu, scale=sigma)
        return X.rvs(2)

    def _drop_object(self, model_path, pose, scale=1.0):
        body = self.world.load_urdf(model_path, scale=scale)
        body.set_pose(pose)
        for _ in range(240):
            self.world.step()

    def _check_success(self, robot):
        # TODO this can be improved
        return robot.gripper.read() > 0.1 * robot.gripper.max_opening_width

    def _draw_task_space(self):
        points = vis.workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )


class Robot(object):
    def __init__(self, world, config, pose):
        self._world = world
        self._T_tool0_tcp = Transform.from_dict(config["T_tool0_tcp"])
        self._T_tcp_tool0 = self._T_tool0_tcp.inverse()
        self._body = self._world.load_urdf(config["urdf_path"])
        self._constraint = self._world.add_constraint(
            self._body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        )
        self.set_tcp(pose)

        self.gripper = Gripper(self._world, self._body, config)

    def __del__(self):
        self._world.remove_body(self._body)

    def set_tcp(self, target):
        T_world_tool0 = target * self._T_tcp_tool0
        self._body.set_pose(T_world_tool0)
        self._constraint.change(T_world_tool0, max_force=300)
        self._world.step()

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        pose = self._body.get_pose() * self._T_tool0_tcp

        pos_diff = target.translation - pose.translation
        n_steps = int(np.linalg.norm(pos_diff) / eef_step)
        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            pose.translation += dist_step
            self._constraint.change(pose * self._T_tcp_tool0, max_force=300)
            for _ in range(int(dur_step / self._world.dt)):
                self._world.step()
            if abort_on_contact and self.detect_collision():
                return

    def detect_collision(self, threshold=10):
        contacts = self._world.check_contacts(self._body)
        for contact in contacts:
            if contact.force > threshold:
                return True
        return False


class Gripper(object):
    def __init__(self, world, body, config):
        self._world = world
        self._body = body
        self._left_joint = self._body.joints["finger_l"]
        self._right_joint = self._body.joints["finger_r"]

        self.max_opening_width = config["max_opening_width"]

        init_pos = 0.5 * self.max_opening_width
        self._left_joint.set_position(init_pos, override_dynamics=True)
        self._right_joint.set_position(init_pos, override_dynamics=True)

    def move(self, width):
        self._left_joint.set_position(0.5 * width)
        self._right_joint.set_position(0.5 * width)
        for _ in range(int(0.5 / self._world.dt)):
            self._world.step()

    def read(self):
        pos_l = self._left_joint.get_position()
        pos_r = self._right_joint.get_position()
        width = pos_l + pos_r
        return width
