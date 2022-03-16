import numpy as np
import pybullet as p
import skimage.transform
import time

from robot_helpers.bullet import BtCamera
from robot_helpers.spatial import Rotation, Transform


def load_urdf(urdf, pose, scale=1.0):
    ori, pos = pose.rotation.as_quat(), pose.translation
    return p.loadURDF(str(urdf), pos, ori, globalScaling=scale)


class GraspSim:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self._configure_physics_engine()
        self._configure_visualizer()
        self.robot = PandaGripper(self)
        self.camera = BtCamera(320, 240, 1.047, 0.1, 2.0, renderer=p.ER_TINY_RENDERER)
        self.object_uids = []
        self.support_uid = -1
        self.robot.reset(Transform.t_[0, 0, 10], self.robot.max_width)

    @property
    def object_count(self):
        return len(self.object_uids)

    def load_support(self, pose, scale):
        self.support_uid = load_urdf("plane/model.urdf", pose, scale)

    def load_object(self, urdf, pose, scale):
        uid = load_urdf(urdf, pose, scale)
        self.object_uids.append(uid)
        p.changeDynamics(uid, -1, lateralFriction=self.cfg["lateral_friction"])
        return uid

    def remove_object(self, uid):
        p.removeBody(uid)
        self.object_uids.remove(uid)

    def clear(self):
        p.removeBody(self.support_uid)
        self.support_uid = -1
        for uid in list(self.object_uids):
            self.remove_object(uid)

    def step(self):
        p.stepSimulation()
        if self.cfg["gui"]:
            time.sleep(self.dt)

    def forward(self, duration):
        for _ in range(int(duration / self.dt)):
            self.step()

    def save_state(self):
        return p.saveState()

    def restore_state(self, id):
        p.restoreState(stateId=id)

    def wait_for_objects_to_rest(self):
        self.forward(1.0)  # TODO

    def _configure_physics_engine(self):
        self.dt = 1.0 / 240.0
        p.connect(p.GUI if self.cfg["gui"] else p.DIRECT)
        p.setAdditionalSearchPath(self.cfg.get("urdf_root", "assets/urdfs"))
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt)
        p.setGravity(0.0, 0.0, -9.81)

    def _configure_visualizer(self):
        p.resetDebugVisualizerCamera(0.6, 0.0, -30, [0.15, 0.0, 0.3])


def generate_pile(sim, origin, size, urdfs, scales):
    center = origin * Transform.t_[0.5 * size, 0.5 * size, 0]
    sim.load_support(center, size)
    uid = load_urdf("box/model.urdf", origin * Transform.t_[0.02, 0.02, 0], 1.3)
    for urdf, scale in zip(urdfs, scales):
        loc_ori = Rotation.random(random_state=sim.rng)
        loc_pos = np.r_[sim.rng.uniform(1.0 / 3.0 * size, 2.0 / 3.0 * size, 2), 0.2]
        sim.load_object(urdf, origin * Transform(loc_ori, loc_pos), scale)
        sim.wait_for_objects_to_rest()
    p.removeBody(uid)
    sim.wait_for_objects_to_rest()
    remove_objects_outside_roi(sim, origin, size)


def generate_packed(sim, origin, size, urdfs, scales, max_attempts=10):
    center = origin * Transform.t_[0.5 * size, 0.5 * size, 0]
    sim.load_support(center, size)
    for urdf, scale in zip(urdfs, scales):
        uid = sim.load_object(urdf, Transform.identity(), scale)
        lower, upper = p.getAABB(uid)
        z_offset = 0.5 * (upper[2] - lower[2]) + 0.002
        state_id = p.saveState()
        for _ in range(max_attempts):
            local_ori = Rotation.from_rotvec([0, 0, sim.rng.uniform(2 * np.pi)])
            local_pos = np.r_[sim.rng.uniform(0.2, 0.8, 2) * size, z_offset]
            pose = origin * Transform(local_ori, local_pos)
            p.resetBasePositionAndOrientation(
                uid,
                pose.translation,
                pose.rotation.as_quat(),
            )
            sim.step()
            if p.getContactPoints(uid):
                p.restoreState(stateId=state_id)
            else:
                break
        else:
            sim.remove_object(uid)
    sim.wait_for_objects_to_rest()
    remove_objects_outside_roi(sim, origin, size)


def remove_objects_outside_roi(sim, origin, size):
    for uid in sim.object_uids:
        xyz = np.asarray(p.getBasePositionAndOrientation(uid)[0])
        xyz = xyz - origin.translation
        if np.any(xyz < 0.0) or np.any(xyz > size):
            sim.remove_object(uid)


scene_fns = {
    "pile": generate_pile,
    "packed": generate_packed,
}


def get_scene(name):
    return scene_fns[name]


class PandaGripper:
    def __init__(self, sim):
        self.sim = sim
        self.max_width = 0.08
        self.max_depth = 0.05
        self.T_ee_com = Transform.t_[0.0, 0.0, -0.026]
        self.uid = p.loadURDF("assets/urdfs/panda/hand.urdf")
        self._create_joints()

    @property
    def width(self):
        return p.getJointState(self.uid, 0)[0] + p.getJointState(self.uid, 1)[0]

    @property
    def contacts(self):
        return p.getContactPoints(self.uid)

    def reset(self, pose, width):
        self._reset_pose(pose)
        self._reset_fingers(width)

    def pose(self):
        pos, ori = p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), pos) * self.T_ee_com.inv()

    def set_desired_pose(self, pose):
        pose = pose * self.T_ee_com
        p.changeConstraint(
            self.fixed_joint_uid,
            jointChildPivot=pose.translation,
            jointChildFrameOrientation=pose.rotation.as_quat(),
            maxForce=50,
        )

    def set_desired_joint_velocity(self, velocity, force):
        p.setJointMotorControlArray(
            self.uid,
            [0, 1],
            p.VELOCITY_CONTROL,
            targetVelocities=[velocity, velocity],
            forces=[force, force],
        )

    def moveL(self, desired, velocity=0.1, allow_contact=False):
        current = self.pose()
        diff = desired.translation - current.translation
        distance = np.linalg.norm(diff)
        direction = diff / distance
        step_count = int(distance / velocity / self.sim.dt)
        for _ in range(step_count):
            current.translation += direction * self.sim.dt * velocity
            self.set_desired_pose(current)
            self.sim.step()
            if not allow_contact and self.contacts:
                return

    def grasp(self, force=5):
        self.set_desired_joint_velocity(-0.1, force)
        self.sim.forward(0.5)

    def _create_joints(self):
        self.fixed_joint_uid = p.createConstraint(
            parentBodyUniqueId=self.uid,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=[0.0, 0.0, 0.0],
        )
        mimic_joint_uid = p.createConstraint(
            self.uid,
            0,
            self.uid,
            1,
            p.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        p.changeConstraint(mimic_joint_uid, gearRatio=-1, erp=0.1, maxForce=50)

    def _reset_pose(self, pose):
        self.set_desired_pose(pose)
        pose = pose * self.T_ee_com
        p.resetBasePositionAndOrientation(
            self.uid,
            pose.translation,
            pose.rotation.as_quat(),
        )

    def _reset_fingers(self, width):
        p.resetJointState(self.uid, 0, 0.5 * width)
        p.resetJointState(self.uid, 1, 0.5 * width)
        self.set_desired_joint_velocity(0.0, 0.0)


class GraspQualityMetric:
    def __init__(self, sim):
        self.sim = sim
        self.robot = sim.robot
        self.rng = sim.rng


class DynamicMetric(GraspQualityMetric):
    def __call__(self, grasp):
        self.robot.reset(grasp.pose, grasp.width)
        self.sim.step()
        if not self.robot.contacts:
            self.robot.grasp()
            self.robot.moveL(Transform.t_[0, 0, 0.1] * grasp.pose, allow_contact=True)
            contacts = self.robot.contacts
            if self.robot.width > 0.1 * self.robot.max_width and contacts:
                return 1.0, {"object_uid": contacts[0][2]}
        return 0.0, {}


class DynamicWithApproachMetric(GraspQualityMetric):
    def __call__(self, grasp):
        self.robot.reset(grasp.pose * Transform.t_[0.0, 0.0, -0.05], grasp.width)
        self.sim.step()
        if not self.robot.contacts:
            self.robot.moveL(grasp.pose)
            self.robot.grasp()
            self.robot.moveL(Transform.t_[0, 0, 0.1] * grasp.pose, allow_contact=True)
            contacts = self.robot.contacts
            if self.robot.width > 0.1 * self.robot.max_width and contacts:
                return 1.0, {"object_uid": contacts[0][2]}
        return 0.0, {}


grasp_metrics = {
    "dynamic": DynamicMetric,
    "dynamic_with_approach": DynamicWithApproachMetric,
}


def get_metric(name):
    return grasp_metrics[name]


def apply_noise(img, k=1000, theta=0.001, sigma=0.005, l=4.0):
    # Multiplicative and additive noise
    img *= np.random.gamma(k, theta)
    h, w = img.shape
    noise = np.random.randn(int(h / l), int(w / l)) * sigma
    img += skimage.transform.resize(noise, img.shape, order=1, mode="constant")
    return img
