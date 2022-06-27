from vgn.utils import *
from robot_helpers.ros.rviz import *

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

cm = lambda s: tuple([float(1 - s), float(s), float(0)])


class Visualizer:
    def __init__(self, base_frame="panda_link0"):
        self.base_frame = base_frame
        self.create_marker_publisher()
        self.create_scene_cloud_publisher()
        self.create_map_cloud_publisher()
        self.create_quality_publisher()

    def create_marker_publisher(self, topic="visualization_marker_array"):
        self.marker_pub = rospy.Publisher(topic, MarkerArray, queue_size=1)

    def create_scene_cloud_publisher(self, topic="scene_cloud"):
        self.scene_cloud_pub = rospy.Publisher(topic, PointCloud2, queue_size=1)

    def create_map_cloud_publisher(self, topic="map_cloud"):
        self.map_cloud_pub = rospy.Publisher(topic, PointCloud2, queue_size=1)

    def create_quality_publisher(self, topic="quality"):
        self.quality_pub = rospy.Publisher(topic, PointCloud2, queue_size=1)

    def clear(self):
        self.clear_markers()
        self.clear_clouds()
        self.clear_quality()

    def clear_markers(self):
        self.draw([Marker(action=Marker.DELETEALL)])

    def clear_clouds(self):
        msg = to_cloud_msg(self.base_frame, np.array([]))
        self.scene_cloud_pub.publish(msg)
        self.map_cloud_pub.publish(msg)

    def clear_quality(self):
        msg = to_cloud_msg(self.base_frame, np.array([]))
        self.quality_pub.publish(msg)

    def clear_grasp(self):
        markers = [Marker(action=Marker.DELETE, ns="grasp", id=i) for i in range(4)]
        self.draw(markers)

    def roi(self, frame, size):
        pose = Transform.identity()
        scale = [size * 0.005, 0.0, 0.0]
        color = [0.5, 0.5, 0.5]
        lines = box_lines(np.full(3, 0), np.full(3, size))
        msg = create_line_list_marker(frame, pose, scale, color, lines, ns="roi")
        self.draw([msg])

    def scene_cloud(self, frame, points):
        msg = to_cloud_msg(frame, points)
        self.scene_cloud_pub.publish(msg)

    def map_cloud(self, frame, points, distances):
        msg = to_cloud_msg(frame, points, distances=distances)
        self.map_cloud_pub.publish(msg)

    def quality(self, frame, voxel_size, grid, threshold=0.9):
        points, values = grid_to_map_cloud(voxel_size, grid, threshold)
        msg = to_cloud_msg(frame, points, intensities=values)
        self.quality_pub.publish(msg)

    def grasp(self, frame, grasp, quality, vmin=0.5, vmax=1.0):
        color = cm((quality - vmin) / (vmax - vmin))
        self.draw(create_grasp_markers(frame, grasp, color, "grasp"))

    def grasps(self, frame, grasps, qualities, vmin=0.5, vmax=1.0):
        markers = []
        for i, (grasp, quality) in enumerate(zip(grasps, qualities)):
            color = cm((quality - vmin) / (vmax - vmin))
            markers.append(create_grasp_marker(frame, grasp, color, "grasps", i))
        self.draw(markers)

    def draw(self, markers):
        self.marker_pub.publish(MarkerArray(markers=markers))


def create_grasp_markers(frame, grasp, color, ns, id=0, depth=0.046, radius=0.005):
    # Nicer looking grasp marker drawn with 4 Marker.CYLINDER
    w, d = grasp.width, depth
    pose = grasp.pose * Transform.t_[0.0, -w / 2, d / 2]
    scale = [radius, radius, d]
    left = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id)
    pose = grasp.pose * Transform.t_[0.0, w / 2, d / 2]
    scale = [radius, radius, d]
    right = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id + 1)
    pose = grasp.pose * Transform.t_[0.0, 0.0, -d / 4]
    scale = [radius, radius, d / 2]
    wrist = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id + 2)
    pose = grasp.pose * Transform.from_rotation(Rotation.from_rotvec([np.pi / 2, 0, 0]))
    scale = [radius, radius, w]
    palm = create_marker(Marker.CYLINDER, frame, pose, scale, color, ns, id + 3)
    return [left, right, wrist, palm]


def create_grasp_marker(frame, grasp, color, ns, id=0, depth=0.05, radius=0.005):
    # Faster grasp marker using Marker.LINE_LIST
    pose, w, d, scale = grasp.pose, grasp.width, depth, [radius, 0.0, 0.0]
    points = [[0, -w / 2, d], [0, -w / 2, 0], [0, w / 2, 0], [0, w / 2, d]]
    return create_line_strip_marker(frame, pose, scale, color, points, ns, id)
