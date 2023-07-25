import os
os.chdir('..')

from pathlib import Path

from vgn.experiments import clutter_removal

logdir = Path("data/experiments/...")

data = clutter_removal.Data(logdir)

print("Num grasps:        ", data.num_grasps())
print("Success rate:      ", data.success_rate())
print("Percent cleared:   ", data.percent_cleared())
print("Avg planning time: ", data.avg_planning_time())

import rospy
from vgn import vis

rospy.init_node("vgn_vis", anonymous=True)

failures = data.grasps[data.grasps["label"] == 0].index.tolist()
iterator = iter(failures)

i = next(iterator)
points, grasp, score, label = data.read_grasp(i)

vis.clear()
vis.draw_workspace(0.3)
vis.draw_points(points)
vis.draw_grasp(grasp, label, 0.05)