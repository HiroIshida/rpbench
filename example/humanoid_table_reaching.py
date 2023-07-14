import time

import numpy as np
from skmp.robot.utils import set_robot_state
from skrobot.model.primitives import Axis
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from rpbench.articulated.jaxon.below_table import HumanoidTableReachingTask
from rpbench.articulated.jaxon.common import (
    InteractiveTaskVisualizer,
    StaticTaskVisualizer,
)

np.random.seed(1)

with mesh_simplify_factor(0.2):
    task = HumanoidTableReachingTask.sample(1, False)
res = task.solve_default()[0]
assert res.traj is not None

jaxon = task.config_provider.get_jaxon()
jaxon_config = task.config_provider.get_config()

vis = TrimeshSceneViewer()
task.world.visualize(vis)
co = task.descriptions[0][0]
axis = Axis.from_coords(co)
vis.add(jaxon)
vis.add(axis)
vis.show()

time.sleep(2)
for q in res.traj.resample(20):
    set_robot_state(jaxon, jaxon_config._get_control_joint_names(), q, base_type=BaseType.FLOATING)
    vis.redraw()
    time.sleep(0.5)
time.sleep(10)
