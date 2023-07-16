import time

import numpy as np
from skrobot.utils.urdf import mesh_simplify_factor

from rpbench.articulated.jaxon.common import InteractiveTaskVisualizer
from rpbench.articulated.jaxon.ground import HumanoidGroundRarmReachingTask

np.random.seed(2)
with mesh_simplify_factor(0.2):
    task = HumanoidGroundRarmReachingTask.sample(1, standard=False)

ts = time.time()
res = task.solve_default()[0]
assert res.traj is not None
print("time to solve: {}".format(time.time() - ts))

vis = InteractiveTaskVisualizer(task)
vis.show()
vis.visualize_trajectory(res.traj.resample(10))
time.sleep(10)
