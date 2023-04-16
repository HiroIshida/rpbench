import time

import numpy as np

from rpbench.pr2.common import InteractiveTaskVisualizer
from rpbench.pr2.kivapod import KivapodEmptyReachingTask

np.random.seed(3)

# task = KivapodEmptyReachingTask.sample(30, standard=False)
# vis = InteractiveTaskVisualizer(task)
# vis.show()
# time.sleep(100)

task = KivapodEmptyReachingTask.sample(1, standard=True)
res = task.solve_default()[0]
assert res.traj is not None

vis = InteractiveTaskVisualizer(task)
vis.show()
vis.visualize_trajectory(res.traj.resample(30))
time.sleep(2000)
