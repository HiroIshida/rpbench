import time

import numpy as np

from rpbench.pr2.common import PR2InteractiveTaskVisualizer, PR2StaticTaskVisualizer
from rpbench.pr2.kivapod import KivapodEmptyReachingTask

np.random.seed(3)

save_visualization_result = False

task = KivapodEmptyReachingTask.sample(1, standard=True)
res = task.solve_default()[0]
assert res.traj is not None

if save_visualization_result:
    static_vis = PR2StaticTaskVisualizer.from_task(task)
    static_vis.save_trajectory_gif(res.traj.resample(30), "./kivapod_result.gif")
else:
    interactive_vis = PR2InteractiveTaskVisualizer.from_task(task)
    interactive_vis.show()
    interactive_vis.visualize_trajectory(res.traj.resample(30))
    time.sleep(2000)
