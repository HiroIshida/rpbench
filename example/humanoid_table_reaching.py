import argparse
import time

import numpy as np
from skrobot.utils.urdf import mesh_simplify_factor

from rpbench.articulated.jaxon.common import (
    InteractiveTaskVisualizer,
    StaticTaskVisualizer,
    TaskVisualizerBase,
)
from rpbench.articulated.jaxon.ground import HumanoidGroundTableRarmReachingTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, default="interactive")
    parser.add_argument("-seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    with mesh_simplify_factor(0.2):
        task = HumanoidGroundTableRarmReachingTask.sample(1, False)

    vis: TaskVisualizerBase
    if args.mode == "debug":
        vis = InteractiveTaskVisualizer(task)
        vis.show()
        time.sleep(20)
    else:
        res = task.solve_default()[0]
        assert res.traj is not None

        jaxon = task.config_provider.get_jaxon()
        jaxon_config = task.config_provider.get_config()

        if args.mode == "static":
            vis = StaticTaskVisualizer(task)
            vis.save_trajectory_gif(res.traj.resample(10), "jaxon_demo.gif")
        elif args.mode == "interactive":
            vis = InteractiveTaskVisualizer(task)
            vis.show()
            vis.visualize_trajectory(res.traj.resample(10))
            time.sleep(10)
        else:
            assert False
