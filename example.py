import numpy as np
from ompl import Algorithm, set_ompl_random_seed
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.tabletop import (
    InteractiveTaskVisualizer,
    StaticTaskVisualizer,
    TabletopBoxDualArmReachingTask,
    TabletopBoxRightArmReachingTask,
    TabletopBoxTaskBase,
)

set_ompl_random_seed(1)
np.random.seed(2)

dual = True

task: TabletopBoxTaskBase
if dual:
    task = TabletopBoxDualArmReachingTask.sample(1, True)
else:
    task = TabletopBoxRightArmReachingTask.sample(1, True)

problem = task.export_problems()[0]

ompl_solcon = OMPLSolverConfig(n_max_call=10000, algorithm=Algorithm.RRTConnect, simplify=True)
ompl_sovler = OMPLSolver.init(ompl_solcon)
ompl_sovler.setup(problem)
ompl_result = ompl_sovler.solve()
assert ompl_result.traj is not None
print(ompl_result.time_elapsed)

n_wp = 30
nlp_solcon = SQPBasedSolverConfig(n_wp, motion_step_satisfaction="explicit", n_max_call=100)
nlp_solver = SQPBasedSolver.init(nlp_solcon)
nlp_solver.setup(problem)
nlp_result = nlp_solver.solve(ompl_result.traj.resample(n_wp))
# nlp_result = nlp_solver.solve()
assert nlp_result.traj is not None
print(nlp_result.time_elapsed)

static_vis = StaticTaskVisualizer(task)
static_vis.save_image("task.png")

dynamic_vis = InteractiveTaskVisualizer(task)
dynamic_vis.show()
dynamic_vis.visualize_trajectory(nlp_result.traj.resample(30))
