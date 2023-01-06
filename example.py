import numpy as np
from ompl import Algorithm, set_ompl_random_seed
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.tabletop import TabletopBoxRightArmReachingTask, TaskVisualizer

set_ompl_random_seed(0)
np.random.seed(0)

task = TabletopBoxRightArmReachingTask.sample(1, True)
problem = task.export_problems()[0]

ompl_solcon = OMPLSolverConfig(n_max_call=100000, algorithm=Algorithm.RRTConnect)
ompl_sovler = OMPLSolver.setup(problem, ompl_solcon)
ompl_result = ompl_sovler.solve()
assert ompl_result.traj is not None
print(ompl_result.time_elapsed)

n_wp = 40
nlp_solcon = SQPBasedSolverConfig(n_wp, motion_step_satisfaction="explicit")
nlp_solver = SQPBasedSolver.setup(problem, nlp_solcon)
nlp_result = nlp_solver.solve(ompl_result.traj.resample(n_wp))
assert nlp_result.traj is not None
print(nlp_result.time_elapsed)

vis = TaskVisualizer(task)
vis.show()
vis.visualize_trajectory(nlp_result.traj.resample(n_wp))
