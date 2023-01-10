from typing import List

from ompl import LightningDB
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import LightningSolver, OMPLSolver, OMPLSolverConfig

from rpbench.interface import SkmpTaskSolver
from rpbench.tabletop import TabletopBoxRightArmReachingTask


def test_task_sovler():
    task_type = TabletopBoxRightArmReachingTask
    task = task_type.sample(1, standard=True)

    solvers: List[SkmpTaskSolver] = []

    rrt_connect = OMPLSolver.init(OMPLSolverConfig(n_max_call=10000))
    solvers.append(SkmpTaskSolver(rrt_connect, task_type))

    db = LightningDB(task_type.get_dof())
    lightning = LightningSolver.init(OMPLSolverConfig(n_max_call=10000), db)
    solvers.append(SkmpTaskSolver(lightning, task_type))

    sqpbased = SQPBasedSolver.init(SQPBasedSolverConfig(n_wp=20))
    solvers.append(SkmpTaskSolver.init(sqpbased, task_type))

    for solver in solvers:
        solver.setup(task)
        solver.solve()
