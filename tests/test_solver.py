from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.articulated.pr2.minifridge import TabletopClutteredFridgeReachingTask
from rpbench.interface import DatadrivenTaskSolver, PlanningDataset, SkmpTaskSolver


def test_task_sovler():
    task_type = TabletopClutteredFridgeReachingTask
    task = task_type.sample(1, standard=True)

    solvers = []

    # create rrtconnect
    rrt_connect = SkmpTaskSolver(OMPLSolver.init(OMPLSolverConfig(n_max_call=10000, n_max_satisfaction_trial=100)), task_type)  # type: ignore
    solvers.append(rrt_connect)

    # create lightning
    rrt_connect.setup(task)
    res = rrt_connect.solve()
    assert res.traj is not None
    dataset = PlanningDataset([(task, res.traj)], task_type, 0.0)

    lightning = DatadrivenTaskSolver.init(OMPLSolver, OMPLSolverConfig(), dataset)  # type: ignore
    solvers.append(lightning)  # type: ignore[arg-type]

    # create sqpbased
    sqpbased = SQPBasedSolver.init(SQPBasedSolverConfig(n_wp=20))
    solvers.append(SkmpTaskSolver.init(sqpbased, task_type))  # type: ignore

    for solver in solvers:
        solver.setup(task)
        solver.solve()
