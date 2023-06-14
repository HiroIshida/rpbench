from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLDataDrivenSolver, OMPLSolver, OMPLSolverConfig

from rpbench.interface import DatadrivenTaskSolver, PlanningDataset, SkmpTaskSolver
from rpbench.pr2.tabletop import TabletopOvenRightArmReachingTask


def test_task_sovler():
    task_type = TabletopOvenRightArmReachingTask
    task = task_type.sample(1, standard=True)

    solvers = []

    # create rrtconnect
    rrt_connect = SkmpTaskSolver(OMPLSolver.init(OMPLSolverConfig(n_max_call=10000)), task_type)  # type: ignore
    solvers.append(rrt_connect)

    # create lightning
    rrt_connect.setup(task)
    res = rrt_connect.solve()
    assert res.traj is not None
    dataset = PlanningDataset([(task, res.traj)], task_type, 0.0)

    lightning = DatadrivenTaskSolver.init(OMPLDataDrivenSolver, OMPLSolverConfig(), dataset)  # type: ignore
    solvers.append(lightning)  # type: ignore[arg-type]

    # create sqpbased
    sqpbased = SQPBasedSolver.init(SQPBasedSolverConfig(n_wp=20))
    solvers.append(SkmpTaskSolver.init(sqpbased, task_type))  # type: ignore

    for solver in solvers:
        solver.setup(task)
        solver.solve()
