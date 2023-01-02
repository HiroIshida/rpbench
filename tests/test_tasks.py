import numpy as np
from ompl import set_ompl_random_seed
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.tabletop import TabletopBoxRightArmReachingTask

np.random.seed(0)
set_ompl_random_seed(0)


def test_tabletop_task():
    n_desc = 10
    task = TabletopBoxRightArmReachingTask.sample(n_desc)

    # test conversion to numpy format
    desc_table = task.as_table()
    assert desc_table.world_dict["world"].ndim == 3
    assert desc_table.world_dict["table_pose"].shape == (6,)

    assert len(desc_table.desc_dicts) == n_desc
    desc_dict = desc_table.desc_dicts[0]
    assert desc_dict["target_pose-0"].shape == (6,)

    # test conversion to problem format
    raw_problems = task.export_problems()
    assert len(raw_problems) == n_desc

    # test if standard problem can be solved by rrt-connect
    task = TabletopBoxRightArmReachingTask.sample(1, standard=True)
    raw_problems = task.export_problems()
    raw_problem = raw_problems[0]
    solcon = OMPLSolverConfig(n_max_eval=100000)
    solver = OMPLSolver.setup(raw_problem, solcon)
    res = solver.solve(None)
    assert res.traj is not None


if __name__ == "__main__":
    test_tabletop_task()
