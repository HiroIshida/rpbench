import pickle
from hashlib import md5

import numpy as np
from ompl import set_ompl_random_seed
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.tabletop import TabletopBoxRightArmReachingTask

np.random.seed(0)
set_ompl_random_seed(0)


def test_tabletop_task():
    # test standard task's consistency
    # note that, because skrobot link has uuid, we must convert it to table by
    # export_table function beforehand
    task_standard = TabletopBoxRightArmReachingTask.sample(1, True)
    table = task_standard.export_table()
    value = md5(pickle.dumps(table)).hexdigest()
    for _ in range(5):
        task_standard = TabletopBoxRightArmReachingTask.sample(1, True)
        table = task_standard.export_table()
        value_test = md5(pickle.dumps(table)).hexdigest()
        assert value == value_test

    n_desc = 10
    task = TabletopBoxRightArmReachingTask.sample(n_desc)
    assert task.n_inner_task == n_desc

    # test dof
    assert task.get_dof() == 7

    # test conversion to numpy format
    desc_table = task.export_table()
    assert desc_table.world_desc_dict["world"].ndim == 3
    assert desc_table.world_desc_dict["table_pose"].shape == (6,)

    assert len(desc_table.wcond_desc_dicts) == n_desc
    desc_dict = desc_table.wcond_desc_dicts[0]
    assert desc_dict["target_pose-0"].shape == (6,)

    # test conversion to problem format
    raw_problems = task.export_problems()
    assert len(raw_problems) == n_desc

    # test if standard problem can be solved by rrt-connect
    task = TabletopBoxRightArmReachingTask.sample(1, standard=True)
    raw_problems = task.export_problems()
    raw_problem = raw_problems[0]
    solcon = OMPLSolverConfig(n_max_call=100000)
    solver = OMPLSolver.setup(raw_problem, solcon)
    res = solver.solve(None)
    assert res.traj is not None

    # test predicated sampling
    def predicate(task: TabletopBoxRightArmReachingTask):
        assert len(task.descriptions) == 1
        for desc in task.descriptions:
            pose = desc[0]
            pos = pose.worldpos()
            if pos[2] > 0.85:
                return False
        return True

    predicated_task = TabletopBoxRightArmReachingTask.predicated_sample(n_desc, predicate, 100)
    assert predicated_task is not None
    assert len(predicated_task.descriptions) == n_desc

    # test solve_default
    task = TabletopBoxRightArmReachingTask.sample(1, standard=True)
    result = task.solve_default()[0]
    assert result.traj is not None


if __name__ == "__main__":
    test_tabletop_task()
