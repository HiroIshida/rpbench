import pickle
from hashlib import md5
from typing import Type

import numpy as np
import pytest
from ompl import set_ompl_random_seed
from skmp.constraint import SELCOL_FOUND
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.tabletop import (
    TabletopBoxDualArmReachingTask,
    TabletopBoxDualArmReachingTaskBase,
    TabletopBoxRightArmReachingTask,
    TabletopBoxTaskBase,
    TabletopBoxVoxbloxDualArmReachingTask,
    TabletopBoxVoxbloxRightArmReachingTask,
    TabletopBoxWorldWrap,
    VoxbloxGridSDFCreator,
)

np.random.seed(0)
set_ompl_random_seed(0)


def test_tabletop_samplable():
    ww = TabletopBoxWorldWrap.sample(10)
    assert ww.n_inner_task == 10
    # cast
    task = TabletopBoxRightArmReachingTask.cast_from(ww)
    assert task.n_inner_task == 0
    TabletopBoxWorldWrap.cast_from(task)


@pytest.mark.parametrize(
    "task_type",
    [
        TabletopBoxRightArmReachingTask,
        TabletopBoxVoxbloxRightArmReachingTask,
        TabletopBoxDualArmReachingTask,
        TabletopBoxVoxbloxDualArmReachingTask,
    ],
)
def test_tabletop_task(task_type: Type[TabletopBoxTaskBase]):

    if issubclass(task_type, TabletopBoxDualArmReachingTaskBase):
        if not SELCOL_FOUND:
            return

    # test standard task's consistency
    # note that, because skrobot link has uuid, we must convert it to table by
    # export_table function beforehand
    task_standard = task_type.sample(1, True)

    # NOTE: voxblox sdf generation is not deterministic so ignore
    if not issubclass(task_type, VoxbloxGridSDFCreator):
        table = task_standard.export_table()
        value = md5(pickle.dumps(table)).hexdigest()
        for _ in range(5):
            task_standard = task_type.sample(1, True)
            table = task_standard.export_table()
            value_test = md5(pickle.dumps(table)).hexdigest()
            assert value == value_test

    n_desc = 10
    task = task_type.sample(n_desc)
    assert task.n_inner_task == n_desc

    # test dof
    if isinstance(task, TabletopBoxDualArmReachingTaskBase):
        assert task.get_dof() == 17
    else:
        assert task.get_dof() == 10

    # test conversion to numpy format
    desc_table = task.export_table()
    assert desc_table.world_desc_dict["world"].ndim == 3
    assert desc_table.world_desc_dict["table_pose"].shape == (6,)

    assert len(desc_table.wcond_desc_dicts) == n_desc
    desc_dict = desc_table.wcond_desc_dicts[0]
    assert desc_dict["target_pose-0"].shape == (6,)

    if isinstance(task, TabletopBoxDualArmReachingTaskBase):
        assert desc_dict["target_pose-1"].shape == (6,)

    # test conversion to problem format
    raw_problems = task.export_problems()
    assert len(raw_problems) == n_desc

    # test if standard problem can be solved by rrt-connect
    task = task_type.sample(1, standard=True)
    raw_problems = task.export_problems()
    raw_problem = raw_problems[0]
    solcon = OMPLSolverConfig(n_max_call=100000)
    solver = OMPLSolver.init(solcon)
    solver.setup(raw_problem)
    res = solver.solve(None)
    assert res.traj is not None

    # test predicated sampling
    def predicate(task: TabletopBoxTaskBase):
        assert len(task.descriptions) == 1
        for desc in task.descriptions:
            pose = desc[0]
            pos = pose.worldpos()
            if pos[2] > 0.85:
                return False
        return True

    predicated_task = task_type.predicated_sample(n_desc, predicate, 100)
    assert predicated_task is not None
    assert len(predicated_task.descriptions) == n_desc

    # test solve_default
    task = task_type.sample(1, standard=True)
    result = task.solve_default()[0]
    assert result.traj is not None


if __name__ == "__main__":
    test_tabletop_task()
