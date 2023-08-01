import pickle
from hashlib import md5
from typing import Any, Type

import numpy as np
import pytest
from ompl import set_ompl_random_seed
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.articulated.pr2.kivapod import KivapodEmptyReachingTask
from rpbench.articulated.pr2.tabletop import (
    TabletopOvenDualArmReachingTask,
    TabletopOvenDualArmReachingTaskBase,
    TabletopOvenRightArmReachingTask,
    TabletopOvenVoxbloxDualArmReachingTask,
    TabletopOvenVoxbloxRightArmReachingTask,
    TabletopOvenWorldWrap,
    TabletopTaskBase,
    VoxbloxGridSDFCreator,
)
from rpbench.two_dimensional.bubbly_world import (
    BubblyComplexMeshPointConnectTask,
    BubblyComplexPointConnectTask,
)
from rpbench.two_dimensional.dummy import DummyConfig, DummySolver, DummyTask
from rpbench.two_dimensional.maze import MazeSolvingTask

np.random.seed(0)
set_ompl_random_seed(0)


def test_intrinsic_dimension():
    for standard in [False, True]:
        task: Any = TabletopOvenWorldWrap.sample(2, standard=standard)
        int_descs = task.export_intrinsic_descriptions()
        assert len(int_descs) == 2
        assert len(int_descs[0]) == 7

        task = TabletopOvenRightArmReachingTask.sample(1)
        assert len(task.export_intrinsic_descriptions()[0]) == 13

        task = TabletopOvenDualArmReachingTask.sample(1)
        assert len(task.export_intrinsic_descriptions()[0]) == 19


def test_tabletop_samplable():
    ww = TabletopOvenWorldWrap.sample(10)
    assert ww.n_inner_task == 10
    # cast
    task = TabletopOvenRightArmReachingTask.cast_from(ww)
    assert task.n_inner_task == 0
    TabletopOvenWorldWrap.cast_from(task)


@pytest.mark.parametrize(
    "task_type",
    [
        TabletopOvenRightArmReachingTask,
        TabletopOvenVoxbloxRightArmReachingTask,
        TabletopOvenDualArmReachingTask,
        TabletopOvenVoxbloxDualArmReachingTask,
    ],
)
def test_tabletop_task(task_type: Type[TabletopTaskBase]):
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
    if isinstance(task, TabletopOvenDualArmReachingTaskBase):
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

    if isinstance(task, TabletopOvenDualArmReachingTaskBase):
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
    def predicate(task: TabletopTaskBase):
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

    # test lazy gridsdf creation
    task = task_type.sample(1, standard=True, create_cache=False)
    result = task.solve_default()[0]
    assert result.traj is not None


def test_kivapot_planning_task():
    task = KivapodEmptyReachingTask.sample(10, False)
    desc_table = task.export_table()
    assert desc_table.get_mesh() is None

    dic = desc_table.wcond_desc_dicts[0]
    assert dic["target_pose-0"].shape == (6,)

    # check if standard task can be solved
    task = KivapodEmptyReachingTask.sample(1, True)
    res = task.solve_default()[0]
    assert res.traj is not None

    # check pickle-depickle
    dumped = pickle.dumps(task)
    byte_size = len(dumped)
    assert byte_size < 8 * 10**6

    task_again: KivapodEmptyReachingTask = pickle.loads(dumped)

    # check that sdf object is not copied through pickle-depickle
    assert task_again.world.kivapod_mesh.sdf.itp is task.world.kivapod_mesh.sdf.itp

    res = task_again.solve_default()[0]
    assert res.traj is not None


def test_maze_solving_task():
    n_inner = 10
    task = MazeSolvingTask.sample(n_inner)
    desc_table = task.export_table()

    assert desc_table.get_mesh() is None

    mesh = desc_table.world_desc_dict["world"]
    assert mesh.ndim == 1
    assert len(desc_table.wcond_desc_dicts) == n_inner

    dic = desc_table.wcond_desc_dicts[0]
    start = dic["start"]
    goal = dic["goal"]
    assert len(start) == 2
    assert len(goal) == 2

    task = MazeSolvingTask.sample(1, True)
    res = task.solve_default()[0]
    assert res.traj is not None


def test_bubbly_world_point_connecting_task():

    # check solvability of standard problem
    for task_type in [BubblyComplexPointConnectTask, BubblyComplexMeshPointConnectTask]:
        task = task_type.sample(1, True)
        result = task.solve_default()[0]
        assert result.traj is not None


def test_dummy_task():
    task = DummyTask.sample(1, standard=True)
    res_off = task.solve_default()[0]
    assert res_off.traj is not None

    conf = DummyConfig(500, random=False)  # dist < 0.5
    online_solver = DummySolver.init(conf)

    task = DummyTask.sample(1, standard=False)
    task.descriptions[0] = np.array([0.49, 0.0])
    prob = task.export_problems()[0]
    online_solver.setup(prob)
    res = online_solver.solve(res_off.traj)
    assert res.traj is not None

    task.descriptions[0] = np.array([0.51, 0.0])
    prob = task.export_problems()[0]
    online_solver.setup(prob)
    res = online_solver.solve(res_off.traj)
    assert res.traj is None


if __name__ == "__main__":
    # test_tabletop_task()
    # test_maze_solving_task()
    test_kivapot_planning_task()
