import pickle
import socket
from hashlib import md5
from typing import Any, Type

import numpy as np
import pytest
from ompl import set_ompl_random_seed
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.articulated.jaxon.below_table import (
    HumanoidTableClutteredReachingTask,
    HumanoidTableClutteredReachingTask2,
    HumanoidTableReachingTask,
    HumanoidTableReachingTask2,
)
from rpbench.articulated.pr2.jskfridge import (
    JskFridgeVerticalReachingTask,
    JskFridgeVerticalReachingTask2,
)
from rpbench.articulated.pr2.minifridge import TabletopClutteredFridgeReachingTask
from rpbench.articulated.pr2.tabletop import (
    TabletopOvenDualArmReachingTask,
    TabletopOvenDualArmReachingTaskBase,
    TabletopOvenRightArmReachingTask,
    TabletopOvenWorldWrap,
    TabletopTaskBase,
)
from rpbench.two_dimensional.bubbly_world import BubblySimpleMeshPointConnectTask
from rpbench.two_dimensional.dummy import DummyConfig, DummySolver, DummyTask

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
        TabletopOvenDualArmReachingTask,
    ],
)
def test_tabletop_task(task_type: Type[TabletopTaskBase]):
    # test standard task's consistency
    # note that, because skrobot link has uuid, we must convert it to table by
    # export_table function beforehand
    task_standard = task_type.sample(1, True)

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


def test_prob_dummy_task():
    conf = DummyConfig(10000, random=False)  # large enough
    online_solver = DummySolver.init(conf)

    for i in range(100):
        task = DummyTask.sample(1)
        sdf = task.world.get_exact_sdf()
        val = sdf(np.array(task.descriptions))[0]
        is_feasible_problem = val > 0

        task.solve_default()[0]

        while True:  # just sample guiding traj
            res = DummyTask.sample(1).solve_default()[0]
            if res.traj is not None:
                guiding_traj = res.traj
                break
        online_solver.setup(task.export_problems()[0])
        res_replan = online_solver.solve(guiding_traj)

        if is_feasible_problem:
            assert res.traj is not None
            assert res_replan is not None
        else:
            assert res.traj is None
            assert res_replan is None


@pytest.mark.parametrize(
    "task_type",
    [
        HumanoidTableReachingTask2,
        HumanoidTableReachingTask,
        HumanoidTableClutteredReachingTask,
        JskFridgeVerticalReachingTask,
        HumanoidTableClutteredReachingTask2,
        JskFridgeVerticalReachingTask2,
        TabletopClutteredFridgeReachingTask,
    ],
)
def test_task_hash(task_type: Type[TabletopTaskBase]):
    hval = task_type.compute_distribution_hash()
    for _ in range(5):
        hval2 = task_type.compute_distribution_hash()
        HumanoidTableReachingTask2,
        assert hval == hval2


def test_task_hash_value():
    # these tasks are used in TRO submission or Phd thesis.
    # if you change the task definition, you must change the hash value here.
    if socket.gethostname() != "azarashi":
        # somehow, the test fails on github actions
        pytest.skip("this test is only for azarashi (my computer)")

    assert (
        HumanoidTableReachingTask.compute_distribution_hash() == "d49ee725d0f62d9382bccfa614c73b0a"
    )
    assert (
        HumanoidTableReachingTask2.compute_distribution_hash() == "d49ee725d0f62d9382bccfa614c73b0a"
    )
    assert (
        JskFridgeVerticalReachingTask.compute_distribution_hash()
        == "6a138d57891b62785f3ae3ca02b7771d"
    )
    assert (
        TabletopClutteredFridgeReachingTask.compute_distribution_hash()
        == "fa3be78522599984748e07670907c3c7"
    )
    assert (
        BubblySimpleMeshPointConnectTask.compute_distribution_hash()
        == "50a20e5db0fc6e1140f51fc8b7e84069"
    )


# test specific tasks (covered in journal) below


def test_humanoid_table_reaching2_description():
    descs = []
    for _ in range(10):
        task = HumanoidTableReachingTask2.sample(1)
        desc = task.export_intrinsic_descriptions()[0]
        assert len(desc) == (2 + 5 + 6)
        descs.append(desc)
    descs = np.array(descs)

    descs_rpy = descs[:, -3:]
    assert np.all(descs_rpy[:, 0] == descs_rpy[0, 0])
    assert np.all(descs_rpy[:, 1] == descs_rpy[0, 1])
    assert np.all(descs_rpy[:, 2] == descs_rpy[0, 2])
