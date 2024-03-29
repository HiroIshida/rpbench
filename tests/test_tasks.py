from typing import Type

import numpy as np
import pytest
from ompl import set_ompl_random_seed

from rpbench.articulated.jaxon.below_table import (
    HumanoidTableClutteredReachingTask,
    HumanoidTableClutteredReachingTask2,
    HumanoidTableReachingTask,
    HumanoidTableReachingTask2,
    HumanoidTableReachingTask3,
)
from rpbench.articulated.pr2.minifridge import FixedPR2MiniFridgeTask, PR2MiniFridgeTask
from rpbench.interface import TaskBase
from rpbench.two_dimensional.bubbly_world import BubblySimpleMeshPointConnectTask
from rpbench.two_dimensional.dummy import (
    DummyConfig,
    DummyMeshTask,
    DummySolver,
    DummyTask,
    ProbDummyTask,
)

np.random.seed(0)
set_ompl_random_seed(0)


def test_dummy_task():
    sample = DummyTask.sample()
    sample.description = np.array([0.0, 0.0])
    res_off = sample.solve_default()
    assert res_off.traj is not None

    conf = DummyConfig(500, random=False)  # dist < 0.5
    online_solver = DummySolver.init(conf)

    task = DummyTask.sample()
    task.description = np.array([0.49, 0.0])
    prob = task.export_problem()
    online_solver.setup(prob)
    res = online_solver.solve(res_off.traj)
    assert res.traj is not None

    task.description = np.array([0.51, 0.0])
    prob = task.export_problem()
    online_solver.setup(prob)
    res = online_solver.solve(res_off.traj)
    assert res.traj is None


def test_prob_dummy_task():
    conf = DummyConfig(10000, random=False)  # large enough
    online_solver = DummySolver.init(conf)

    for i in range(100):
        task = DummyTask.sample()
        sdf = task.world.get_exact_sdf()
        val = sdf(np.array(task.description))[0]
        is_feasible_problem = val > 0
        task.solve_default()

        while True:  # just sample guiding traj
            sample = DummyTask.sample()
            res = sample.solve_default()
            res = DummyTask.sample().solve_default()
            if res.traj is not None:
                guiding_traj = res.traj
                break
        online_solver.setup(task.export_problem())
        res_replan = online_solver.solve(guiding_traj)

        if is_feasible_problem:
            assert res.traj is not None
            assert res_replan is not None
        else:
            assert res.traj is None
            assert res_replan is None


task_type_list = [
    FixedPR2MiniFridgeTask,
    PR2MiniFridgeTask,
    HumanoidTableReachingTask3,
    HumanoidTableReachingTask2,
    HumanoidTableReachingTask,
    HumanoidTableClutteredReachingTask,
    HumanoidTableClutteredReachingTask2,
    BubblySimpleMeshPointConnectTask,
    DummyTask,
    DummyMeshTask,
    ProbDummyTask,
]


@pytest.mark.parametrize("task_type", task_type_list)
def test_task_hash(task_type: Type[TaskBase]):
    vec1 = task_type.distribution_vector()
    for _ in range(5):
        vec2 = task_type.distribution_vector()
        # assert np.allclose(vec1, vec2)
        np.testing.assert_almost_equal(vec1, vec2, decimal=5)


@pytest.mark.parametrize("task_type", task_type_list)
def test_sample_validity(task_type: Type[TaskBase]):
    for i in range(100):
        task = task_type.sample(timeout=5.0)
        problem = task.export_problem()
        valid, _ = problem.check_init_feasibility()
        assert valid, f"Failed to sample {task_type.__name__} at {i}-th trial"


@pytest.mark.parametrize("task_type", task_type_list)
def test_reconstruction_from_intrinsic(task_type: Type[TaskBase]):
    task = task_type.sample()
    mat = task.export_task_expression(use_matrix=True).world_mat
    param = task.to_task_param()
    task_again = task_type.from_task_param(param)

    param_again = task_again.to_task_param()
    mat_again = task_again.export_task_expression(use_matrix=True).world_mat
    assert param.shape == param_again.shape
    # assert np.allclose(param, param_again)
    if mat is not None:
        assert np.allclose(mat, mat_again)


@pytest.mark.parametrize("task_type", task_type_list)
def test_default_solve(task_type: Type[TaskBase]):
    task = task_type.sample()
    task.solve_default()
    # we don't care if tasks are solvable or not


@pytest.mark.parametrize("task_type", task_type_list)
def test_sampler_statelessness(task_type):
    np.random.seed(0)
    task = task_type.sample()
    param1a = task.to_task_param()
    param1b = task.to_task_param()

    np.random.seed(0)
    task = task_type.sample()
    param2a = task.to_task_param()
    param2b = task.to_task_param()

    assert np.allclose(param1a, param2a)
    assert np.allclose(param1b, param2b)
