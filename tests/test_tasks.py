from typing import Type

import numpy as np
import pytest
from ompl import set_ompl_random_seed

from rpbench.articulated.jaxon.below_table import (
    HumanoidTableClutteredReachingTask,
    HumanoidTableClutteredReachingTask2,
    HumanoidTableReachingTask,
    HumanoidTableReachingTask2,
)
from rpbench.articulated.pr2.minifridge import (
    FixedPR2MiniFridgeTask,
    MovingPR2MiniFridgeTask,
)
from rpbench.interface import TaskBase
from rpbench.two_dimensional.bubbly_world import BubblySimpleMeshPointConnectTask
from rpbench.two_dimensional.dummy import (
    DummyConfig,
    DummySolver,
    DummyTask,
    ProbDummyTask,
)

np.random.seed(0)
set_ompl_random_seed(0)


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
        FixedPR2MiniFridgeTask,
        MovingPR2MiniFridgeTask,
        HumanoidTableReachingTask2,
        HumanoidTableReachingTask,
        HumanoidTableClutteredReachingTask,
        HumanoidTableClutteredReachingTask2,
        BubblySimpleMeshPointConnectTask,
        DummyTask,
        ProbDummyTask,
    ],
)
def test_task_hash(task_type: Type[TaskBase]):
    vec1 = task_type.distribution_vector()
    for _ in range(5):
        vec2 = task_type.distribution_vector()
        # assert np.allclose(vec1, vec2)
        np.testing.assert_almost_equal(vec1, vec2, decimal=5)


@pytest.mark.parametrize(
    "task_type",
    [
        FixedPR2MiniFridgeTask,
        MovingPR2MiniFridgeTask,
        HumanoidTableReachingTask2,
        HumanoidTableReachingTask,
        HumanoidTableClutteredReachingTask,
        HumanoidTableClutteredReachingTask2,
        BubblySimpleMeshPointConnectTask,
        DummyTask,
        ProbDummyTask,
    ],
)
def test_reconstruction_from_intrinsic(task_type: Type[TaskBase]):
    task = task_type.sample(5)
    mat = task.export_task_expression(use_matrix=True).world_mat
    intr_vecs = task.to_task_params()
    task_again = task_type.from_task_params(intr_vecs)

    intr_vecs_again = task_again.to_task_params()
    mat_again = task_again.export_task_expression(use_matrix=True).world_mat
    assert np.allclose(intr_vecs, intr_vecs_again)
    if mat is not None:
        assert np.allclose(mat, mat_again)


@pytest.mark.parametrize(
    "task_type",
    [
        FixedPR2MiniFridgeTask,
        MovingPR2MiniFridgeTask,
        HumanoidTableReachingTask2,
        HumanoidTableReachingTask,
        HumanoidTableClutteredReachingTask,
        HumanoidTableClutteredReachingTask2,
        BubblySimpleMeshPointConnectTask,
        DummyTask,
        ProbDummyTask,
    ],
)
def test_default_solve(task_type: Type[TaskBase]):
    task = task_type.sample(1)
    task.solve_default()
    # we don't care if tasks are solvable or not
