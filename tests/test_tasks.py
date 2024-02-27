import hashlib
import pickle
import socket
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
from rpbench.articulated.pr2.minifridge import TabletopClutteredFridgeReachingTask
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
        HumanoidTableReachingTask2,
        HumanoidTableReachingTask,
        HumanoidTableClutteredReachingTask,
        HumanoidTableClutteredReachingTask2,
        TabletopClutteredFridgeReachingTask,
        BubblySimpleMeshPointConnectTask,
        DummyTask,
        ProbDummyTask,
    ],
)
def test_task_hash(task_type: Type[TaskBase]):
    hval = task_type.compute_distribution_hash()
    for _ in range(5):
        hval2 = task_type.compute_distribution_hash()
        HumanoidTableReachingTask2,
        assert hval == hval2


@pytest.mark.parametrize(
    "task_type",
    [
        HumanoidTableReachingTask2,
        HumanoidTableReachingTask,
        HumanoidTableClutteredReachingTask,
        HumanoidTableClutteredReachingTask2,
        TabletopClutteredFridgeReachingTask,
        BubblySimpleMeshPointConnectTask,
        DummyTask,
        ProbDummyTask,
    ],
)
def test_standard(task_type: Type[TaskBase]):
    # consistency
    hash_value = hashlib.md5(pickle.dumps(task_type.sample(1, standard=True))).hexdigest()
    for _ in range(5):
        task = task_type.sample(1, standard=True)
        assert hash_value == hashlib.md5(pickle.dumps(task)).hexdigest()

    # solvability
    task = task_type.sample(1, standard=True)
    res = task.solve_default()[0]
    assert res.traj is not None


@pytest.mark.parametrize(
    "task_type",
    [
        BubblySimpleMeshPointConnectTask,
        DummyTask,
        ProbDummyTask,
    ],
)
def test_reconstruction_from_intrinsic(task_type: Type[TaskBase]):
    task = task_type.sample(5)
    intr_vecs = task.to_task_params()
    intr_vecs_again = task_type.from_task_params(intr_vecs).to_task_params()
    assert np.allclose(intr_vecs, intr_vecs_again)


def _test_task_hash_value():
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
    # assert (
    #     JskFridgeVerticalReachingTask.compute_distribution_hash()
    #     == "6a138d57891b62785f3ae3ca02b7771d"
    # )
    assert (
        TabletopClutteredFridgeReachingTask.compute_distribution_hash()
        == "fa3be78522599984748e07670907c3c7"
    )
    assert (
        BubblySimpleMeshPointConnectTask.compute_distribution_hash()
        == "50a20e5db0fc6e1140f51fc8b7e84069"
    )


def test_vector_descriptions():
    test_table = {
        HumanoidTableReachingTask: ((2 + 5 + 4), False),
        HumanoidTableReachingTask2: ((2 + 5 + 3), False),
        HumanoidTableClutteredReachingTask: ((2 + 4), True),
        HumanoidTableClutteredReachingTask2: ((2 + 3), True),
    }

    for task_type, (desc_dim, has_mesh) in test_table.items():
        descs = []
        for _ in range(10):
            task = task_type.sample(1)
            table = task.export_table(use_matrix=True)
            desc = table.get_desc_vecs()[0]
            assert len(desc) == desc_dim
            descs.append(desc)

            if has_mesh:
                assert table.world_mat is not None
            else:
                assert table.world_mat is None

        descs = np.array(descs)

    # check that all descs are different
    assert len(np.unique(descs, axis=0)) == len(descs)
