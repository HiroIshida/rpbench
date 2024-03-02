import hashlib
import multiprocessing
import os
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import threadpoolctl
import tqdm
from skmp.solver.interface import (
    AbstractScratchSolver,
    ConfigT,
    NearestNeigborSolver,
    ParallelSolver,
    Problem,
    ResultProtocol,
    ResultT,
)
from skmp.trajectory import Trajectory

from rpbench.utils import temp_seed

WorldT = TypeVar("WorldT", bound="WorldBase")
TaskT = TypeVar("TaskT", bound="TaskBase")
DescriptionT = TypeVar("DescriptionT", bound=Any)
RobotModelT = TypeVar("RobotModelT", bound=Any)


class SDFProtocol(Protocol):
    def __call__(self, __X: np.ndarray) -> np.ndarray:
        """return signed distance corresponds to _x
        Parameters
        ----------
        __x: np.ndarray[float, 2]
            2dim (n_point, n_dim) array of points

        Returns
        ----------
        sd: np.ndarray[float, 1]
            1dim (n_point) array of signed distances of each points
        """
        ...


class WorldBase(ABC):
    @classmethod
    @abstractmethod
    def sample(cls: Type[WorldT], standard: bool = False) -> Optional[WorldT]:
        ...

    @abstractmethod
    def get_exact_sdf(self) -> SDFProtocol:
        """get an exact sdf"""
        ...


@dataclass(frozen=True)
class TaskExpression:
    world_vec: Optional[np.ndarray]
    world_mat: Optional[np.ndarray]
    other_vecs: List[np.ndarray]  # world-conditioned description

    def get_desc_vecs(self) -> List[np.ndarray]:
        if self.world_vec is None:
            return self.other_vecs
        else:
            return [np.hstack([self.world_vec, desc]) for desc in self.other_vecs]


@dataclass
class TaskBase(ABC, Generic[WorldT, DescriptionT, RobotModelT]):
    world: WorldT
    descriptions: List[DescriptionT]

    @property
    def n_inner_task(self) -> int:
        return len(self.descriptions)

    @classmethod
    def from_task_params(cls: Type[TaskT], params: np.ndarray) -> "TaskT":
        raise NotImplementedError()

    def to_task_params(self) -> np.ndarray:
        return np.array(self.export_task_expression(use_matrix=False).get_desc_vecs())

    @classmethod
    def sample(
        cls: Type[TaskT],
        n_wcond_desc: int,
        standard: bool = False,
        timeout: float = 180.0,
    ) -> TaskT:
        """Sample task with a single scene with n_wcond_desc descriptions."""
        cls.get_robot_model()  # to create cache of robot model (we really need this?)
        world_t = cls.get_world_type()

        t_start = time.time()
        while True:
            t_elapsed = time.time() - t_start
            if t_elapsed > timeout:
                assert False, "sample: timeout! after {} sec".format(timeout)

            world = world_t.sample(standard=standard)
            if world is None:
                continue

            descriptions = cls.sample_descriptions(world, n_wcond_desc, standard)
            if descriptions is None:
                continue
            return cls(world, descriptions)

    @classmethod
    def predicated_sample(
        cls: Type[TaskT],
        n_wcond_desc: int,
        predicate: Callable[[TaskT], bool],
        max_trial_per_desc: int,
        timeout: int = 180,
    ) -> Optional[TaskT]:
        """sample task that maches the predicate function"""

        # predicated sample cannot be a standard task
        standard = False

        cls.get_robot_model()  # to create cache of robot model (we really need this?)
        world_t = cls.get_world_type()

        t_start = time.time()
        while True:
            t_elapsed = time.time() - t_start
            if t_elapsed > timeout:
                print("predicated_sample: timeout!")
                return None

            world = world_t.sample(standard=standard)
            if world is None:
                continue

            # do some bit tricky thing.
            # Naively, we can sample task with multiple description and then check if
            # it satisfies the predicate. However, by this method, as the number of
            # sample descriptions increase, satisfying the predicate becomes exponentially
            # difficult. Thus, we sample the description one by one, and then concatanate
            # and marge into a task.
            descriptions: List[DescriptionT] = []
            count_trial_before_first_success = 0
            while len(descriptions) < n_wcond_desc:

                if len(descriptions) == 0:
                    count_trial_before_first_success += 1

                descs = cls.sample_descriptions(world, 1, standard)

                if descs is not None:
                    desc = descs[0]
                    temp_problem = cls(world, [desc])
                    if predicate(temp_problem):
                        descriptions.append(desc)

                if count_trial_before_first_success > max_trial_per_desc:
                    return None

            return cls(world, descriptions)

    @classmethod
    def distribution_vector(cls: Type[TaskT]) -> np.ndarray:
        # express the distribution as 100 dim vector
        # by check this vector, we can check if the distribution definition has been changed
        # NOTE that this is "classmethod" because this vector is not for an instance from
        # the distribution but for the distribution itself.

        cls.sample(10, False)  # this line somehow affects the result of the following line
        # actually we don't cache any thing by sample() procedure, so this line is not necessary
        # python's bug?

        with temp_seed(0, True):
            data = np.array([cls.sample(10, False).to_task_params() for _ in range(10)]).flatten()
        return data

    @classmethod
    def distribution_hash(cls: Type[TaskT]) -> str:
        # hash is more strict way to check the distribution definition
        # but easily affected by the tiny change like order of operations
        return hashlib.md5(cls.distribution_vector().tobytes()).hexdigest()

    def solve_default(self) -> List[ResultProtocol]:
        return [self.solve_default_each(p) for p in self.export_problems()]

    # please implement the following methods
    @staticmethod
    @abstractmethod
    def get_world_type() -> Type[WorldT]:
        ...

    @classmethod
    @abstractmethod
    def get_robot_model(cls) -> RobotModelT:
        """get robot model set by initial joint angles
        Because loading the model everytime takes time a lot,
        we assume this function utilize some cache.
        Also, we assume that robot joint configuration for every
        call of this method is consistent.
        """
        ...

    @classmethod
    @abstractmethod
    def sample_descriptions(
        cls, world: WorldT, n_sample: int, standard: bool = False
    ) -> Optional[List[DescriptionT]]:
        ...

    @abstractmethod
    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        ...

    @abstractmethod
    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        ...

    @abstractmethod
    def export_problems(self) -> List[Problem]:
        ...


class AbstractTaskSolver(ABC, Generic[TaskT, ConfigT, ResultT]):
    """TaskSolver interface

    Unlike AbstractSolver in skmp, this solver is task-specific.
    Of corse, non-task-specific solver such as those in skmp can be
    used as task-specific solver. See SkmpTaskSolver for the detail.
    """

    task_type: Type[TaskT]

    @abstractmethod
    def setup(self, task: TaskT) -> None:
        """setup solver for a paticular task"""
        ...

    @abstractmethod
    def solve(self) -> ResultT:
        """solve problem

        NOTE: unlike AbstractSolver, this function does not
        take init solution
        """
        ...


@dataclass
class SkmpTaskSolver(AbstractTaskSolver[TaskT, ConfigT, ResultT]):
    """Task solver for non-datadriven solver such rrt and sqp
    this class is just a wrapper of skmp non-datadriven solver
    to fit AbstractTaskSolver interface
    """

    skmp_solver: Union[
        AbstractScratchSolver[ConfigT, ResultT], ParallelSolver[ConfigT, ResultT, Trajectory]
    ]
    task_type: Type[TaskT]

    @classmethod
    def init(
        cls,
        skmp_solver: Union[
            AbstractScratchSolver[ConfigT, ResultT], ParallelSolver[ConfigT, ResultT, Trajectory]
        ],
        task_type: Type[TaskT],
    ) -> "SkmpTaskSolver[TaskT, ConfigT, ResultT]":
        return cls(skmp_solver, task_type)

    def setup(self, task: TaskT) -> None:
        assert task.n_inner_task == 1
        probs = [p for p in task.export_problems()]
        prob = probs[0]
        self.skmp_solver.setup(prob)

    def solve(self) -> ResultT:
        return self.skmp_solver.solve()


@dataclass
class PlanningDataset(Generic[TaskT]):
    pairs: List[Tuple[TaskT, Trajectory]]
    task_type: Type[TaskT]
    time_stamp: float

    def resample_trajectory(self, n_wp: int) -> None:
        for i, (task, traj) in enumerate(self.pairs):
            self.pairs[i] = (task, traj.resample(n_wp))

    @staticmethod
    def create_inner(
        task_type: Type[TaskT], n_data: int, q: multiprocessing.Queue, with_bar: bool
    ) -> None:
        # this function is static for the use in multiprocessing
        unique_id = (uuid.getnode() + os.getpid()) % (2**32 - 1)
        np.random.seed(unique_id)

        count = 0
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            with tqdm.tqdm(total=n_data, disable=not with_bar) as pbar:
                while count < n_data:
                    task = task_type.sample(1, False)
                    res = task.solve_default()[0]
                    success = res.traj is not None
                    if success:
                        assert res.traj is not None
                        q.put((task, res.traj))
                        pbar.update(1)
                        count += 1

    @classmethod
    def create(cls, task_type: Type[TaskT], n_data: int, m_process: int = 1) -> "PlanningDataset":
        def split_number(num, div):
            return [num // div + (1 if x < num % div else 0) for x in range(div)]

        process_list = []
        n_data_list = split_number(n_data, m_process)
        q = multiprocessing.Queue()  # type: ignore
        for i, n_data_split in enumerate(n_data_list):
            args = (task_type, n_data_split, q, i == 0)
            p = multiprocessing.Process(target=cls.create_inner, args=args)
            p.start()
            process_list.append(p)

        pairs: List[Tuple[TaskT, Trajectory]] = [q.get() for _ in range(n_data)]

        for p in process_list:
            p.join()

        return cls(pairs, task_type, time.time())

    @staticmethod
    def default_base_path() -> Path:
        base_path = Path("~/.rpbench/dataset").expanduser()
        base_path.mkdir(exist_ok=True, parents=True)
        return base_path

    def save(self, base_path: Optional[Path] = None) -> None:
        if base_path is None:
            base_path = self.default_base_path()

        uuid_str = str(uuid.uuid4())
        p = base_path / "planning-dataset-{}-{}-{}.pkl".format(
            self.task_type.__name__, uuid_str, self.time_stamp
        )

        with p.open(mode="wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls, task_type: Type[TaskT], base_path: Optional[Path] = None
    ) -> "PlanningDataset[TaskT]":
        """load the newest dataset found in the path"""

        if base_path is None:
            base_path = cls.default_base_path()

        dataset_list: List[PlanningDataset] = []
        for p in base_path.iterdir():

            if not p.name.startswith("planning-dataset"):
                continue

            if task_type.__name__ not in p.name:
                continue

            with p.open(mode="rb") as f:
                dataset_list.append(pickle.load(f))

        dirs_sorted = sorted(dataset_list, key=lambda x: x.time_stamp)
        return dirs_sorted[-1]


class NotEnoughDataException(Exception):
    pass


@dataclass
class DatadrivenTaskSolver(AbstractTaskSolver[TaskT, ConfigT, ResultT]):
    skmp_solver: NearestNeigborSolver
    query_desc: Optional[np.ndarray]
    task_type: Type[TaskT]

    @classmethod
    def init(
        cls,
        skmp_solver_type: Type[AbstractScratchSolver[ConfigT, ResultT]],
        solver_config: ConfigT,
        dataset: PlanningDataset[TaskT],
        n_data_use: Optional[int] = None,
        knn: int = 1,
    ) -> "DatadrivenTaskSolver[TaskT, ConfigT, ResultT]":

        if n_data_use is None:
            n_data_use = len(dataset.pairs)
        if n_data_use > len(dataset.pairs):
            message = "request: {}, available {}".format(n_data_use, len(dataset.pairs))
            raise NotEnoughDataException(message)

        pairs_modified: List[Tuple[np.ndarray, Optional[Trajectory]]] = []
        dim_desc = None
        for i in tqdm.tqdm(range(n_data_use)):
            task, traj = dataset.pairs[i]
            # FIXME: use_matrix is just for now
            assert False, "not tested yet"
            desc = task.export_task_expression(use_matrix=False).get_desc_vecs()[0]
            assert desc.ndim == 1
            pair = (desc, traj)
            pairs_modified.append(pair)
        print("dim desc: {}".format(dim_desc))
        solver = NearestNeigborSolver.init(skmp_solver_type, solver_config, pairs_modified, knn)
        return cls(solver, None, dataset.task_type)

    def setup(self, task: TaskT) -> None:
        assert task.n_inner_task == 1
        probs = [p for p in task.export_problems()]
        prob = probs[0]
        self.skmp_solver.setup(prob)
        # FIXME: use_matrix is just for now
        self.query_desc = task.export_task_expression(use_matrix=False).get_desc_vecs()[0]
        assert self.query_desc.ndim == 1

    def solve(self) -> ResultT:
        assert self.query_desc is not None
        result = self.skmp_solver.solve(self.query_desc)
        self.query_desc = None
        return result

    @property
    def previous_est_positive(self) -> Optional[bool]:
        return self.skmp_solver.previous_est_positive

    @property
    def previous_false_positive(self) -> Optional[bool]:
        return self.skmp_solver.previous_false_positive
