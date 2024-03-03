import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from skmp.solver.interface import (
    AbstractScratchSolver,
    ConfigT,
    ParallelSolver,
    Problem,
    ResultProtocol,
    ResultT,
)
from skmp.trajectory import Trajectory

from rpbench.utils import temp_seed

WorldT = TypeVar("WorldT", bound="SamplableWorldBase")
TaskT = TypeVar("TaskT", bound="TaskBase")
WCondTaskT = TypeVar("WCondTaskT", bound="TaskWithWorldCondBase")
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


class TaskExpressionProtocol(Protocol):
    def get_matrix(self) -> Optional[np.ndarray]:
        ...

    def get_vector(self) -> np.ndarray:
        ...


@dataclass(frozen=True)
class TaskExpression:
    world_vec: Optional[np.ndarray]
    world_mat: Optional[np.ndarray]
    other_vec: np.ndarray

    def get_matrix(self) -> Optional[np.ndarray]:
        return self.world_mat

    def get_vector(self) -> np.ndarray:
        if self.world_vec is None:
            return self.other_vec
        else:
            return np.hstack([self.world_vec, self.other_vec])


class TaskBase(ABC, Generic[DescriptionT]):
    description: DescriptionT

    @classmethod
    @abstractmethod
    def from_task_param(cls: Type[TaskT], param: np.ndarray) -> "TaskT":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sample(
        cls: Type[TaskT],
        standard: bool = False,
        predicate: Optional[Callable[[TaskT], bool]] = None,
        timeout: int = 180,
    ) -> TaskT:
        raise NotImplementedError

    def to_task_param(self) -> np.ndarray:
        param = self.export_task_expression(use_matrix=False).get_vector()
        return param

    @classmethod
    def distribution_vector(cls: Type[TaskT]) -> np.ndarray:
        # express the distribution as 100 dim vector
        # by check this vector, we can check if the distribution definition has been changed
        # NOTE that this is "classmethod" because this vector is not for an instance from
        # the distribution but for the distribution itself.

        cls.sample()  # cache something
        tasks: List[TaskT] = []
        with temp_seed(0, True):
            while len(tasks) < 5:
                task = cls.sample()
                if task is not None:
                    tasks.append(task)
        data = np.array([t.to_task_param() for t in tasks])
        return data

    @classmethod
    def distribution_hash(cls: Type[TaskT]) -> str:
        # hash is more strict way to check the distribution definition
        # but easily affected by the tiny change like order of operations
        return hashlib.md5(cls.distribution_vector().tobytes()).hexdigest()

    @abstractmethod
    def export_task_expression(self, use_matrix: bool) -> TaskExpressionProtocol:
        raise NotImplementedError

    @abstractmethod
    def solve_default(self) -> ResultProtocol:
        raise NotImplementedError

    @abstractmethod
    def export_problem(self) -> Problem:
        raise NotImplementedError


class SamplableWorldBase(ABC):
    @classmethod
    @abstractmethod
    def sample(cls: Type[WorldT], standard: bool = False) -> Optional[WorldT]:
        ...


class TaskWithWorldCondBase(TaskBase[DescriptionT], Generic[WorldT, DescriptionT, RobotModelT]):
    world: WorldT

    def __init__(self, world: WorldT, description: DescriptionT) -> None:
        self.world = world
        self.description = description

    @staticmethod
    def get_world_type() -> Type[WorldT]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sample_description(cls, world: WorldT, standard: bool = False) -> Optional[DescriptionT]:
        raise NotImplementedError

    @classmethod
    def sample(
        cls: Type[WCondTaskT],
        standard: bool = False,
        predicate: Optional[Callable[[WCondTaskT], bool]] = None,
        timeout: int = 180,
    ) -> WCondTaskT:

        cls.get_robot_model()  # to create cache of robot model (do we really need this?)
        world_t = cls.get_world_type()

        t_start = time.time()
        while True:
            t_elapsed = time.time() - t_start
            if t_elapsed > timeout:
                raise TimeoutError("predicated_sample: timeout!")

            world = world_t.sample(standard=standard)
            if world is not None:
                description = cls.sample_description(world, standard)
                if description is not None:
                    task = cls(world, description)
                    if predicate is None or predicate(task):
                        return task

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
        prob = task.export_problem()
        self.skmp_solver.setup(prob)

    def solve(self) -> ResultT:
        return self.skmp_solver.solve()
