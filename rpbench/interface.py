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
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np
import threadpoolctl
import tqdm
from skmp.solver.interface import (
    AbstractDataDrivenSolver,
    AbstractSolver,
    ConfigT,
    Problem,
    ResultProtocol,
    ResultT,
)
from skmp.trajectory import Trajectory

WorldT = TypeVar("WorldT", bound="WorldBase")
SamplableT = TypeVar("SamplableT", bound="SamplableBase")
OtherSamplableT = TypeVar("OtherSamplableT", bound="SamplableBase")
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


class GridProtocol(Protocol):
    lb: np.ndarray
    ub: np.ndarray

    @property
    def sizes(self) -> Tuple[int, ...]:
        ...


class GridSDFProtocol(SDFProtocol, Protocol):
    values: np.ndarray

    @property
    def grid(self) -> GridProtocol:
        ...


class WorldBase(ABC):
    @classmethod
    @abstractmethod
    def sample(cls: Type[WorldT], standard: bool = False) -> WorldT:
        ...

    @abstractmethod
    def get_exact_sdf(self) -> SDFProtocol:
        """get an exact sdf"""
        ...

    @abstractmethod
    def get_grid(self) -> GridProtocol:
        ...

    def export_intrinsic_description(self) -> Dict:
        raise NotImplementedError()


@dataclass(frozen=True)
class DescriptionTable:
    """Unified format of description for all tasks
    both world and descriptions should be encoded into ArrayData
    """

    world_desc_dict: Dict[str, np.ndarray]  # world common for all sub tasks
    wcond_desc_dicts: List[Dict[str, np.ndarray]]  # world conditioned

    # currently we assume that world and world-conditioned descriptions follows
    # the following rule.
    # wd refere to world description and wcd refere to world-condtioned descriotion
    # - wd has only one key for 2dim or 3dim array
    # - wd may have 1 dim array
    # - wcd does not have 2dim or 3dim array
    # - wcd must have 1dim array
    # - wcd may be empty, but wd must not be empty
    # to remove the above limitation, create a task class which inherit rpbench Task
    # which is equipped with desc-2-tensor-tuple conversion rule

    def get_mesh(self) -> Optional[np.ndarray]:
        wd_ndim_to_value = {v.ndim: v for v in self.world_desc_dict.values()}
        ndim_set = wd_ndim_to_value.keys()

        contains_either_2or3_not_both = (2 in ndim_set) ^ (3 in ndim_set)
        if not contains_either_2or3_not_both:
            return None

        if 2 in ndim_set:
            mesh = wd_ndim_to_value[2]
        elif 3 in ndim_set:
            mesh = wd_ndim_to_value[3]
        else:
            assert False
        return mesh

    def get_vector_descs(self) -> List[np.ndarray]:
        wd_ndim_to_value = {v.ndim: v for v in self.world_desc_dict.values()}
        ndim_set = wd_ndim_to_value.keys()
        one_key_per_dim = len(wd_ndim_to_value) == len(self.world_desc_dict)
        assert one_key_per_dim
        wd_1dim_desc: Optional[np.ndarray] = None
        if 1 in ndim_set:
            wd_1dim_desc = wd_ndim_to_value[1]

        if len(self.wcond_desc_dicts) == 0:
            # FIXME: when wcd len == 0, wd_1dim_desc_tensor is ignored ...?
            return []
        else:
            wcd_desc_dict = self.wcond_desc_dicts[0]
            ndims = set([v.ndim for v in wcd_desc_dict.values()])
            assert ndims == {1}

            np_wcd_desc_list = []
            for wcd_desc_dict in self.wcond_desc_dicts:
                wcd_desc_vec_list = []
                if wd_1dim_desc is not None:
                    wcd_desc_vec_list.append(wd_1dim_desc)
                wcd_desc_vec_list.extend(list(wcd_desc_dict.values()))
                wcd_desc_vec_cat = np.concatenate(wcd_desc_vec_list)
                np_wcd_desc_list.append(wcd_desc_vec_cat)
            return np_wcd_desc_list


@dataclass
class SamplableBase(ABC, Generic[WorldT, DescriptionT, RobotModelT]):
    """Task base class
    Task is composed of world and *descriptions*

    One may wonder why *descriptions* instead of a description.
    When serialize the task to a data, serialized world data tends to
    be very larget, though the description is light. So, if mupltiple tasks
    instance share the same world, it should be handle it as a single task
    for the memory efficiency.
    """

    world: WorldT
    descriptions: List[DescriptionT]
    _gridsdf: Optional[GridSDFProtocol]

    @property
    def n_inner_task(self) -> int:
        return len(self.descriptions)

    @classmethod
    def sample(
        cls: Type[SamplableT], n_wcond_desc: int, standard: bool = False, with_gridsdf: bool = True
    ) -> SamplableT:
        """Sample task with a single scene with n_wcond_desc descriptions."""
        robot_model = cls.get_robot_model()
        world_t = cls.get_world_type()
        world = world_t.sample(standard=standard)
        descriptions = cls.sample_descriptions(world, n_wcond_desc, standard)
        if with_gridsdf:
            gridsdf = cls.create_gridsdf(world, robot_model)
        else:
            gridsdf = None
        return cls(world, descriptions, gridsdf)

    @classmethod
    def predicated_sample(
        cls: Type[SamplableT],
        n_wcond_desc: int,
        predicate: Callable[[SamplableT], bool],
        max_trial_per_desc: int,
        with_gridsdf: bool = True,
    ) -> Optional[SamplableT]:
        """sample task that maches the predicate function"""

        # predicated sample cannot be a standard task
        standard = False

        robot_model = cls.get_robot_model()
        world_t = cls.get_world_type()
        world = world_t.sample(standard=standard)

        if with_gridsdf:
            gridsdf = cls.create_gridsdf(world, robot_model)
        else:
            gridsdf = None

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

            desc = cls.sample_descriptions(world, 1, standard)[0]
            temp_problem = cls(world, [desc], gridsdf)

            # note that some predicate may depends on gridsdf
            # thus, you must be careful if you set with_grisdf=False
            if predicate(temp_problem):
                descriptions.append(desc)

            if count_trial_before_first_success > max_trial_per_desc:
                return None

        return cls(world, descriptions, gridsdf)

    @staticmethod
    @abstractmethod
    def get_world_type() -> Type[WorldT]:
        ...

    @staticmethod
    @abstractmethod
    def get_robot_model() -> RobotModelT:
        """get robot model set by initial joint angles
        Because loading the model everytime takes time a lot,
        we assume this function utilize some cache.
        Also, we assume that robot joint configuration for every
        call of this method is consistent.
        """
        ...

    @staticmethod
    @abstractmethod
    def create_gridsdf(world: WorldT, robot_model: RobotModelT) -> Optional[GridSDFProtocol]:
        """create gridsdf of the world given robot state
        The reason why this takes RobotModel as input is that, this method
        may involves vision-simulation using robot model (e.g. synthetic pcloud)
        """
        ...

    @classmethod
    @abstractmethod
    def sample_descriptions(
        cls, world: WorldT, n_sample: int, standard: bool = False
    ) -> List[DescriptionT]:
        ...

    @abstractmethod
    def export_table(self) -> DescriptionTable:
        ...

    def export_intrinsic_descriptions(self) -> List[Dict]:
        raise NotImplementedError()

    def __len__(self) -> int:
        """return number of descriptions"""
        return len(self.descriptions)

    @classmethod
    def cast_from(cls: Type[SamplableT], obj: OtherSamplableT) -> SamplableT:
        raise NotImplementedError()


@dataclass
class TaskBase(SamplableBase[WorldT, DescriptionT, RobotModelT]):
    def solve_default(self) -> List[ResultProtocol]:
        """solve the task by using default setting without initial solution
        This solve function is expected to successfully solve
        the problem and get smoother solution if the task is feasible.
        Thus, typically the implementation would be the combination of
        sampling-based algorithm with large sampling budget and nlp based
        smoother.

        This method is abstract, because depending on the task type
        sampling budget could be much different.
        """
        return [self.solve_default_each(p) for p in self.export_problems()]

    @abstractmethod
    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        ...

    @classmethod
    @abstractmethod
    def get_dof(cls) -> int:
        """get dof of robot in this task"""
        ...

    @abstractmethod
    def export_problems(self) -> List[Problem]:
        ...


class AbstractTaskSolver(ABC, Generic[TaskT, ResultT]):
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
class SkmpTaskSolver(AbstractTaskSolver[TaskT, ResultT]):
    """Task solver for non-datadriven solver such rrt and sqp
    this class is just a wrapper of skmp non-datadriven solver
    to fit AbstractTaskSolver interface
    """

    skmp_solver: AbstractSolver
    task_type: Type[TaskT]

    @classmethod
    def init(
        cls, skmp_solver: AbstractSolver[ConfigT, ResultT], task_type: Type[TaskT]
    ) -> "SkmpTaskSolver[TaskT, ResultT]":
        return cls(skmp_solver, task_type)

    def setup(self, task: TaskT) -> None:
        assert task.n_inner_task == 1
        prob = task.export_problems()[0]
        self.skmp_solver.setup(prob)

    def solve(self) -> ResultT:
        return self.skmp_solver.solve()


@dataclass
class PlanningDataset(Generic[TaskT]):
    pairs: List[Tuple[TaskT, Trajectory]]
    task_type: Type[TaskT]
    time_stamp: float

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

    def save(self, base_path: Path) -> None:
        uuid_str = str(uuid.uuid4())
        p = base_path / "planning-dataset-{}-{}-{}.pkl".format(
            self.task_type.__name__, uuid_str, self.time_stamp
        )

        with p.open(mode="wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, base_path: Path, task_type: Type[TaskT]) -> "PlanningDataset[TaskT]":
        """load the newest dataset found in the path"""
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


@dataclass
class DatadrivenTaskSolver(AbstractTaskSolver[TaskT, ResultT]):
    """Task solver for non-datadriven solver such rrt and sqp"""

    skmp_solver: AbstractDataDrivenSolver
    task_type: Type[TaskT]

    @classmethod
    def init(
        cls,
        skmp_dd_solver_type: Type[AbstractDataDrivenSolver[ConfigT, ResultT]],
        solver_config: ConfigT,
        dataset: PlanningDataset[TaskT],
    ) -> "DatadrivenTaskSolver[TaskT, ResultT]":

        pairs_modified = []
        for task, traj in dataset.pairs:
            pair = (task.export_problems()[0], traj)
            pairs_modified.append(pair)

        solver = skmp_dd_solver_type.init(solver_config, pairs_modified)
        return cls(solver, dataset.task_type)

    def setup(self, task: TaskT) -> None:
        assert task.n_inner_task == 1
        prob = task.export_problems()[0]
        self.skmp_solver.setup(prob)

    def solve(self) -> ResultT:
        return self.skmp_solver.solve()
