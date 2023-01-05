from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Type, TypeVar

import numpy as np
from skmp.solver.interface import Problem, ResultProtocol
from voxbloxpy.core import Grid, GridSDF

WorldT = TypeVar("WorldT", bound="WorldBase")
SamplableT = TypeVar("SamplableT", bound="SamplableBase")
TaskT = TypeVar("TaskT", bound="TaskBase")
DescriptionT = TypeVar("DescriptionT", bound=Any)


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
    def sample(cls: Type[WorldT], standard: bool = False) -> WorldT:
        ...

    @abstractmethod
    def get_exact_sdf(self) -> SDFProtocol:
        """get an exact sdf"""
        ...

    @abstractmethod
    def get_grid(self) -> Grid:
        ...


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
    # - wd must have either 2dim or 3dim array and not both
    # - wd has only one key for 2dim or 3dim array
    # - wd may have 1 dim array
    # - wcd does not have 2dim or 3dim array
    # - wcd must have 1dim array
    # - wcd may be empty, but wd must not be empty
    # to remove the above limitation, create a task class which inherit rpbench Task
    # which is equipped with desc-2-tensor-tuple conversion rule

    def get_mesh(self) -> np.ndarray:
        wd_ndim_to_value = {v.ndim: v for v in self.world_desc_dict.values()}
        ndim_set = wd_ndim_to_value.keys()

        contains_either_2or3_not_both = (2 in ndim_set) ^ (3 in ndim_set)
        assert contains_either_2or3_not_both
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
class SamplableBase(ABC, Generic[WorldT, DescriptionT]):
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
    _gridsdf: Optional[GridSDF]

    @property
    def n_inner_task(self) -> int:
        return len(self.descriptions)

    @classmethod
    def sample(
        cls: Type[SamplableT], n_wcond_desc: int, standard: bool = False, with_gridsdf: bool = True
    ) -> SamplableT:
        """Sample task with a single scene with n_wcond_desc descriptions."""
        world_t = cls.get_world_type()
        world = world_t.sample(standard=standard)
        descriptions = cls.sample_descriptions(world, n_wcond_desc, standard)
        if with_gridsdf:
            gridsdf = cls.create_gridsdf(world)
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

        world_t = cls.get_world_type()
        world = world_t.sample(standard=standard)

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
            temp_problem = cls(world, [desc], None)

            if predicate(temp_problem):
                descriptions.append(desc)

            if count_trial_before_first_success > max_trial_per_desc:
                return None

        if with_gridsdf:
            gridsdf = cls.create_gridsdf(world)
        else:
            gridsdf = None
        return cls(world, descriptions, gridsdf)

    @staticmethod
    @abstractmethod
    def get_world_type() -> Type[WorldT]:
        ...

    @staticmethod
    @abstractmethod
    def create_gridsdf(world: WorldT) -> GridSDF:
        ...

    @staticmethod
    @abstractmethod
    def sample_descriptions(
        world: WorldT, n_sample: int, standard: bool = False
    ) -> List[DescriptionT]:
        ...

    @abstractmethod
    def export_table(self) -> DescriptionTable:
        ...

    def __len__(self) -> int:
        """return number of descriptions"""
        return len(self.descriptions)


@dataclass
class TaskBase(SamplableBase[WorldT, DescriptionT]):
    def solve_default(self) -> List[ResultProtocol]:
        """solve the task by using default setting without initial solution
        This solve function is expected to successfully solve
        the problem if it is feasible. Thus, typically a sampling-based
        algorithm with large sampling budget would be used. Contrary,
        nlp-based algrithm, which depends heavily on init solution
        should be avoided.

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
