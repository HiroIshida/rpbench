from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Protocol, Type, TypeVar

import numpy as np
from voxbloxpy.core import Grid, GridSDF

WorldT = TypeVar("WorldT", bound="WorldBase")
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


@dataclass
class DescriptionTable:
    """Unified format of description for all tasks
    both world and descriptions should be encoded into ArrayData
    """

    world_dict: Dict[str, np.ndarray]  # world common for all sub tasks
    desc_dicts: List[Dict[str, np.ndarray]]  # sub task wise


@dataclass
class TaskBase(ABC, Generic[WorldT, DescriptionT]):
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

    @classmethod
    def sample(
        cls: Type[TaskT], n_sample: int, standard: bool = False, with_gridsdf: bool = True
    ) -> TaskT:
        """Sample task with a single scene with n_sample descriptions."""
        world_t = cls.get_world_type()
        world = world_t.sample(standard=standard)
        descriptions = cls.sample_descriptions(world, n_sample, standard)
        if with_gridsdf:
            gridsdf = cls.create_gridsdf(world)
        else:
            gridsdf = None
        return cls(world, descriptions, gridsdf)

    def __len__(self) -> int:
        """return number of descriptions"""
        return len(self.descriptions)

    @staticmethod
    @abstractmethod
    def get_world_type() -> Type[WorldT]:
        """Typically this function is implmented by mixin"""
        ...

    @staticmethod
    @abstractmethod
    def create_gridsdf(world: WorldT) -> GridSDF:
        """Typically this function is implmented by mixin"""
        ...

    @staticmethod
    @abstractmethod
    def sample_descriptions(
        world: WorldT, n_sample: int, standard: bool = False
    ) -> List[DescriptionT]:
        """Typically this function is implmented by mixin"""
        ...

    @abstractmethod
    def as_table(self) -> DescriptionTable:
        ...


@dataclass
class SolverResult:
    nit: int
    success: bool
    x: np.ndarray

    @classmethod
    def cast_from(cls, result: Any):
        return cls(result.nit, result.success, result.x)


@dataclass
class SolverConfig:
    maxiter: int

    @classmethod
    def cast_from(cls, config: Any):
        return cls(config.maxiter)


class SolverProtocol(ABC, Generic[TaskT]):
    @classmethod
    @abstractmethod
    def get_config(cls) -> SolverConfig:
        ...

    @abstractmethod
    def solve(self, task: TaskT) -> List[SolverResult]:
        ...
