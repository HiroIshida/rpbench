from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Protocol, Tuple, Type, TypeVar

import numpy as np
from voxbloxpy.core import Grid, GridSDF

WorldT = TypeVar("WorldT", bound="WorldBase")
ProblemT = TypeVar("ProblemT", bound="ProblemBase")


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


class DescriptionTable:
    """Unified format of description for all problems
    both world and descriptions should be encoded into ArrayData
    """

    @dataclass
    class ArrayData:
        shape: Tuple[int, ...]
        data: np.ndarray

    table: Dict[str, ArrayData]


@dataclass
class ProblemBase(ABC, Generic[WorldT]):
    """Problem base class
    Problem is composed of world and *descriptions*

    One may wonder why *descriptions* instead of a description.
    When serialize the problem to a data, serialized world data tends to
    be very larget, though the description is light. So, if mupltiple problems
    instance share the same world, it should be handle it as a single problem
    for the memory efficiency.
    """

    world: WorldT
    descriptions: List[Any]
    _gridsdf: Optional[GridSDF]

    @classmethod
    def sample(
        cls: Type[ProblemT], n_sample: int, standard: bool = False, with_gridsdf: bool = True
    ) -> ProblemT:
        """Sample problem with a single scene with n_sample descriptions."""
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
    def sample_descriptions(world: WorldT, n_sample: int, standard: bool = False) -> List[Any]:
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


class SolverProtocol(ABC, Generic[ProblemT]):
    @classmethod
    @abstractmethod
    def get_config(cls) -> SolverConfig:
        ...

    @abstractmethod
    def solve(self, problem: ProblemT) -> List[SolverResult]:
        ...
