from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Protocol, Tuple, Type, TypeVar

import numpy as np

WorldT = TypeVar("WorldT", bound="WorldProtocol")
ProblemT = TypeVar("ProblemT", bound="ProblemProtocol")


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


class WorldProtocol(ABC):
    @classmethod
    @abstractmethod
    def sample(cls: Type[WorldT], standard: bool = True) -> WorldT:
        ...

    @abstractmethod
    def get_exact_sdf(self) -> SDFProtocol:
        """get an exact sdf"""
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


class ProblemProtocol(ABC, Generic[WorldT]):
    """Problem Protocol
    Problem is composed of world and *descriptions*

    One may wonder why *descriptions* instead of a description.
    When serialize the problem to a data, serialized world data tends to
    be very larget, though the description is light. So, if mupltiple problems
    instance share the same world, it should be handle it as a single problem
    for the memory efficiency.
    """

    world: WorldT
    descriptions: List[Any]

    @classmethod
    @abstractmethod
    def sample(cls: Type[ProblemT], n_sample: int, standard: bool = True) -> ProblemT:
        """Sample problem with a single scene with n_sample descriptions."""
        ...

    @abstractmethod
    def as_table(self) -> DescriptionTable:
        ...

    @abstractmethod
    def get_sdf(self) -> SDFProtocol:
        """get sdf which will be used for planning

        Note that the sdf could be different from world's exact sdf. For example, sdf
        generated by ray-tracing and voxblox could be differnt from the exact one.
        """
        ...

    def __len__(self) -> int:
        """return number of descriptions"""
        return len(self.descriptions)


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
