from dataclasses import dataclass
import numpy as np
from typing import Protocol, Generic, TypeVar, Type, Dict, Any, Tuple, List


WorldT = TypeVar("WorldT", bound="WorldProtocol")
ProblemT = TypeVar("ProblemT", bound="ProblemProtocol")


class SDFProtocol(Protocol):

    def __call__(self, __X: np.ndarray) -> np.ndarray:
        """ return signed distance corresponds to _x
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


class WorldProtocol(Protocol):

    @classmethod
    def sample(cls: Type[WorldT], standard: bool = True) -> WorldT:
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


class ProblemProtocol(Protocol, Generic[WorldT]):
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
    def sample(cls: Type[ProblemT], n_sample: int, standard: bool = True) -> ProblemT:
        """Sample problem with a single scene with n_sample descriptions. """
        ...

    def as_table(self) -> DescriptionTable:
        ...

    def get_sdf(self) -> SDFProtocol:
        ...

    def __len__(self) -> int:
        """ return number of descriptions"""
        return len(self.descriptions)


class ResultProtocol(Protocol):
    nit: int
    success: bool
    x: np.ndarray


class SolverConfigProtocol(Protocol):
    maxiter: int


class SolverProtocol(Protocol, Generic[ProblemT]):

    @classmethod
    def get_config(cls) -> SolverConfigProtocol:
        ...


    def solve(self, problem: ProblemT) -> List[ResultProtocol]:
        ...
