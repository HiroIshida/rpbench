from abc import ABC, abstractmethod
from typing import Generic, Literal, Optional, Tuple, TypeVar

import numpy as np
from skrobot.coordinates import CascadedCoords
from skrobot.model.primitives import Box, Cylinder, Link
from skrobot.sdf import BoxSDF, CylinderSDF, SignedDistanceFunction

PrimitiveT = TypeVar("PrimitiveT", bound=Link)
SelfT = TypeVar("SlefT", bound="PrimitiveSkelton")


class PrimitiveSkelton(ABC, Generic[PrimitiveT]):
    """light weight primitive shapes
    here, light weight means that each instance
    does not contain trimesh object, rather each
    only has analytical property as atributes.
    """

    sdf: SignedDistanceFunction

    def to_visualizable(self, color: Optional[Tuple[int, int, int, int]] = None) -> PrimitiveT:
        primitive = self._to_skrobot_primitive()
        if color is not None:
            primitive.set_color(color)
        return primitive

    @abstractmethod
    def _to_skrobot_primitive(self) -> PrimitiveT:
        ...

    @abstractmethod
    def detach_clone(self: SelfT) -> SelfT:
        ...


class BoxSkeleton(CascadedCoords, PrimitiveSkelton[Box]):
    # works as Box but does not have trimesh geometries
    _extents: np.ndarray

    def __init__(self, extents, pos=None):
        CascadedCoords.__init__(self, pos=pos)
        self._extents = extents

        sdf = BoxSDF(extents)
        self.assoc(sdf, relative_coords="local")
        self.sdf = sdf

    @property
    def extents(self) -> np.ndarray:
        return np.array(self._extents)

    def _to_skrobot_primitive(self) -> Box:
        box = Box(self.extents)
        box.newcoords(self.copy_worldcoords())
        return box

    def sample_points(self, n_sample: int, wrt: Literal["world", "local"] = "world") -> np.ndarray:
        points_local = np.random.rand(n_sample, 3) * self.extents[None, :] - 0.5 * self.extents
        if wrt == "local":
            return points_local
        else:
            return self.transform_vector(points_local)

    def detach_clone(self) -> "BoxSkeleton":
        b = BoxSkeleton(self.extents)
        b.newcoords(self.copy_worldcoords())
        return b


class CylinderSkelton(CascadedCoords, PrimitiveSkelton[Cylinder]):
    radius: float
    height: float

    def __init__(self, radius, height, pos=(0, 0, 0)):
        CascadedCoords.__init__(self, pos=pos)
        self.radius = radius
        self.height = height
        sdf = CylinderSDF(height, radius)
        self.assoc(sdf, relative_coords="local")
        self.sdf = sdf

    def _to_skrobot_primitive(self) -> Cylinder:
        cylidner = Cylinder(self.radius, self.height)
        cylidner.newcoords(self.copy_worldcoords())
        return cylidner

    def detach_clone(self) -> "CylinderSkelton":
        c = CylinderSkelton(self.radius, self.height)
        c.newcoords(self.copy_worldcoords())
        return c
