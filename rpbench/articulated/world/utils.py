from abc import ABC, abstractmethod
from typing import Generic, Literal, Optional, Tuple, TypeVar

import numpy as np
from skrobot.coordinates import CascadedCoords
from skrobot.model.primitives import Box, Cylinder
from skrobot.sdf import BoxSDF, CylinderSDF, SignedDistanceFunction

PrimitiveT = TypeVar("PrimitiveT")


class PrimitiveSkelton(ABC, Generic[PrimitiveT]):
    """light weight primitive shapes
    here, light weight means that each instance
    does not contain trimesh object, rather each
    only has analytical property as atributes.
    """

    sdf: Optional[SignedDistanceFunction]

    def to_visualizable(self, color: Optional[Tuple[int, int, int, int]] = None) -> PrimitiveT:
        primitive = self.to_skrobot_primitive()
        if color is not None:
            primitive.visual_mesh.visual.face_colors = color  # type: ignore
        return primitive

    @abstractmethod
    def to_skrobot_primitive(self) -> PrimitiveT:
        ...


class BoxSkeleton(CascadedCoords, PrimitiveSkelton[Box]):
    # works as Box but does not have trimesh geometries
    _extents: np.ndarray

    def __init__(self, extents, pos=None, with_sdf=False):
        CascadedCoords.__init__(self, pos=pos)
        self._extents = extents
        if with_sdf:
            sdf = BoxSDF(np.zeros(3), extents)
            self.assoc(sdf.coords, relative_coords="local")
            self.sdf = sdf
        else:
            self.sdf = None

    @property
    def extents(self) -> np.ndarray:
        return np.array(self._extents)

    def to_skrobot_primitive(self) -> Box:
        box = Box(self.extents)
        box.newcoords(self.copy_worldcoords())
        return box

    def sample_points(self, n_sample: int, wrt: Literal["world", "local"] = "world") -> np.ndarray:
        points_local = np.random.rand(n_sample, 3) * self.extents[None, :] - 0.5 * self.extents
        if wrt == "local":
            return points_local
        else:
            return self.transform_vector(points_local)


class CylinderSkelton(CascadedCoords, PrimitiveSkelton[Cylinder]):
    radius: float
    height: float

    def __init__(self, radius, height, pos=(0, 0, 0), with_sdf=False):
        CascadedCoords.__init__(self, pos=pos)
        self.radius = radius
        self.height = height
        if with_sdf:
            sdf = CylinderSDF(np.zeros(3), height, radius)
            self.assoc(sdf.coords, relative_coords="local")
            self.sdf = sdf
        else:
            self.sdf = None

    def to_skrobot_primitive(self) -> Cylinder:
        cylidner = Cylinder(self.radius, self.height)
        cylidner.newcoords(self.copy_worldcoords())
        return cylidner
