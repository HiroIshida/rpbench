import lzma
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, ClassVar, Generic, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
from skrobot.coordinates import CascadedCoords, Transform
from skrobot.model.primitives import Box, Cylinder, Link, MeshLink
from skrobot.sdf import BoxSDF, CylinderSDF, SignedDistanceFunction, trimesh2sdf
from trimesh import Trimesh

PrimitiveT = TypeVar("PrimitiveT", bound=Link)
SelfT = TypeVar("SelfT", bound="PrimitiveSkelton")


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


class MeshSkelton(CascadedCoords, PrimitiveSkelton[MeshLink]):
    mesh: Trimesh

    def __init__(self, mesh: Trimesh, **gridsdf_kwargs):
        CascadedCoords.__init__(self)
        self.mesh = mesh
        mesh.metadata["origin"] = np.eye(4)
        sdf = trimesh2sdf(mesh, **gridsdf_kwargs)
        self.assoc(sdf, relative_coords="local")
        self.sdf = sdf

    def _to_skrobot_primitive(self) -> MeshLink:
        mesh_link = MeshLink(self.mesh)
        mesh_link.newcoords(self.copy_worldcoords())
        return mesh_link

    def detach_clone(self) -> "MeshSkelton":
        m = MeshSkelton(self.mesh)
        m.newcoords(self.copy_worldcoords())
        return m

    @cached_property
    def surface_points(self):
        return self.sdf.surface_points(n_sample=100)[0]

    @property
    def vertices(self) -> np.ndarray:
        return self.transform_vector(self.mesh.vertices)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        b_min = np.min(self.vertices, axis=0)
        b_max = np.max(self.vertices, axis=0)
        return b_min, b_max

    @property
    def itp_fill_value(self) -> float:
        return self.sdf.itp.fill_value

    @itp_fill_value.setter
    def itp_fill_value(self, value: float):
        self.sdf.itp.fill_value = value


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

    def sample_points(
        self, n_sample: int, wrt: Literal["world", "local"] = "world", margin: float = 0.0
    ) -> np.ndarray:
        extents = self.extents - margin * 2
        points_local = np.random.rand(n_sample, 3) * extents[None, :] - 0.5 * extents
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


class SerializableTransform(Transform):
    n_bytes: ClassVar[int] = 96  # 8 * (3 + 9)

    def serialize(self) -> bytes:
        trans_bytes = self.translation.tobytes()
        rot_bytes = self.rotation.tobytes()
        return trans_bytes + rot_bytes

    @classmethod
    def deserialize(cls, serialized: bytes) -> "SerializableTransform":
        assert len(serialized) == cls.n_bytes
        translation = np.frombuffer(serialized[:24], dtype=np.float64)
        rotation = np.frombuffer(serialized[24:], dtype=np.float64).reshape(3, 3)
        return cls(translation, rotation)


@dataclass
class VoxelGridSkelton:
    tf_local_to_world: Transform
    extents: Tuple[float, float, float]
    resols: Tuple[int, int, int]
    n_bytes_decomp: ClassVar[int] = 120  # 96 for tf, 12 for extents, 12 for resols

    @property
    def intervals(self) -> np.ndarray:
        return np.array(self.extents) / np.array(self.resols)

    @classmethod
    def from_box(cls, box: BoxSkeleton, resols: Tuple[int, int, int]) -> "VoxelGridSkelton":
        tf_local_to_world = SerializableTransform(box.worldpos(), box.worldrot())
        return cls(tf_local_to_world, box.extents, resols)

    def get_eval_points(self) -> np.ndarray:
        half_ext = np.array(self.extents) * 0.5
        lins = [np.linspace(-half_ext[i], half_ext[i], self.resols[i]) for i in range(3)]
        X, Y, Z = np.meshgrid(*lins)
        points_wrt_local = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        points_wrt_world = self.tf_local_to_world.transform_vector(points_wrt_local)
        return points_wrt_world

    def points_to_indices(self, points_wrt_world: np.ndarray) -> np.ndarray:
        tf_world_to_local = self.tf_local_to_world.inverse_transformation()
        points_wrt_local = tf_world_to_local.transform_vector(points_wrt_world)
        lb = np.array(self.extents) * -0.5
        indices = np.floor((points_wrt_local - lb) / self.intervals).astype(int)
        indices_flat = (
            indices[:, 0]
            + indices[:, 1] * self.resols[0]
            + indices[:, 2] * self.resols[0] * self.resols[1]
        ).astype(int)
        indices_flat.sort()
        return indices_flat.astype(np.uint32)

    def indices_to_points(self, indices_flat: np.ndarray) -> np.ndarray:
        indices_z = indices_flat // (self.resols[0] * self.resols[1])
        indices_y = (indices_flat % (self.resols[0] * self.resols[1])) // self.resols[0]
        indices_x = indices_flat % self.resols[0]
        indices = np.stack([indices_x, indices_y, indices_z], axis=-1)

        lb = np.array(self.extents) * -0.5
        points_wrt_local = np.array(indices) * self.intervals + lb
        points_wrt_world = self.tf_local_to_world.transform_vector(points_wrt_local)
        return points_wrt_world

    def serialize(self) -> bytes:
        extents_bytes = struct.pack("fff", *self.extents)
        resols_bytes = struct.pack("iii", *self.resols)
        tf_bytes = self.tf_local_to_world.serialize()
        serialized = extents_bytes + resols_bytes + tf_bytes
        return zlib.compress(serialized, 0)

    @classmethod
    def deserialize(cls, serialized: bytes) -> "VoxelGridSkelton":
        unziped = zlib.decompress(serialized)
        assert len(unziped) == cls.n_bytes_decomp
        extents = struct.unpack("fff", unziped[:12])
        resols = struct.unpack("iii", unziped[12:24])
        tf_local_to_world = SerializableTransform.deserialize(unziped[24:])
        return cls(tf_local_to_world, extents, resols)


class ZlibCompressor:
    @staticmethod
    def compress(data: bytes) -> bytes:
        return zlib.compress(data, 1)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        return zlib.decompress(data)


class LzmaCompressor:
    @staticmethod
    def compress(data: bytes) -> bytes:
        return lzma.compress(data, preset=1)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        return lzma.decompress(data)


@dataclass
class VoxelGrid(LzmaCompressor):
    skelton: VoxelGridSkelton
    indices: np.ndarray

    @classmethod
    def from_points(
        cls,
        points_world: np.ndarray,
        skelton: VoxelGridSkelton,
        np_type: Union[None, np.uint8, np.uint16, np.uint32, np.uint64] = None,
    ) -> "VoxelGrid":
        indices = skelton.points_to_indices(points_world)
        return cls(skelton, indices)

    @classmethod
    def from_sdf(
        cls, sdf: Callable[[np.ndarray], np.ndarray], skelton: VoxelGridSkelton
    ) -> "VoxelGrid":
        points = skelton.get_eval_points()
        sdf_values = sdf(points)
        width = np.max(skelton.intervals)
        surface_indices = np.logical_and(sdf_values < 0, sdf_values > -width)
        return cls.from_points(points[surface_indices], skelton)

    def serialize(self) -> bytes:
        skelton_bytes = self.skelton.serialize()
        indices_bytes = self.indices.tobytes()
        indices_comp_bytes = self.compress(indices_bytes)
        skelton_bytes_size_bytes = struct.pack("I", len(skelton_bytes))
        return skelton_bytes_size_bytes + skelton_bytes + indices_comp_bytes

    @classmethod
    def deserialize(cls, serialized: bytes) -> "VoxelGrid":
        skelton_bytes_size = struct.unpack("I", serialized[:4])[0]
        bytes_skelton = serialized[4 : 4 + skelton_bytes_size]
        skelton = VoxelGridSkelton.deserialize(bytes_skelton)
        bytes_other = serialized[4 + skelton_bytes_size :]
        unziped = cls.decompress(bytes_other)
        indices = np.frombuffer(unziped, dtype=np.uint32)
        return cls(skelton, indices)

    def to_points(self) -> np.ndarray:
        points_wrt_world = self.skelton.indices_to_points(self.indices)
        return points_wrt_world
