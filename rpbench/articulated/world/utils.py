import numpy as np
from skrobot.model import Link
from skrobot.model.primitives import Box, Cylinder
from skrobot.sdf import BoxSDF, CylinderSDF


class BoxSkeleton(Link):
    # works as Box but does not have trimesh geometries

    def __init__(self, extents, pos=(0, 0, 0), rot=np.eye(3), with_sdf=False):
        super().__init__(pos=pos, rot=rot, name="", collision_mesh=None, visual_mesh=None)
        self._extents = extents
        if with_sdf:
            sdf = BoxSDF(np.zeros(3), extents)
            self.assoc(sdf.coords, relative_coords="local")
            self.sdf = sdf

    def to_visualizable(self) -> Box:
        box = Box(self._extents)
        box.newcoords(self.copy_worldcoords())
        return box


class CylinderSkelton(Link):
    radius: float
    height: float

    def __init__(self, radius, height, pos=(0, 0, 0), rot=np.eye(3), with_sdf=False):
        super().__init__(pos=pos, rot=rot, name="", collision_mesh=None, visual_mesh=None)
        self.radius = radius
        self.height = height
        if with_sdf:
            sdf = CylinderSDF(np.zeros(3), height, radius)
            self.assoc(sdf.coords, relative_coords="local")
            self.sdf = sdf

    def to_visualizable(self) -> Cylinder:
        cylidner = Cylinder(self.radius, self.height)
        cylidner.newcoords(self.copy_worldcoords())
        return cylidner
