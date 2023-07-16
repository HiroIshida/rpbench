import uuid

import numpy as np
from skrobot.model import Link
from skrobot.model.primitives import Box
from skrobot.sdf import BoxSDF


class BoxSkeleton(Link):
    # works as Box but does not have trimesh geometries

    def __init__(self, extents, pos=(0, 0, 0), rot=np.eye(3), name=None, with_sdf=False):
        if name is None:
            name = "box_{}".format(str(uuid.uuid1()).replace("-", "_"))

        super().__init__(pos=pos, rot=rot, name=name, collision_mesh=None, visual_mesh=None)
        self._extents = extents
        if with_sdf:
            sdf = BoxSDF(np.zeros(3), extents)
            self.assoc(sdf.coords, relative_coords="local")
            self.sdf = sdf

    def to_box(self) -> Box:
        box = Box(self._extents)
        box.newcoords(self.copy_worldcoords())
        return box
