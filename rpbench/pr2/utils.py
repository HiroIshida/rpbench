import uuid
from typing import Optional

import numpy as np
from skrobot.model import Link
from skrobot.sdf import GridSDF, trimesh2sdf


class MeshLink(Link):
    # this class is almost the same as the original MeshLink in skrobot
    # only difference is that the constructor accept forced_sdf

    def __init__(
        self,
        visual_mesh=None,
        pos=(0, 0, 0),
        rot=np.eye(3),
        name=None,
        with_sdf=False,
        dim_grid=100,
        padding_grid=5,
        forced_sdf: Optional[GridSDF] = None,
    ):
        if name is None:
            name = "meshlink_{}".format(str(uuid.uuid1()).replace("-", "_"))

        super(MeshLink, self).__init__(pos=pos, rot=rot, name=name)
        self.visual_mesh = visual_mesh
        if self.visual_mesh is not None:
            if isinstance(self.visual_mesh, list):
                self._collision_mesh = self.visual_mesh[0] + self.visual_mesh[1:]
            else:
                self._collision_mesh = self.visual_mesh
            self._collision_mesh.metadata["origin"] = np.eye(4)

        if with_sdf:
            if forced_sdf is None:
                sdf = trimesh2sdf(
                    self._collision_mesh, dim_grid=dim_grid, padding_grid=padding_grid
                )
                self.assoc(sdf.coords, relative_coords="local")
                self.sdf = sdf
            else:
                self.assoc(forced_sdf.coords, relative_coords="local")
                self.sdf = forced_sdf
