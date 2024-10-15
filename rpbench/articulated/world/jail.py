import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, List, Union

import numpy as np
from conway_tower import evolve_conway
from plainmp.psdf import CloudSDF
from plainmp.psdf import UnionSDF as pUnionSDF
from plainmp.utils import sksdf_to_cppsdf
from scipy.spatial import KDTree
from skrobot.model.primitives import Axis, PointCloudLink
from skrobot.sdf import UnionSDF
from skrobot.viewers import PyrenderViewer, TrimeshSceneViewer

from rpbench.articulated.world.utils import (
    BoxSkeleton,
    CylinderSkelton,
    VoxelGrid,
    VoxelGridSkelton,
)
from rpbench.interface import SamplableWorldBase

BROWN_COLOR = (204, 102, 0, 200)


@dataclass
class JailWorldBase(SamplableWorldBase):
    region: BoxSkeleton
    panels: List[BoxSkeleton]
    voxels: VoxelGrid
    box_width: ClassVar[float] = 0.7
    box_height: ClassVar[float] = 0.7
    box_depth: ClassVar[float] = 0.6
    panel_thickness: ClassVar[float] = 0.01

    @classmethod
    def create_region_and_panels(cls):
        # define attention region
        region = BoxSkeleton([cls.box_depth, cls.box_width, cls.box_height])
        region.translate([0, 0, cls.box_height * 0.5])

        # jail box
        bottom = BoxSkeleton([cls.box_depth, cls.box_width, cls.panel_thickness])
        bottom.translate([0, 0, cls.panel_thickness * 0.5])
        top = BoxSkeleton([cls.box_depth, cls.box_width, cls.panel_thickness])
        top.translate([0, 0, cls.box_height - cls.panel_thickness * 0.5])
        left = BoxSkeleton([cls.box_depth, cls.panel_thickness, cls.box_height])
        left.translate([0, cls.box_width * 0.5 - cls.panel_thickness * 0.5, cls.box_height * 0.5])
        right = BoxSkeleton([cls.box_depth, cls.panel_thickness, cls.box_height])
        right.translate([0, -cls.box_width * 0.5 + cls.panel_thickness * 0.5, cls.box_height * 0.5])
        panels = [bottom, top, left, right]
        for panel in panels:
            region.assoc(panel)
        # slide jail
        region.translate([0.7, 0, 0.6])
        return region, panels

    def visualize(self, viewer: Union[PyrenderViewer, TrimeshSceneViewer]):
        for panel in self.panels:
            viewer.add(panel.to_visualizable(BROWN_COLOR))
        cloud = self.voxels.to_points()
        plink = PointCloudLink(cloud)
        viewer.add(plink)

    def get_plainmp_sdf(self):
        cloud_sdf = CloudSDF(self.voxels.to_points(), 0.0)
        sdf_list = [cloud_sdf]
        for panel in self.panels:
            sdf = sksdf_to_cppsdf(panel.sdf)
            sdf_list.append(sdf)
        union_sdf = pUnionSDF(sdf_list, False)
        return union_sdf

    def get_sdf(self, optional_sdfs=None) -> Callable[[np.ndarray], np.ndarray]:
        kdtree = KDTree(self.voxels.to_points())

        if optional_sdfs is None:
            optional_sdfs = []

        def sdf(X: np.ndarray) -> np.ndarray:
            dists, _ = kdtree.query(X)
            for panel in self.panels + optional_sdfs:
                dists = np.minimum(dists, panel.sdf(X))
            return dists

        return sdf

    def serialize(self) -> bytes:
        return self.voxels.serialize()

    @classmethod
    def deserialize(cls, data: bytes) -> "JailWorldBase":
        voxels = VoxelGrid.deserialize(data)
        region, panels = cls.create_region_and_panels()
        return cls(region, panels, voxels)

    @classmethod
    @abstractmethod
    def sample(cls, standard: bool = False) -> "JailWorldBase":
        pass


class JailWorld(JailWorldBase):
    @classmethod
    def sample(cls, standard: bool = False) -> "JailWorld":
        region, panels = cls.create_region_and_panels()

        # sample jail bars
        sizes = np.array([cls.box_depth, cls.box_width, cls.box_height])
        margin = 0.1
        radius_min = 0.015
        radius_max = 0.05
        lb = -sizes * 0.5 + margin
        ub = sizes * 0.5 - margin
        n_bar = np.random.randint(1, 6)
        infinite_length = 10.0
        bars = []
        for _ in range(n_bar):
            pos = np.random.uniform(lb, ub)
            radius = np.random.uniform(radius_min, radius_max)
            bar = CylinderSkelton(radius, infinite_length)
            bar.translate(pos)
            roll = np.random.uniform(0, np.pi / 4)
            bar.rotate(roll, axis=[1, 0, 0])
            pitch = np.random.uniform(0, 2 * np.pi)
            bar.rotate(pitch, axis=[0, 0, 1], wrt="world")
            bars.append(bar)
        for bar in bars:
            region.assoc(bar, relative_coords="local")

        bar_sdf = UnionSDF([bar.sdf for bar in bars])

        # create voxel grid
        voxel_skeleton = VoxelGridSkelton.from_box(region, (56, 56, 56))
        voxel_grid = VoxelGrid.from_sdf(bar_sdf, voxel_skeleton)
        return cls(region, panels, voxel_grid)


class ConwayJailWorld(JailWorldBase):
    @classmethod
    def sample(cls, standard: bool = False) -> "JailWorld":
        region, panels = cls.create_region_and_panels()
        x_margin = 10
        y_margin = 5
        n_bar_from_bottom = np.random.randint(1, 5)
        mat_3d_now = np.zeros((56, 56, 56), dtype=bool)
        for i in range(n_bar_from_bottom):
            x_center = np.random.randint(x_margin, 56 - x_margin)
            y_center = np.random.randint(y_margin, 56 - y_margin)
            init_layer = np.zeros((56, 56), dtype=bool)
            init_layer[x_center - 5 : x_center + 5, y_center - 5 : y_center + 5] = np.random.choice(
                [0, 1], (10, 10), p=[0.5, 0.5]
            )
            random_perturbs = np.random.randint(-1, 2, (56, 2))
            mat_3d = evolve_conway(init_layer, 56, perturb=random_perturbs, z_as_time=True)
            if np.random.rand() < 0.5:
                mat_3d = np.flip(mat_3d, axis=2)
            mat_3d_now = np.logical_or(mat_3d_now, mat_3d)

        skelton = VoxelGridSkelton.from_box(region, (56, 56, 56))
        indices = np.where(mat_3d_now)
        indices_flat = indices[0] + indices[1] * 56 + indices[2] * 56 * 56
        grid = VoxelGrid(skelton, indices_flat)
        return cls(region, panels, grid)


if __name__ == "__main__":
    from plainmp.ik import solve_ik
    from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
    from plainmp.problem import Problem
    from plainmp.robot_spec import FetchSpec
    from skrobot.models.fetch import Fetch

    np.random.seed(3)
    world = JailWorld.sample()
    sdf = world.get_plainmp_sdf()
    co = world.region.copy_worldcoords()
    co.translate([0.25, 0.05, 0.0])
    axis = Axis.from_coords(co)

    fs = FetchSpec()
    eq_cst = fs.create_gripper_pose_const(co.worldpos())
    ineq_cst = fs.create_collision_const()
    ineq_cst.set_sdf(sdf)
    lb, ub = fs.angle_bounds()

    start = np.array([0.0, 1.31999949, 1.40000015, -0.20000077, 1.71999929, 0.0, 1.6600001, 0.0])
    msbox = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.1, 0.1])
    solver = OMPLSolver(OMPLSolverConfig(50000000, algorithm_range=None, simplify=True))

    ts = time.time()
    while True:
        ret_ik = solve_ik(eq_cst, ineq_cst, lb, ub, q_seed=None, max_trial=100)
        if not ineq_cst.is_valid(ret_ik.q):
            continue
        goal = ret_ik.q
        problem = Problem(start, lb, ub, goal, ineq_cst, None, msbox)
        ret = solver.solve(problem)
        if ret.traj is not None:
            break
    print(f"elapsed time: {time.time() - ts}")

    co = world.region.copy_worldcoords()
    co.translate([0.27, 0.04, 0.0])
    eq_cst = fs.create_gripper_pose_const(co.worldpos())
    problem = Problem(start, lb, ub, eq_cst, ineq_cst, None, msbox)

    solver = OMPLSolver(
        OMPLSolverConfig(
            50000000, algorithm_range=None, simplify=True, ertconnect_eps=0.1, n_max_ik_trial=1
        )
    )
    ret = solver.solve(problem, ret.traj)
    print(ret)

    fetch = Fetch()
    set_robot_state(fetch, fs.control_joint_names, ret_ik.q)
    viewer = PyrenderViewer()
    world.visualize(viewer)
    viewer.add(fetch)
    viewer.add(axis)
    viewer.show()

    for q in ret.traj:
        set_robot_state(fetch, fs.control_joint_names, q)
        time.sleep(1)
        viewer.redraw()
    import time

    time.sleep(1000)
