from typing import ClassVar, List, Optional, Type

import numpy as np
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, OMPLSolverResult, Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import FetchSpec
from plainmp.utils import primitive_to_plainmp_sdf
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle
from skrobot.model.primitives import Axis
from skrobot.viewers import PyrenderViewer

from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.thesis_jsk_table import (
    JskMessyTableWorld,
    JskTable,
    fit_radian,
)
from rpbench.interface import TaskExpression, TaskWithWorldCondBase


class ThesisJskTableTask(TaskWithWorldCondBase[JskMessyTableWorld, Coordinates, None]):
    REACHING_HEIGHT_MIN: ClassVar[float] = 0.07
    REACHING_HEIGHT_MAX: ClassVar[float] = 0.2
    REACHING_YAW_MIN: ClassVar[float] = -0.25 * np.pi
    REACHING_YAW_MAX: ClassVar[float] = 0.25 * np.pi
    REACHABILITY_RADIUS: ClassVar[float] = 0.9

    @classmethod
    def get_world_type(cls) -> Type[JskMessyTableWorld]:
        return JskMessyTableWorld

    @classmethod
    def from_semantic_params(
        cls, table_2dpos: np.ndarray, bbox_param_list: List[np.ndarray], target_pose: np.ndarray
    ) -> "TidyupTableTaskBase":
        """
        Args:
            table_2dpos: 2D position of the table
            bbox_param_list: List of bbox parameters [[x, y, yaw, w, d, h], ...]
            target_pose: Reaching target pose (x, y, z, yaw)
        NOTE: all in world (robot's root) frame
        """
        wt = cls.get_world_type()
        table = wt.from_semantic_params(table_2dpos, bbox_param_list)
        co = Coordinates()
        co.translate(target_pose[:3])
        co.rotate(target_pose[3], [0, 0, 1])
        return cls(table, co)

    def export_problem(self) -> Problem:
        fetch_spec = FetchSpec()
        xyz = self.description.worldpos()
        yaw = rpy_angle(self.description.worldrot())[0][0]
        np_pose = np.array([xyz[0], xyz[1], xyz[2], 0, 0, yaw])

        pose_cst = fetch_spec.create_gripper_pose_const(np_pose)

        create_bvh = False  # plainmp bvh is buggy
        sdf = UnionSDF([sksdf_to_cppsdf(o.sdf) for o in self.world.get_all_obstacles()], create_bvh)
        ineq_cst = fetch_spec.create_collision_const()
        ineq_cst.set_sdf(sdf)

        lb, ub = fetch_spec.angle_bounds()
        motion_step_box = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2]
        return Problem(fetch_spec.q_reset_pose(), lb, ub, pose_cst, ineq_cst, None, motion_step_box)

    @classmethod
    def from_task_param(
        cls: Type["TidyupTableTaskBase"], param: np.ndarray
    ) -> "TidyupTableTaskBase":
        world_param, other_param = param[:-4], param[-4:]
        wt = cls.get_world_type()
        world = wt.from_parameter(world_param)
        x, y, z, yaw = other_param
        co = Coordinates()
        co.translate([x, y, z])
        co.rotate(yaw, [0, 0, 1])
        return cls(world, co)

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        xyz = self.description.worldpos()
        yaw = rpy_angle(self.description.worldrot())[0][0]
        other_vector = np.array([xyz[0], xyz[1], xyz[2], yaw])

        if use_matrix:
            world_vec = self.world.table.worldpos()[:2]
            world_mat = create_heightmap_z_slice(
                self.world.obstacle_env_region, self.world.tabletop_obstacle_list, 112
            )
        else:
            world_vec = self.world.to_parameter()
            world_mat = None
        return TaskExpression(world_vec, world_mat, other_vector)

    def solve_default(self) -> OMPLSolverResult:
        prob = self.export_problem()
        conf = OMPLSolverConfig(n_max_call=50000, simplify=True)
        solver = OMPLSolver(conf)
        return solver.solve(prob)

    def is_out_of_distribution(self) -> bool:
        raise NotImplementedError
        # co = self.description
        # pos = co.worldpos()
        # table_pos = self.world.table.worldpos()
        # if self.world.is_out_of_distribution():
        #     return True
        # if np.linalg.norm(pos[:2]) > self.FETCH_REACHABLE_RADIUS:
        #     return True

        # x_min, x_max, y_min, y_max = self.reaching_target_xy_minmax()
        # if not (x_min <= pos[0] - table_pos[0] <= x_max):
        #     return True
        # if not (y_min <= pos[1] - table_pos[1] <= y_max):
        #     return True
        # if not (self.REACHING_HEIGHT_MIN <= pos[2] - table_pos[2] <= self.REACHING_HEIGHT_MAX):
        #     return True
        # yaw = rpy_angle(co.worldrot())[0][0]
        # if not (self.REACHING_YAW_MIN <= yaw <= self.REACHING_YAW_MAX):
        #     return True
        # # TODO: we need to check the collision with obstale condition...
        # return False

    @classmethod
    def sample_description(cls, world: JskMessyTableWorld) -> Optional[Coordinates]:
        while True:
            x = np.random.uniform(low=JskTable.TABLE_DEPTH * 0.5, high=JskTable.TABLE_DEPTH * 0.5)
            y = np.random.uniform(low=-JskTable.TABLE_WIDTH * 0.5, high=JskTable.TABLE_WIDTH * 0.5)
            z = np.random.uniform(low=cls.REACHING_HEIGHT_MIN, high=cls.REACHING_HEIGHT_MAX)
            pos = np.array([x, y, z])
            pr2_pos = world.pr2_coords[:2]
            if np.linalg.norm(pos[:2] - pr2_pos) < cls.REACHABILITY_RADIUS:
                continue
            pr2_yaw_angle = world.pr2_coords[2]
            yaw_plus = np.random.uniform(low=cls.REACHING_YAW_MIN, high=cls.REACHING_YAW_MAX)
            yaw = fit_radian(pr2_yaw_angle + yaw_plus)
            co = Coordinates(pos, [yaw, 0, 0])
            break

        # Collision check with objects on the table
        sdf = UnionSDF(
            [primitive_to_plainmp_sdf(o.to_skrobot_primitive()) for o in world.get_all_obstacles()]
        )
        dist = sdf.evaluate(co.worldpos())
        if dist < 0.05:
            return None

        co_slided = co.copy_worldcoords()
        co_slided.translate([-0.1, 0.0, 0.0])
        dist = sdf.evaluate(co_slided.worldpos())
        if dist < 0.05:
            return None

        return co

    @classmethod
    def get_robot_model(cls) -> None:
        # we dont use skrobot model in this task
        return None

    def create_viewer(self) -> PyrenderViewer:
        v = PyrenderViewer()
        self.world.visualize(v)
        ax = Axis.from_coords(self.description)
        v.add(ax)
        return v


if __name__ == "__main__":
    task = ThesisJskTableTask.sample()
