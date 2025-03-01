import time
from typing import ClassVar, List, Optional, Type

import numpy as np
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, OMPLSolverResult, Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import PR2LarmSpec, PR2RarmSpec
from plainmp.utils import primitive_to_plainmp_sdf
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle
from skrobot.model.primitives import Axis
from skrobot.viewers import PyrenderViewer

from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.thesis_jsk_table import (
    AV_INIT,
    LARM_INIT_ANGLES,
    RARM_INIT_ANGLES,
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

    # @classmethod
    # @abstractmethod
    # def is_rarm(cls) -> bool:
    #     ...
    def is_rarm(self) -> bool:
        return False

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
        if self.is_rarm():
            spec = PR2RarmSpec()
            q_init = RARM_INIT_ANGLES
        else:
            spec = PR2LarmSpec()
            q_init = LARM_INIT_ANGLES
        pr2 = spec.get_robot_model(deepcopy=False)
        pr2.angle_vector(AV_INIT)
        x_pos, y_pos, yaw = self.world.pr2_coords
        pr2.newcoords(Coordinates([x_pos, y_pos, 0], [yaw, 0, 0]))
        spec.reflect_skrobot_model_to_kin(pr2)

        xyz = self.description.worldpos()
        yaw = rpy_angle(self.description.worldrot())[0][0]
        np_pose = np.array([xyz[0], xyz[1], xyz[2], 0, 0, yaw])
        pose_cst = spec.create_gripper_pose_const(np_pose)
        sdf = UnionSDF([p.to_plainmp_sdf() for p in self.world.get_all_obstacles()])
        ineq_cst = spec.create_collision_const()
        ineq_cst.set_sdf(sdf)

        lb, ub = spec.angle_bounds()
        motion_step_box = np.array([0.02] * 7)
        return Problem(q_init, lb, ub, pose_cst, ineq_cst, None, motion_step_box)

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
        n_max_trial = 30
        count = 0
        while True:
            if count > n_max_trial:
                return None
            count += 1
            x = np.random.uniform(low=-JskTable.TABLE_DEPTH * 0.5, high=JskTable.TABLE_DEPTH * 0.5)
            y = np.random.uniform(low=-JskTable.TABLE_WIDTH * 0.5, high=JskTable.TABLE_WIDTH * 0.5)
            z = (
                np.random.uniform(low=cls.REACHING_HEIGHT_MIN, high=cls.REACHING_HEIGHT_MAX)
                + JskTable.TABLE_HEIGHT
            )
            pos = np.array([x, y, z])
            pr2_pos = world.pr2_coords[:2]
            dist = np.linalg.norm(pos[:2] - pr2_pos)
            print(dist)
            if dist > cls.REACHABILITY_RADIUS:
                continue
            pr2_yaw_angle = world.pr2_coords[2]
            yaw_plus = np.random.uniform(low=cls.REACHING_YAW_MIN, high=cls.REACHING_YAW_MAX)
            yaw = fit_radian(pr2_yaw_angle + yaw_plus)
            yaw = pr2_yaw_angle
            co = Coordinates(pos, [yaw, 0, 0])
            break

        # Collision check with objects on the table
        sdf = UnionSDF(
            [
                primitive_to_plainmp_sdf(o.to_skrobot_primitive())
                for o in world.tabletop_obstacle_list
            ]
        )
        dist = sdf.evaluate(co.worldpos())
        print(dist)
        if dist > 0.15:  # too far
            return None
        if dist < 0.03:
            return None

        co_slided = co.copy_worldcoords()
        co_slided.translate([-0.1, 0.0, 0.0])
        dist = sdf.evaluate(co_slided.worldpos())
        if dist < 0.03:
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
    # np.random.seed(14)
    task = ThesisJskTableTask.sample()
    problem = task.export_problem()

    solver = OMPLSolver(OMPLSolverConfig(shortcut=True, bspline=True))
    ts = time.time()
    ret = solver.solve(problem)
    print(ret.terminate_state)
    print(ret.n_call)
    print(ret.time_elapsed)

    x, y, yaw = task.world.pr2_coords
    from skrobot.models.pr2 import PR2

    pr2 = PR2(use_tight_joint_limit=False)
    pr2.angle_vector(AV_INIT)
    pr2.newcoords(Coordinates([x, y, 0], [yaw, 0, 0]))
    v = task.create_viewer()
    v.add(pr2)
    v.show()
    rarm_spec = PR2LarmSpec()
    for q in ret.traj.resample(40):
        rarm_spec.set_skrobot_model_state(pr2, q)
        v.redraw()
        time.sleep(0.1)
    time.sleep(10)
