from typing import ClassVar, List, Optional, Type

import numpy as np
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, OMPLSolverResult, Problem
from plainmp.psdf import UnionSDF
from plainmp.robot_spec import FetchSpec
from plainmp.utils import sksdf_to_cppsdf
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle
from skrobot.model.primitives import Axis
from skrobot.models.fetch import Fetch
from skrobot.sdf import UnionSDF as skUnionSDF
from skrobot.viewers import PyrenderViewer

from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.jsk_table import JskMessyTableWorld
from rpbench.interface import TaskExpression, TaskWithWorldCondBase


class TidyupTableTask(TaskWithWorldCondBase[JskMessyTableWorld, Coordinates, None]):
    FETCH_REACHABLE_RADIUS: ClassVar[float] = 1.0
    RELATIVE_REACHING_HEIGHT_MIN: ClassVar[float] = 0.07
    RELATIVE_REACHING_HEIGHT_MAX: ClassVar[float] = 0.2
    RELATIVE_REACHING_YAW_MIN: ClassVar[float] = -0.25 * np.pi
    RELATIVE_REACHING_YAW_MAX: ClassVar[float] = 0.25 * np.pi
    MAX_ATTACK_ANGLE: ClassVar[float] = 0.5 * np.pi

    @classmethod
    def from_semantic_params(
        cls,
        relative_fetch_pose: np.ndarray,
        bbox_param_list: List[np.ndarray],
        relative_target_pose: np.ndarray,
    ) -> "TidyupTableTask":
        """
        Args:
            relative_fetch_pose: [x, y, yaw]
            bbox_param_list: List of bbox parameters [[x, y, yaw, w, d, h], ...] where pose of the bbox is relative to table
            target_pose: Reaching target pose (x, y, z, yaw)
        NOTE: all in world (robot's root) frame
        """
        world = JskMessyTableWorld.from_semantic_params(relative_fetch_pose, bbox_param_list)
        co = world.table.copy_worldcoords()
        co.translate(relative_target_pose[:3])
        co.rotate(relative_target_pose[3], "z")
        return cls(world, co)

    @staticmethod
    def get_world_type() -> Type[JskMessyTableWorld]:
        return JskMessyTableWorld

    def export_problem(self) -> Problem:
        fetch_spec = FetchSpec()
        xyz = self.description.worldpos()
        yaw = rpy_angle(self.description.worldrot())[0][0]
        np_pose = np.array([xyz[0], xyz[1], xyz[2], 0, 0, yaw])

        pose_cst = fetch_spec.create_gripper_pose_const(np_pose)

        create_bvh = True
        sdf = UnionSDF([sksdf_to_cppsdf(o.sdf) for o in self.world.get_all_obstacles()], create_bvh)
        ineq_cst = fetch_spec.create_collision_const()
        ineq_cst.set_sdf(sdf)

        lb, ub = fetch_spec.angle_bounds()
        motion_step_box = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2]
        return Problem(fetch_spec.q_reset_pose(), lb, ub, pose_cst, ineq_cst, None, motion_step_box)

    @classmethod
    def from_task_param(cls: Type["TidyupTableTask"], param: np.ndarray) -> "TidyupTableTask":
        world_param, relative_reaching_pose = param[:-4], param[-4:]
        world = JskMessyTableWorld.from_parameter(world_param)
        x, y, z, yaw = relative_reaching_pose
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
        if self.world.is_out_of_distribution():
            return True

        co_reaching = self.description
        # check in world coords
        xy_pos = co_reaching.worldpos()[:2]
        if np.linalg.norm(xy_pos) > self.FETCH_REACHABLE_RADIUS:
            return True
        yaw = rpy_angle(co_reaching.worldrot())[0][0]
        if abs(yaw) > self.MAX_ATTACK_ANGLE:
            return True

        # check in table coords
        tf_reach2world = co_reaching.get_transform()
        tf_table2world = self.world.table.get_transform()
        tf_reach2table = tf_reach2world * tf_table2world.inverse_transformation()
        x, y, _ = tf_reach2table.translation
        if not abs(x) <= 0.5 * self.world.table.TABLE_DEPTH:
            return True
        if not abs(y) <= 0.5 * self.world.table.TABLE_WIDTH:
            return True
        yaw = rpy_angle(tf_reach2table.rotation)[0][0]
        if not (self.RELATIVE_REACHING_YAW_MIN <= yaw <= self.RELATIVE_REACHING_YAW_MAX):
            return True
        # TODO: we need to check the collision with obstale condition...
        return False

    @classmethod
    def sample_description(cls, world: JskMessyTableWorld) -> Optional[Coordinates]:
        while True:
            x = np.random.uniform(
                low=-0.5 * world.table.TABLE_DEPTH, high=0.5 * world.table.TABLE_DEPTH
            )
            y = np.random.uniform(
                low=-0.5 * world.table.TABLE_WIDTH, high=0.5 * world.table.TABLE_WIDTH
            )
            z = np.random.uniform(low=cls.RELATIVE_REACHING_HEIGHT_MIN, high=cls.RELATIVE_REACHING_HEIGHT_MAX)
            yaw = np.random.uniform(low=cls.RELATIVE_REACHING_YAW_MIN, high=cls.RELATIVE_REACHING_YAW_MAX)
            co = world.table.copy_worldcoords()
            co.translate([x, y, z])
            co.rotate(yaw, [0, 0, 1])
            if np.linalg.norm(co.worldpos()[:2]) > cls.FETCH_REACHABLE_RADIUS:
                continue
            ypr = rpy_angle(co.worldrot())[0]
            yaw = ypr[0]
            if abs(yaw) < cls.MAX_ATTACK_ANGLE:
                break

        # Collision check with objects on the table
        sdf = skUnionSDF([o.sdf for o in world.get_all_obstacles()])
        dist = sdf(np.array([co.worldpos()]))[0]
        if dist < 0.1:
            return None

        co_slided = co.copy_worldcoords()
        co_slided.translate([-0.1, 0.0, 0.0])
        dist = sdf(np.array([co_slided.worldpos()]))[0]
        if dist < 0.1:
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
    import time

    from skmp.robot.utils import set_robot_state

    np.random.seed(0)

    table_pos = np.array([0.8, 0.0, 0.6])
    bbox_param_list = [
        np.array([0.8, 0.2, 0.4, 0.1, 0.1, 0.2]),
        np.array([0.6, -0.4, 0.0, 0.1, 0.2, 0.25]),
    ]
    # task = TidyupTableTask.from_semantic_params(table_pos, bbox_param_list, [-0.3, 0.0, 0.1, 0.0])
    task = TidyupTableTask.sample()
    task = task.from_task_param(task.to_task_param())
    v = task.create_viewer()
    fs = FetchSpec()
    fetch = Fetch()
    fetch.reset_pose()
    v.add(fetch)
    v.show()
    time.sleep(1)

    solve = False

    if solve:
        ret = task.solve_default()

        for q in ret.traj.numpy():
            set_robot_state(fetch, fs.control_joint_names, q)
            v.redraw()
            time.sleep(0.3)
        time.sleep(1000000000)
    else:
        time.sleep(1000000000)
