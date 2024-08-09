from typing import Optional, Type

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
    FETCH_REACHABLE_RADIUS = 1.0

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
        world_param, other_param = param[:-4], param[-4:]
        world = JskMessyTableWorld.from_parameter(world_param)
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

    @classmethod
    def sample_description(cls, world: JskMessyTableWorld) -> Optional[Coordinates]:
        while True:
            x = np.random.uniform(low=-0.5 * world.table.size[0], high=0.5 * world.table.size[0])
            y = np.random.uniform(low=-0.5 * world.table.size[1], high=0.5 * world.table.size[1])
            z = np.random.uniform(low=0.07, high=0.2)
            yaw = np.random.uniform(low=-0.25 * np.pi, high=0.25 * np.pi)
            co = world.table.copy_worldcoords()
            co.translate([x, y, z])
            co.rotate(yaw, [0, 0, 1])
            if np.linalg.norm(co.worldpos()[:2]) < cls.FETCH_REACHABLE_RADIUS:
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

    while True:
        task = TidyupTableTask.sample()
        task = TidyupTableTask.from_task_param(task.to_task_param())
        ret = task.solve_default()
        if ret.traj is not None:
            print(ret)
            fs = FetchSpec()
            fetch = Fetch()
            fetch.reset_pose()
            v = task.create_viewer()
            v.add(fetch)
            v.show()
            time.sleep(1)
            for q in ret.traj.numpy():
                set_robot_state(fetch, fs.control_joint_names, q)
                v.redraw()
                time.sleep(0.3)
            time.sleep(1000000000)
