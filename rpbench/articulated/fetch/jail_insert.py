from dataclasses import dataclass
from typing import Optional, Type

import numpy as np

from rpbench.interface import ResultProtocol, TaskWithWorldCondBase

try:
    from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, Problem
    from plainmp.robot_spec import FetchSpec

    from rpbench.articulated.world.jail import JailWorld
except ImportError:
    raise ImportError("Please install plainmp (private repo) to run this task.")

_debug_mode_flag = [False]


@dataclass
class BytesTaskExpression:
    # this does not follows the protocol!
    # because get_vector returns bytes
    world_vec: Optional[bytes]
    world_mat: Optional[np.ndarray]
    other_vec: np.ndarray  # must be double and 24 (3 * 8) bytes

    def get_matrix(self) -> Optional[np.ndarray]:
        return self.world_mat

    def get_vector(self) -> bytes:
        other_vec_bytes = self.other_vec.tobytes()
        return self.world_vec + other_vec_bytes


class JailInsertTask(TaskWithWorldCondBase[JailWorld, np.ndarray, None]):
    @classmethod
    def from_task_param(cls, param: bytes) -> "JailInsertTask":
        world_bytes, other_vec_bytes = param[:-24], param[-24:]
        world = JailWorld.deserialize(world_bytes)
        other_vec = np.frombuffer(other_vec_bytes, dtype=np.float64)
        return cls(world, other_vec)

    def export_task_expression(self, use_matrix: bool) -> "BytesTaskExpression":
        if use_matrix:
            world_vec = None
            world_mat = self.world.voxels.to_3darray()
        else:
            world_vec = self.world.serialize()
            world_mat = None
        return BytesTaskExpression(world_vec, world_mat, self.description)

    def solve_default(self) -> ResultProtocol:
        prob = self.export_problem()
        if _debug_mode_flag[0]:
            conf = OMPLSolverConfig(1000_0000, algorithm_range=None, simplify=True, timeout=3.0)
        else:
            conf = OMPLSolverConfig(1000_0000, algorithm_range=None, simplify=True, timeout=30.0)
        solver = OMPLSolver(conf)
        return solver.solve(prob)

    def export_problem(self) -> Problem:
        fetch_spec = FetchSpec()
        pose_cst = fetch_spec.create_gripper_pose_const(self.description)
        sdf = self.world.get_plainmp_sdf()
        ineq_cst = fetch_spec.create_collision_const()
        ineq_cst.set_sdf(sdf)
        lb, ub = fetch_spec.angle_bounds()
        motion_step_box = [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.1, 0.1]
        return Problem(fetch_spec.q_reset_pose(), lb, ub, pose_cst, ineq_cst, None, motion_step_box)
        raise NotImplementedError

    @staticmethod
    def get_world_type() -> Type[JailWorld]:
        return JailWorld

    @classmethod
    def get_robot_model(cls) -> None:
        return None

    @classmethod
    def sample_description(cls, world: JailWorld) -> Optional[np.ndarray]:
        co = world.region.copy_worldcoords()
        x_trans = 0.2  # fixed for now
        y_trans = np.random.uniform(-0.2, 0.2)
        z_trans = np.random.uniform(-0.2, 0.2)
        co.translate([x_trans, y_trans, z_trans])
        p = co.worldpos()
        sdf = world.get_plainmp_sdf()
        val = sdf.evaluate(p)
        collision_free = val > 0.05
        if collision_free:
            return p
        return None


if __name__ == "__main__":
    np.random.seed(0)
    task = JailInsertTask.sample()
    problem = task.export_problem()
    conf = OMPLSolverConfig(1000_0000, algorithm_range=None, simplify=True, timeout=1)
    solver = OMPLSolver(conf)
    ret = solver.solve(problem)
    print(ret.time_elapsed)
    assert ret.traj is not None
