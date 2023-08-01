from typing import ClassVar, List, Tuple, Type

from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skrobot.coordinates import Coordinates
from skrobot.model.robot_model import RobotModel

from rpbench.articulated.pr2.common import (
    CachedPR2ConstProvider,
    CachedRArmPR2ConstProvider,
)
from rpbench.articulated.world.oven import TabletopClutteredOvenWorld
from rpbench.interface import (
    DescriptionTable,
    Problem,
    ReachingTaskBase,
    ResultProtocol,
)
from rpbench.utils import skcoords_to_pose_vec


class TabletopClutteredOvenReachingTask(ReachingTaskBase[TabletopClutteredOvenWorld, RobotModel]):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]] = CachedRArmPR2ConstProvider

    @staticmethod
    def get_world_type() -> Type[TabletopClutteredOvenWorld]:
        return TabletopClutteredOvenWorld

    @staticmethod
    def get_robot_model() -> RobotModel:
        return CachedPR2ConstProvider.get_pr2()

    @classmethod
    def get_dof(cls) -> int:
        return cls.config_provider.get_dof()

    @classmethod
    def sample_descriptions(
        cls, world: TabletopClutteredOvenWorld, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates, ...]]:

        if standard:
            assert n_sample == 1

        pose_list: List[Tuple[Coordinates]] = []
        while len(pose_list) < n_sample:
            pose = world.sample_pregrasp_coords()
            if pose is not None:
                pose_list.append((pose,))
        return pose_list

    @staticmethod
    def create_cache(world: TabletopClutteredOvenWorld, robot_model: RobotModel) -> None:
        return None

    def export_table(self) -> DescriptionTable:
        world_dict = {}
        world_dict["vector"] = self.world.vector_description
        world_dict["mesh"] = self.world.oven_conts.create_heightmap()

        desc_dicts = []
        for desc in self.descriptions:
            desc_dict = {}
            for idx, co in enumerate(desc):
                pose = skcoords_to_pose_vec(co)
                name = "target_pose-{}".format(idx)
                desc_dict[name] = pose
            desc_dicts.append(desc_dict)
        return DescriptionTable(world_dict, desc_dicts)

    def export_problems(self) -> List[Problem]:
        provider = self.config_provider
        q_start = provider.get_start_config()
        box_const = provider.get_box_const()

        sdf = self.world.get_exact_sdf()
        ineq_const = provider.get_collfree_const(sdf)

        problems = []
        for desc in self.descriptions:
            pose_const = provider.get_pose_const(list(desc))
            problem = Problem(q_start, box_const, pose_const, ineq_const, None)
            problems.append(problem)
        return problems

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        solcon = OMPLSolverConfig(n_max_call=10000, n_max_satisfaction_trial=200, simplify=True)

        ompl_solver = OMPLSolver.init(solcon)
        ompl_solver.setup(problem)
        return ompl_solver.solve()
