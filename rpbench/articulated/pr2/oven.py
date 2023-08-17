from typing import Any, ClassVar, Iterator, List, Literal, Tuple, Type, overload

import numpy as np
from skmp.constraint import CollFreeConst
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.visualization.solution_visualizer import (
    InteractiveSolutionVisualizer,
    SolutionVisualizerBase,
    StaticSolutionVisualizer,
)
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.model.robot_model import RobotModel
from tinyfk import BaseType

from rpbench.articulated.pr2.common import (
    CachedRArmFixedPR2ConstProvider,
    CachedRArmPR2ConstProvider,
)
from rpbench.articulated.world.oven import TabletopClutteredOvenWorld
from rpbench.interface import DescriptionTable, Problem, ResultProtocol, TaskBase
from rpbench.utils import skcoords_to_pose_vec, temp_seed


class TabletopClutteredOvenReachingTask(
    TaskBase[TabletopClutteredOvenWorld, Tuple[Coordinates, np.ndarray], RobotModel]
):
    config_provider: ClassVar[
        Type[CachedRArmFixedPR2ConstProvider]
    ] = CachedRArmFixedPR2ConstProvider

    @staticmethod
    def get_world_type() -> Type[TabletopClutteredOvenWorld]:
        return TabletopClutteredOvenWorld

    @staticmethod
    def get_robot_model() -> RobotModel:
        return CachedRArmFixedPR2ConstProvider.get_pr2()

    @classmethod
    def get_dof(cls) -> int:
        return cls.config_provider.get_dof()

    @classmethod
    def sample_descriptions(
        cls, world: TabletopClutteredOvenWorld, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates, np.ndarray]]:

        if standard:
            assert n_sample == 1

        with temp_seed(0, use_tempseed=standard):
            pose_list: List[Coordinates] = []
            while len(pose_list) < n_sample:
                pose = world.sample_pregrasp_coords()
                if pose is not None:
                    pose_list.append(pose)

        # note that we temporaly use Planer base here
        # but in the planning time the base is fixed
        # colkin = CachedRArmPR2ConstProvider.get_colkin()
        colkin = CachedRArmFixedPR2ConstProvider.get_colkin()
        BaseType.PLANER

        pr2 = CachedRArmPR2ConstProvider.get_pr2()
        sdf = world.get_exact_sdf()
        collfree_const = CollFreeConst(colkin, sdf, pr2)

        while True:
            base_pos = np.random.randn(3) * np.array([0.1, 0.2, 0.3]) - np.array([-0.05, 0.0, 0.0])
            set_robot_state(pr2, [], base_pos, base_type=BaseType.PLANER)
            colkin.reflect_skrobot_model(pr2)
            q = get_robot_state(pr2, colkin.control_joint_names)

            if collfree_const.is_valid(q):
                break

        descriptions = [(pose, base_pos) for pose in pose_list]
        return descriptions

    def export_intrinsic_descriptions(self) -> List[np.ndarray]:
        world_vec = self.world.export_intrinsic_description()

        intrinsic_descs = []
        for desc in self.descriptions:
            target_pose, base_pose = desc
            vecs = [world_vec] + [skcoords_to_pose_vec(target_pose)] + [base_pose]
            intrinsic_desc = np.hstack(vecs)
            intrinsic_descs.append(intrinsic_desc)
        return intrinsic_descs

    @staticmethod
    def create_cache(world: TabletopClutteredOvenWorld, robot_model: RobotModel) -> None:
        return None

    def export_table(self) -> DescriptionTable:
        world_dict = {}  # type: ignore
        assert False, "vector description is not implemented yet"
        # world_dict["vector"] = self.world.vector_description
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

    def export_problems(self) -> Iterator[Problem]:
        provider = self.config_provider
        q_start = provider.get_start_config()
        box_const = provider.get_box_const()

        sdf = self.world.get_exact_sdf()
        ineq_const = provider.get_collfree_const(sdf)

        pr2 = provider.get_pr2()

        for target_pose, base_pose in self.descriptions:
            set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)
            pose_const = provider.get_pose_const([target_pose])

            pose_const.reflect_skrobot_model(pr2)
            ineq_const.reflect_skrobot_model(pr2)

            problem = Problem(
                q_start, box_const, pose_const, ineq_const, None, motion_step_box_=0.05
            )
            yield problem

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        solcon = OMPLSolverConfig(n_max_call=40000, n_max_satisfaction_trial=200, simplify=True)

        ompl_solver = OMPLSolver.init(solcon)
        ompl_solver.setup(problem)
        return ompl_solver.solve()

    @overload
    def create_viewer(self, mode: Literal["static"]) -> StaticSolutionVisualizer:
        ...

    @overload
    def create_viewer(self, mode: Literal["interactive"]) -> InteractiveSolutionVisualizer:
        ...

    def create_viewer(self, mode: str) -> Any:
        cls: Type[SolutionVisualizerBase]
        if mode == "static":
            cls = StaticSolutionVisualizer
        elif mode == "interactive":
            cls = InteractiveSolutionVisualizer
        else:
            assert False

        assert len(self.descriptions) == 1
        target_co, base_pose = self.descriptions[0]
        geometries = [Axis.from_coords(target_co)]

        config = self.config_provider.get_config()  # type: ignore[attr-defined]
        pr2 = self.config_provider.get_pr2()  # type: ignore[attr-defined]
        set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)

        def robot_updator(robot, q):
            set_robot_state(pr2, config._get_control_joint_names(), q, config.base_type)

        obj = cls(pr2, geometry=geometries, visualizable=self.world, robot_updator=robot_updator)

        t = np.array(
            [
                [-0.74452768, 0.59385861, -0.30497620, -0.28438419],
                [-0.66678662, -0.68392597, 0.29604201, 0.80949977],
                [-0.03277405, 0.42376552, 0.90517879, 3.65387983],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        obj.viewer.camera_transform = t
        return obj
