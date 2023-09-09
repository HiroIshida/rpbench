from abc import abstractmethod
from dataclasses import dataclass, fields
from typing import (
    Any,
    ClassVar,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
from scipy.stats import lognorm
from skmp.constraint import CollFreeConst, IneqCompositeConst, PoseConstraint
from skmp.robot.jaxon import Jaxon
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import SatisfactionConfig
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver, MyRRTResult
from skmp.solver.nlp_solver.sqp_based_solver import (
    SQPBasedSolver,
    SQPBasedSolverConfig,
    SQPBasedSolverResult,
)
from skmp.visualization.solution_visualizer import (
    InteractiveSolutionVisualizer,
    SolutionVisualizerBase,
    StaticSolutionVisualizer,
)
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.model.robot_model import RobotModel
from skrobot.sdf.signed_distance_function import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType, RotationType

from rpbench.articulated.jaxon.common import CachedJaxonConstProvider
from rpbench.articulated.vision import HeightmapConfig, LocatedHeightmap
from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.interface import (
    DescriptionTable,
    Problem,
    ReachingTaskBase,
    ResultProtocol,
    WorldBase,
)
from rpbench.timeout_decorator import TimeoutError, timeout
from rpbench.utils import SceneWrapper, skcoords_to_pose_vec

BelowTableWorldT = TypeVar("BelowTableWorldT", bound="BelowTableWorldBase")


@dataclass
class BelowTableWorldBase(WorldBase):
    target_region: BoxSkeleton
    table: BoxSkeleton
    obstacles: List[BoxSkeleton]
    _intrinsic_desc: np.ndarray

    def get_exact_sdf(self) -> UnionSDF:
        return UnionSDF([self.table.sdf] + [obs.sdf for obs in self.obstacles])

    def export_intrinsic_description(self) -> np.ndarray:
        return self._intrinsic_desc

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        # self.target_region.visual_mesh.visual.face_colors = [255, 255, 255, 120]
        # viewer.add(self.target_region)
        viewer.add(self.table.to_visualizable())
        for obs in self.obstacles:
            skobs = obs.to_visualizable()
            # skobs.visual_mesh.visual.face_colors = [255, 0, 0, 150]
            skobs.visual_mesh.visual.face_colors = [0, 255, 0, 200]
            viewer.add(skobs)

    @staticmethod
    def sample_table_and_target_region(standard: bool = False) -> Tuple[BoxSkeleton, BoxSkeleton]:
        if standard:
            table_position = np.array([0.8, 0.0, 0.8])
        else:
            table_position = np.array([0.7, 0.0, 0.6]) + np.random.rand(3) * np.array(
                [0.5, 0.0, 0.4]
            )
        table = BoxSkeleton([1.0, 3.0, 0.1])
        table.translate(table_position)

        table_height = table_position[2]
        target_region = BoxSkeleton([0.8, 0.8, table_height])
        target_region.translate([0.6, -0.7, 0.5 * table_height])
        return table, target_region


@dataclass
class BelowTableSingleObstacleWorld(BelowTableWorldBase):
    @classmethod
    def sample(cls, standard: bool = False) -> "BelowTableSingleObstacleWorld":

        table, target_region = cls.sample_table_and_target_region(standard)
        table_position = table.worldpos()

        # determine obstacle
        if standard:
            obs = BoxSkeleton([0.1, 0.1, 0.5], pos=[0.6, -0.2, 0.25])
        else:
            region_width = np.array(target_region._extents[:2])
            region_center = target_region.worldpos()[:2]
            b_min = region_center - region_width * 0.5
            b_max = region_center + region_width * 0.5

            obs_width = np.random.rand(2) * np.ones(2) * 0.2 + np.ones(2) * 0.1
            obs_height = 0.3 + np.random.rand() * 0.5
            b_min = region_center - region_width * 0.5 + 0.5 * obs_width
            b_max = region_center + region_width * 0.5 - 0.5 * obs_width
            pos2d = np.random.rand(2) * (b_max - b_min) + b_min
            pos = np.hstack([pos2d, obs_height * 0.5])
            obs = BoxSkeleton(np.hstack([obs_width, obs_height]), pos=pos)

        return cls(target_region, table, [obs], table_position)


@dataclass
class BelowTableClutteredWorld(BelowTableWorldBase):
    _heightmap: Optional[np.ndarray] = None  # lazy

    @classmethod
    def sample(cls, standard: bool = False) -> "BelowTableClutteredWorld":
        table, target_region = cls.sample_table_and_target_region(standard)
        table_position = table.worldpos()

        # determine obstacle
        if standard:
            obs = BoxSkeleton([0.1, 0.1, 0.5], pos=[0.6, -0.2, 0.25])
            obstacles = [obs]
        else:
            obstacles = []

            n_obstacle = np.random.randint(8)
            for _ in range(n_obstacle):
                region_width = np.array(target_region._extents[:2])
                region_center = target_region.worldpos()[:2]

                obs_width = lognorm(s=0.5, scale=1.0).rvs(size=3) * np.array([0.2, 0.2, 0.4])
                obs_width[2] = min(obs_width[2], table_position[2] - 0.1)
                assert obs_width[2] < target_region.extents[2]

                b_min = region_center - region_width * 0.5 + 0.5 * obs_width[:2]
                b_max = region_center + region_width * 0.5 - 0.5 * obs_width[:2]
                pos2d = np.random.rand(2) * (b_max - b_min) + b_min
                pos = np.hstack([pos2d, obs_width[2] * 0.5])
                obs = BoxSkeleton(obs_width, pos=pos)
                obstacles.append(obs)

        return cls(target_region, table, obstacles, table_position)

    def heightmap(self) -> np.ndarray:
        if self._heightmap is None:
            hmap_config = HeightmapConfig(112, 112)
            hmap = LocatedHeightmap.by_raymarching(
                self.target_region, self.obstacles, conf=hmap_config
            )
            self._heightmap = hmap.heightmap
        return self._heightmap

    def __reduce__(self):
        args = []  # type: ignore
        for field in fields(self):
            if field.name == "_heightmap":
                # delete _heightmap cache for now.
                args.append(None)
            else:
                args.append(getattr(self, field.name))
        return (self.__class__, tuple(args))


class HumanoidTableReachingTaskBase(ReachingTaskBase[BelowTableWorldT, Jaxon]):
    config_provider: ClassVar[Type[CachedJaxonConstProvider]] = CachedJaxonConstProvider

    @staticmethod
    def get_robot_model() -> RobotModel:
        return CachedJaxonConstProvider.get_jaxon()

    @staticmethod
    def create_cache(world: BelowTableWorldT, robot_model: RobotModel) -> None:
        return None

    @classmethod
    def get_dof(cls) -> int:
        config = CachedJaxonConstProvider.get_config()
        return len(config._get_control_joint_names()) + 6

    @classmethod
    @abstractmethod
    def sample_target_poses(cls, world: BelowTableWorldT, standard: bool) -> Tuple[Coordinates]:
        ...

    @staticmethod
    @abstractmethod
    def rarm_rot_type() -> RotationType:
        ...

    @classmethod
    def sample_descriptions(
        cls, world: BelowTableWorldT, n_sample: int, standard: bool = False
    ) -> List[Tuple[Coordinates, ...]]:
        # TODO: duplication of tabletop.py
        if standard:
            assert n_sample == 1
        pose_list: List[Tuple[Coordinates, ...]] = []
        while len(pose_list) < n_sample:
            poses = cls.sample_target_poses(world, standard)
            is_valid_poses = True
            for pose in poses:
                position = np.expand_dims(pose.worldpos(), axis=0)
                if world.get_exact_sdf()(position)[0] < 1e-3:
                    is_valid_poses = False
            if is_valid_poses:
                pose_list.append(poses)
        return pose_list

    def export_problems(self) -> List[Problem]:
        provider = self.config_provider
        jaxon_config = provider.get_config()

        jaxon = provider.get_jaxon()

        # ineq const
        com_const = provider.get_com_const(jaxon)
        colkin = jaxon_config.get_collision_kin()
        colfree_const = CollFreeConst(
            colkin, self.world.get_exact_sdf(), jaxon, only_closest_feature=True
        )

        # the order of ineq const is important here. see comment in IneqCompositeConst
        ineq_const = IneqCompositeConst([com_const, colfree_const])

        q_start = get_robot_state(jaxon, jaxon_config._get_control_joint_names(), BaseType.FLOATING)
        box_const = provider.get_box_const()

        # eq const
        leg_coords_list = [jaxon.rleg_end_coords, jaxon.lleg_end_coords]
        efkin_legs = jaxon_config.get_endeffector_kin(rarm=False, larm=False)
        global_eq_const = PoseConstraint.from_skrobot_coords(leg_coords_list, efkin_legs, jaxon)  # type: ignore

        problems = []
        for desc in self.descriptions:
            goal_eq_const = provider.get_dual_legs_pose_const(
                jaxon, co_rarm=desc[0], arm_rot_type=self.rarm_rot_type()
            )

            problem = Problem(
                q_start,
                box_const,
                goal_eq_const,
                ineq_const,
                global_eq_const,
                motion_step_box_=jaxon_config.get_motion_step_box() * 0.5,
            )
            problems.append(problem)
        return problems

    def solve_default_each(self, problem: Problem) -> ResultProtocol:
        try:
            return self._solve_default_each(problem)
        except TimeoutError:
            print("timeout!! solved default failed.")
            return MyRRTResult.abnormal()

    @timeout(180)
    def _solve_default_each(self, problem: Problem) -> ResultProtocol:
        rrt_conf = MyRRTConfig(5000, satisfaction_conf=SatisfactionConfig(n_max_eval=20))

        sqp_config = SQPBasedSolverConfig(
            n_wp=40,
            n_max_call=20,
            motion_step_satisfaction="explicit",
            verbose=False,
            ctol_eq=1e-3,
            ctol_ineq=1e-3,
            ineq_tighten_coef=0.0,
        )

        for _ in range(4):
            rrt = MyRRTConnectSolver.init(rrt_conf)
            rrt.setup(problem)
            rrt_result = rrt.solve()

            if rrt_result.traj is not None:
                sqp = SQPBasedSolver.init(sqp_config)
                sqp.setup(problem)
                sqp_result = sqp.solve(rrt_result.traj)
                if sqp_result.traj is not None:
                    return sqp_result

        return SQPBasedSolverResult.abnormal()

    def export_intrinsic_descriptions(self) -> List[np.ndarray]:
        world_vec = self.world.export_intrinsic_description()

        intrinsic_descs = []
        for desc in self.descriptions:
            pose_vecs = [skcoords_to_pose_vec(pose) for pose in desc]
            vecs = [world_vec] + pose_vecs
            intrinsic_desc = np.hstack(vecs)
            intrinsic_descs.append(intrinsic_desc)
        return intrinsic_descs

    @overload
    def create_viewer(self, mode: Literal["static"]) -> StaticSolutionVisualizer:
        ...

    @overload
    def create_viewer(self, mode: Literal["interactive"]) -> InteractiveSolutionVisualizer:
        ...

    def create_viewer(self, mode: str) -> Any:
        assert len(self.descriptions) == 1
        target_co = self.descriptions[0][0]
        geometries = [Axis.from_coords(target_co)]

        config = self.config_provider.get_config()  # type: ignore[attr-defined]
        jaxon = self.config_provider.get_jaxon()
        colkin = config.get_collision_kin()
        sdf = self.world.get_exact_sdf()

        def robot_updator(robot, q):
            set_robot_state(robot, config._get_control_joint_names(), q, BaseType.FLOATING)

        cls: Type[SolutionVisualizerBase]
        if mode == "static":
            obj = StaticSolutionVisualizer(
                jaxon,
                geometry=geometries,
                visualizable=self.world,
                robot_updator=robot_updator,
                show_wireframe=False,
            )
        elif mode == "interactive":
            obj = InteractiveSolutionVisualizer(
                jaxon,
                geometry=geometries,
                visualizable=self.world,
                robot_updator=robot_updator,
                enable_colvis=True,
                colkin=colkin,
                sdf=sdf,
            )
        else:
            assert False

        # side
        # t = np.array([[ 0.989917  ,  0.0582524 , -0.12911616,  0.12857397],
        #     [-0.12970017,  0.00634257, -0.99153297, -3.63181173],
        #     [-0.05694025,  0.99828174,  0.01383397,  0.95512276],
        #     [ 0.        ,  0.        ,  0.        ,  1.        ]])
        t = np.array(
            [
                [0.57710263, 0.149482, -0.80287464, -2.3424832],
                [-0.81327753, 0.0156558, -0.58166533, -2.13427884],
                [-0.07437885, 0.98864049, 0.13060536, 1.38784946],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        obj.viewer.camera_transform = t
        return obj


class HumanoidTableNotClutteredReachingTaskBase(
    HumanoidTableReachingTaskBase[BelowTableSingleObstacleWorld]
):
    @staticmethod
    def get_world_type() -> Type[BelowTableSingleObstacleWorld]:
        return BelowTableSingleObstacleWorld

    def export_table(self) -> DescriptionTable:
        assert len(self.world.obstacles) == 1
        world_dict = {}
        world_dict["world"] = np.hstack(
            [
                self.world.table.worldpos(),
                self.world.obstacles[0].worldpos(),
                np.array(self.world.obstacles[0]._extents),
            ]
        )

        desc_dicts = []
        for desc in self.descriptions:
            desc_dict = {}
            for idx, co in enumerate(desc):
                pose = skcoords_to_pose_vec(co)
                name = "target_pose-{}".format(idx)
                desc_dict[name] = pose
            desc_dicts.append(desc_dict)
        return DescriptionTable(world_dict, desc_dicts)

    @classmethod
    def sample_target_poses(
        cls, world: BelowTableSingleObstacleWorld, standard: bool
    ) -> Tuple[Coordinates]:
        if standard:
            co = Coordinates([0.55, -0.6, 0.45], rot=[0, -0.5 * np.pi, 0])
            return (co,)

        sdf = world.get_exact_sdf()

        n_max_trial = 100
        ext = np.array(world.target_region._extents)
        for _ in range(n_max_trial):
            p_local = -0.5 * ext + np.random.rand(3) * ext
            co = world.target_region.copy_worldcoords()
            co.translate(p_local)
            points = np.expand_dims(co.worldpos(), axis=0)
            sd_val = sdf(points)[0]
            if sd_val > 0.03:
                co.rotate(-0.5 * np.pi, "y")
                return (co,)
        assert False


class HumanoidTableReachingTask(HumanoidTableNotClutteredReachingTaskBase):
    @staticmethod
    def rarm_rot_type() -> RotationType:
        return RotationType.XYZW


class HumanoidTableReachingTask2(HumanoidTableNotClutteredReachingTaskBase):
    @staticmethod
    def rarm_rot_type() -> RotationType:
        return RotationType.IGNORE


class HumanoidTableClutteredReachingTaskBase(
    HumanoidTableReachingTaskBase[BelowTableClutteredWorld]
):
    @staticmethod
    def get_world_type() -> Type[BelowTableClutteredWorld]:
        return BelowTableClutteredWorld

    def export_table(self) -> DescriptionTable:
        world_dict = {}
        world_dict["world_vector"] = self.world.table.worldpos()
        world_dict["world_mat"] = self.world.heightmap()

        desc_dicts = []
        for desc in self.descriptions:
            desc_dict = {}
            for idx, co in enumerate(desc):
                pose = skcoords_to_pose_vec(co)
                name = "target_pose-{}".format(idx)
                desc_dict[name] = pose
            desc_dicts.append(desc_dict)
        return DescriptionTable(world_dict, desc_dicts)

    @classmethod
    def sample_target_poses(
        cls, world: BelowTableClutteredWorld, standard: bool
    ) -> Tuple[Coordinates]:
        if standard:
            co = Coordinates([0.55, -0.6, 0.45], rot=[0, -0.5 * np.pi, 0])
            return (co,)

        sdf = world.get_exact_sdf()

        n_max_trial = 100
        ext = np.array(world.target_region._extents)
        for _ in range(n_max_trial):
            p_local = -0.5 * ext + np.random.rand(3) * ext
            co = world.target_region.copy_worldcoords()
            co.translate(p_local)
            points = np.expand_dims(co.worldpos(), axis=0)
            sd_val = sdf(points)[0]

            co_backward = co.copy_worldcoords()
            co_backward.translate([0, 0, 0.1])
            points = np.expand_dims(co_backward.worldpos(), axis=0)
            sd_val_back = sdf(points)[0]

            margin = 0.08
            if sd_val > margin and sd_val_back > margin:
                co.rotate(-0.5 * np.pi, "y")
                return (co,)

        # because no way to sample,
        co = Coordinates([0.55, -0.6, 0.45], rot=[0, -0.5 * np.pi, 0])
        return (co,)


class HumanoidTableClutteredReachingTask(HumanoidTableClutteredReachingTaskBase):
    @staticmethod
    def rarm_rot_type() -> RotationType:
        return RotationType.XYZW


class HumanoidTableClutteredReachingTask2(HumanoidTableClutteredReachingTaskBase):
    @staticmethod
    def rarm_rot_type() -> RotationType:
        return RotationType.IGNORE
