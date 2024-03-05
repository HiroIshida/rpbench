import copy
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
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
from skmp.constraint import CollFreeConst
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig
from skmp.visualization.solution_visualizer import (
    InteractiveSolutionVisualizer,
    SolutionVisualizerBase,
    StaticSolutionVisualizer,
)
from skrobot.coordinates import CascadedCoords, Coordinates, rpy_angle
from skrobot.model.primitives import Axis
from skrobot.model.robot_model import RobotModel
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from rpbench.articulated.pr2.common import (
    CachedPR2ConstProvider,
    CachedRArmFixedPR2ConstProvider,
    CachedRArmPR2ConstProvider,
)
from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.utils import (
    BoxSkeleton,
    CylinderSkelton,
    PrimitiveSkelton,
)
from rpbench.interface import (
    Problem,
    ResultProtocol,
    TaskExpression,
    VisualizableTaskBase,
)
from rpbench.planer_box_utils import Box2d, Circle, PlanerCoords, is_colliding
from rpbench.utils import (
    SceneWrapper,
    lru_cache_keeping_random_state,
    skcoords_to_pose_vec,
)

PR2MiniFridgeTaskT = TypeVar("PR2MiniFridgeTaskT", bound="PR2MiniFridgeTaskBase")


_N_MAX_OBSTACLE = 10


class Fridge(CascadedCoords):
    panels: Dict[str, BoxSkeleton]
    table: BoxSkeleton
    angle: float
    target_region: BoxSkeleton
    joint: CascadedCoords

    def __init__(self, angle: float):
        size = np.array([0.6, 0.6, 0.4])
        thickness = 0.03
        CascadedCoords.__init__(self)
        d, w, h = size
        plane_xaxis = BoxSkeleton([thickness, w, h])
        plane_yaxis = BoxSkeleton([d, thickness, h])
        plane_zaxis = BoxSkeleton([d, w, thickness])

        bottom = copy.deepcopy(plane_zaxis)
        bottom.translate([0, 0, 0.5 * thickness])
        self.assoc(bottom, relative_coords="local")

        top = copy.deepcopy(plane_zaxis)
        top.translate([0, 0, h - 0.5 * thickness])
        self.assoc(top, relative_coords="local")

        right = copy.deepcopy(plane_yaxis)
        right.translate([0, -0.5 * w + 0.5 * thickness, 0.5 * h])
        self.assoc(right, relative_coords="local")

        left = copy.deepcopy(plane_yaxis)
        left.translate([0, +0.5 * w - 0.5 * thickness, 0.5 * h])
        self.assoc(left, relative_coords="local")

        back = copy.deepcopy(plane_xaxis)
        back.translate([0.5 * d - 0.5 * thickness, 0.0, 0.5 * h])
        self.assoc(back, relative_coords="local")

        joint = CascadedCoords()
        joint.translate([-0.5 * d, -0.5 * w, 0.0])
        self.assoc(joint, relative_coords="local")
        self.joint = joint

        door = copy.deepcopy(plane_xaxis)
        door.translate([-0.5 * d + 0.5 * thickness, 0.0, 0.5 * h])
        self.joint.assoc(door, relative_coords="world")
        self.joint.rotate(angle, [0, 0, 1.0])

        self.panels = {
            "bottom": bottom,
            "top": top,
            "right": right,
            "left": left,
            "back": back,
            "door": door,
        }

        target_region = BoxSkeleton(size - 2 * thickness, [0, 0, 0.5 * h])
        self.assoc(target_region, relative_coords="local")

        # prepare table. table is floating. PR2 is a robot that has huge
        # footprint. So, if the table is not floating, PR2 will easily collide
        # with the table and most of reaching problem turns to infeasible.
        # so, we prepare the table as floating.
        slide_x = 0.75
        slide_y = -0.1
        table_height = 0.8
        table_size = np.array([0.6, 3.0, 0.03])
        table = BoxSkeleton(table_size)
        table.translate(np.array([slide_x, slide_y, table_height]))
        self.table = table
        self.translate([slide_x, slide_y, table_height])

        self.angle = angle
        self.target_region = target_region

    def set_joint_angle(self, angle: float) -> None:
        current_angle = self.angle
        diff = angle - current_angle
        self.joint.rotate(diff, [0, 0, 1.0])
        self.angle = angle

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        for panel in self.panels.values():
            viewer.add(panel.to_visualizable((255, 0, 0, 150)))
        viewer.add(self.table.to_visualizable((0, 255, 0, 150)))

    def get_exact_sdf(self) -> UnionSDF:
        # sdf = UnionSDF([p.sdf for p in self.panels.values()] + [self.table.sdf])
        sdf = UnionSDF([p.sdf for p in self.panels.values()])
        return sdf


class MiniFridgeWorld:
    fridge: Fridge
    contents: List[PrimitiveSkelton]

    def __init__(self, fridge: Fridge, contents: List[PrimitiveSkelton]):
        super().__init__()
        self.fridge = fridge
        self.contents = contents

    def to_parameter(self) -> np.ndarray:
        # serialize this object to compact vector representation
        region_pos = self.fridge.target_region.worldpos()

        dof_per_cylinder = 4  # x, y, h, r
        dof_per_box = 6  # x, y, yaw, w, d, h
        dof_per_obj = 1 + max(dof_per_cylinder, dof_per_box)  # +1 for type number
        param = np.zeros(_N_MAX_OBSTACLE * dof_per_obj + 1)  # +1 for fridge door angle
        head = 0
        for obj in self.contents:
            if isinstance(obj, BoxSkeleton):
                type_num = 0
                x, y = (obj.worldpos() - region_pos)[:2]
                yaw, _, _ = rpy_angle(obj.worldrot())[0]
                w, d, h = obj.extents
                obj_param = np.array([type_num, x, y, yaw, w, d, h])
            elif isinstance(obj, CylinderSkelton):
                type_num = 1
                x, y = (obj.worldpos() - region_pos)[:2]
                h = obj.height
                r = obj.radius
                obj_param = np.array([type_num, x, y, h, r, 0.0, 0.0])
            else:
                assert False
            param[head : head + dof_per_obj] = obj_param
            head += dof_per_obj
        param[-1] = self.fridge.angle
        return param

    @classmethod
    def from_parameter(cls, param: np.ndarray) -> "MiniFridgeWorld":
        angle = param[-1]
        fridge = Fridge(angle)
        H = fridge.target_region.extents[2]
        contents: List[PrimitiveSkelton] = []
        head = 0
        while head < len(param) - 1:
            type_num = int(param[head])
            if type_num == 0:
                x, y, yaw, w, d, h = param[head + 1 : head + 7]
                obj = BoxSkeleton(np.array([w, d, h]), pos=np.array([x, y, -0.5 * H + 0.5 * h]))
                obj.rotate(yaw, "z")
            elif type_num == 1:
                x, y, h, r = param[head + 1 : head + 5]
                obj = CylinderSkelton(r, h, pos=np.array([x, y, -0.5 * H + 0.5 * h]))
            else:
                assert False
            contents.append(obj)
            head += 7
            fridge.target_region.assoc(obj, relative_coords="local")
        return cls(fridge, contents)

    def create_heightmap(self, n_grid: int = 56) -> np.ndarray:
        hmap = create_heightmap_z_slice(self.fridge.target_region, self.contents, n_grid)
        return hmap

    def get_exact_sdf(self) -> UnionSDF:
        fridge_sdf = self.fridge.get_exact_sdf()
        sdfs = [c.sdf for c in self.contents] + [fridge_sdf]
        sdf = UnionSDF(sdfs)
        return sdf

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        self.fridge.visualize(viewer)
        for obj in self.contents:
            viewer.add(obj.to_visualizable((0, 0, 255, 150)))

    @classmethod
    def get_world_dof(cls) -> int:
        return 7 * _N_MAX_OBSTACLE + 1


class PR2MiniFridgeTaskBase(VisualizableTaskBase):
    vector_param: Tuple[Coordinates, np.ndarray]
    world: MiniFridgeWorld

    def __init__(
        self, vector_param: Tuple[Coordinates, np.ndarray], minfridge_world: MiniFridgeWorld
    ):
        self.vector_param = vector_param
        self.world = minfridge_world

    @classmethod
    @abstractmethod
    def get_config_provider(cls) -> Type[CachedPR2ConstProvider]:
        ...

    @classmethod
    def get_robot_model(cls) -> RobotModel:
        pr2 = cls.get_config_provider().get_pr2()
        pr2.l_shoulder_lift_joint.joint_angle(-0.3)
        return pr2

    @classmethod
    def sample(
        cls: Type[PR2MiniFridgeTaskT],
        standard: bool = False,
        predicate: Optional[Callable[[PR2MiniFridgeTaskT], bool]] = None,
        timeout: int = 180,
    ) -> PR2MiniFridgeTaskT:
        # sample base pos first and then sample the world
        provider = cls.get_config_provider()
        config = provider.get_config()

        while True:
            fridge = cls.get_empty_fridge()
            if standard:
                angle = 140 * (np.pi / 180.0)
            else:
                angle = (45 + np.random.rand() * 90) * (np.pi / 180.0)
            fridge.set_joint_angle(angle)

            # determine base pos
            if standard:
                base_pos = np.array([-0.1, 0.0, 0.0])
            else:
                xpos = np.random.uniform(-0.2, 0.2)
                ypos = np.random.uniform(-0.4, 0.6)
                yaw = np.random.uniform(-0.3, 0.3)
                base_pos = np.array([xpos, ypos, yaw])
            pr2 = cls.get_robot_model()
            set_robot_state(pr2, [], base_pos, base_type=BaseType.PLANER)
            colkin = provider.get_whole_body_colkin()
            colkin.reflect_skrobot_model(pr2)
            q = get_robot_state(pr2, colkin.control_joint_names, base_type=config.base_type)

            sdf = fridge.get_exact_sdf()
            collfree_const = CollFreeConst(colkin, sdf, pr2)
            if not collfree_const.is_valid(q):
                continue

            # determine target pose
            target_region = fridge.target_region
            pos = target_region.sample_points(1, margin=0.05)[0]

            pose = Coordinates(pos)
            yaw = np.random.uniform(-0.3 * np.pi, 0.3 * np.pi)
            pose.rotate(yaw, "z")
            pose.rotate(0.5 * np.pi, "x")

            # fridge = copy.deepcopy(cls.get_empty_fridge())
            # world = MiniFridgeWorld(fridge, [])
            # return cls((pose, base_pos), world)
            # # determine the world
            world = cls.sample_fridge(pose)
            if world is None:
                continue
            return cls((pose, base_pos), world)

    @classmethod
    def sample_fridge(cls, target_pose) -> Optional[MiniFridgeWorld]:
        trans_list = [
            (0.0, 0.0, 0.0),
            (0.03, 0.06, 0.0),
            (0.03, -0.06, 0.0),
            (-0.06, 0.0, 0.0),
            (-0.12, 0.0, 0.0),
        ]
        check_pos_list = []
        for trans in trans_list:
            co = target_pose.copy_worldcoords()
            co.translate(trans)
            check_pos_list.append(co.worldpos())
        P_check = np.array(check_pos_list)
        radius_list = [0.03, 0.03, 0.03, 0.03, 0.03]

        fridge = copy.deepcopy(cls.get_empty_fridge())
        if np.any(fridge.target_region.sdf(P_check) > -np.array(radius_list)):
            return None

        n_obstacles = np.random.randint(1, _N_MAX_OBSTACLE + 1)

        target_region = fridge.target_region
        D, W, H = target_region._extents
        obstacle_h_max = H - 0.02
        obstacle_h_min = 0.1

        region2d = Box2d(np.array([D, W]), PlanerCoords.standard())

        obj2d_list = []
        object_list = []
        while len(object_list) < n_obstacles:
            center = region2d.sample_point()
            sample_circle = np.random.rand() < 0.5
            h = np.random.rand() * (obstacle_h_max - obstacle_h_min) + obstacle_h_min
            if sample_circle:
                r = np.random.rand() * 0.05 + 0.02
                obj2d = Circle(center, r)
                obj = CylinderSkelton(obj2d.radius, h, pos=np.hstack([obj2d.center, 0.0]))
            else:
                w = np.random.uniform(0.05, 0.20)
                d = np.random.uniform(0.05, 0.20)
                yaw = np.random.uniform(0.0, np.pi)
                obj2d = Box2d(np.array([w, d]), PlanerCoords(center, yaw))  # type: ignore

                extent = np.hstack([obj2d.extent, h])
                obj = BoxSkeleton(extent, pos=np.hstack([obj2d.coords.pos, 0.0]))
                obj.rotate(obj2d.coords.angle, "z")
            obj.translate([0.0, 0.0, -0.5 * H + 0.5 * h])

            if not region2d.contains(obj2d):
                continue
            if any([is_colliding(obj2d, o) for o in obj2d_list]):
                continue

            # needed to check collision
            target_region.assoc(obj, relative_coords="local")

            # if obj collides with any checking spheres, return None
            # not continue because we need to reject the pose
            # use obj.sdf and radius_list to check collision
            sdvals = obj.sdf(P_check)
            if np.any(sdvals < np.array(radius_list)):
                target_region.dissoc(obj)
                return None

            obj2d_list.append(obj2d)
            object_list.append(obj)
        return MiniFridgeWorld(fridge, object_list)

    @staticmethod
    @lru_cache_keeping_random_state
    def get_empty_fridge() -> Fridge:
        # cache this as fridge will be used many times in sampling
        fridge = Fridge(2.0)
        return fridge

    def export_task_expression(self, use_matrix: bool) -> TaskExpression:
        if use_matrix:
            world_vec = np.array([self.world.fridge.angle])
            world_mat = self.world.create_heightmap()
        else:
            world_vec = self.world.to_parameter()
            world_mat = None

        target_pose, init_pose = self.vector_param
        other_vec = np.hstack([skcoords_to_pose_vec(target_pose, yaw_only=True), init_pose])
        return TaskExpression(world_vec, world_mat, other_vec)

    @classmethod
    def from_task_param(cls: Type[PR2MiniFridgeTaskT], param: np.ndarray) -> PR2MiniFridgeTaskT:
        world_type = MiniFridgeWorld
        world_param_dof = world_type.get_world_dof()
        world_param = param[:world_param_dof]
        world = world_type.from_parameter(world_param)

        other_param = param[world_param_dof:]
        pose_param = other_param[:4]
        ypr = (pose_param[3], 0, 0)
        co = Coordinates(pose_param[:3], ypr)
        base_param = other_param[4:]
        vector_param = (co, base_param)
        return cls(vector_param, world)

    def export_problem(self) -> Problem:
        provider = self.get_config_provider()
        config = provider.get_config()
        q_start = provider.get_start_config()

        sdf = self.world.get_exact_sdf()
        if config.base_type == BaseType.PLANER:
            ineq_const = provider.get_collfree_const(sdf, whole_body=True)
            # base movement makes motion step larger...
            motion_step_box = provider.get_config().get_default_motion_step_box() * 0.5
        else:
            ineq_const = provider.get_collfree_const(sdf)
            motion_step_box = provider.get_config().get_default_motion_step_box()

        target_pose, base_pose = self.vector_param
        if config.base_type == BaseType.PLANER:
            lb = base_pose - np.array([0.5, 0.5, 0.5])
            ub = base_pose + np.array([0.5, 0.5, 0.5])
            base_bound = (tuple(lb), tuple(ub))
            q_start[-3:] = base_pose  # dirty...
        else:
            base_bound = None
        box_const = provider.get_box_const(base_bound=base_bound)

        pose_const = provider.get_pose_const([target_pose])

        # set pr2 to the initial state
        # NOTE: because provider cache's pr2 state and when calling any function
        # it reset the pr2 state to the original state. So the following
        # two lines must be placed here right before reflecting the model
        pr2 = self.get_robot_model()
        set_robot_state(pr2, config.get_control_joint_names(), q_start, base_type=config.base_type)
        if config.base_type == BaseType.FIXED:
            set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)
        pose_const.reflect_skrobot_model(pr2)
        ineq_const.reflect_skrobot_model(pr2)

        problem = Problem(
            q_start, box_const, pose_const, ineq_const, None, motion_step_box_=motion_step_box
        )
        return problem

    def solve_default(self) -> ResultProtocol:
        problem = self.export_problem()
        solcon = OMPLSolverConfig(
            n_max_call=10000000000, n_max_satisfaction_trial=10000000, simplify=True, timeout=30
        )
        ompl_solver = OMPLSolver.init(solcon)
        ompl_solver.setup(problem)
        return ompl_solver.solve()

    @classmethod
    def get_task_dof(cls) -> int:
        return cls.get_world_type().get_world_dof() + 4 + 3  # type: ignore[attr-defined]

    @overload
    def create_viewer(self, mode: Literal["static"]) -> StaticSolutionVisualizer:
        ...

    @overload
    def create_viewer(self, mode: Literal["interactive"]) -> InteractiveSolutionVisualizer:
        ...

    def create_viewer(self, mode: str) -> Any:
        target_co, base_pose = self.vector_param
        geometries = [Axis.from_coords(target_co)]
        provider = self.get_config_provider()
        config = provider.get_config()
        # pr2 = provider.get_pr2()  # type: ignore[attr-defined]
        pr2 = self.get_robot_model()
        set_robot_state(pr2, [], base_pose, base_type=BaseType.PLANER)

        def robot_updator(robot, q):
            set_robot_state(pr2, config._get_control_joint_names(), q, config.base_type)

        if mode == "static":
            obj: SolutionVisualizerBase = StaticSolutionVisualizer(
                pr2,
                geometry=geometries,
                visualizable=self.world,
                robot_updator=robot_updator,
                show_wireframe=True,
            )
        elif mode == "interactive":
            obj = InteractiveSolutionVisualizer(
                pr2, geometry=geometries, visualizable=self.world, robot_updator=robot_updator
            )
        else:
            assert False

        t = np.array(
            [
                [-0.74452768, 0.59385861, -0.30497620, -0.28438419],
                [-0.66678662, -0.68392597, 0.29604201, 0.80949977],
                [-0.03277405, 0.42376552, 0.90517879, 3.65387983],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # the below for specifically for visualizing the container contents
        # t = np.array([[-0.00814724,  0.72166326, -0.69219633, -0.01127641],
        #               [-0.99957574,  0.01348003,  0.02581901,  0.06577777],
        #               [ 0.02796346,  0.69211302,  0.72124726,  1.52418492],
        #               [ 0.        ,  0.        ,  0.        ,  1.        ]])

        obj.viewer.camera_transform = t
        return obj


class FixedPR2MiniFridgeTask(PR2MiniFridgeTaskBase):
    @classmethod
    def get_config_provider(cls) -> Type[CachedPR2ConstProvider]:
        return CachedRArmFixedPR2ConstProvider


class PR2MiniFridgeTask(PR2MiniFridgeTaskBase):
    @classmethod
    def get_config_provider(cls) -> Type[CachedPR2ConstProvider]:
        return CachedRArmPR2ConstProvider
