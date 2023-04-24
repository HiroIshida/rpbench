import copy
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Generic,
    List,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from skmp.constraint import (
    AbstractIneqConst,
    BoxConst,
    CollFreeConst,
    IneqCompositeConst,
    PoseConstraint,
)
from skmp.kinematics import (
    ArticulatedCollisionKinematicsMap,
    ArticulatedEndEffectorKinematicsMap,
)
from skmp.robot.pr2 import PR2Config
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.trajectory import Trajectory
from skrobot.coordinates import Coordinates
from skrobot.model import RobotModel
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from rpbench.utils import SceneWrapper


class CachedPR2ConstProvider(ABC):
    """
    loading robot model is a process that takes some times.
    So, by utilizing classmethod with lru_cache, all program
    that calls this class share the same robot model and
    other stuff.
    """

    @classmethod
    @abstractmethod
    def get_config(cls) -> PR2Config:
        ...

    @classmethod
    @lru_cache
    def get_box_const(cls) -> BoxConst:
        config = cls.get_config()
        return config.get_box_const()

    @classmethod
    def get_pose_const(cls, target_pose_list: List[Coordinates]) -> PoseConstraint:
        config = cls.get_config()
        const = PoseConstraint.from_skrobot_coords(
            target_pose_list, config.get_endeffector_kin(), cls.get_pr2()
        )
        return const

    @classmethod
    def get_start_config(cls) -> np.ndarray:
        config = cls.get_config()
        pr2 = cls.get_pr2()
        angles = get_robot_state(pr2, config._get_control_joint_names(), base_type=config.base_type)
        return angles

    @classmethod
    @abstractmethod
    def get_collfree_const(cls, sdf: Callable[[np.ndarray], np.ndarray]) -> AbstractIneqConst:
        """get collision free constraint"""
        # make this method abstract because usually self-collision must be considerd
        # when dual-arm planning, but not have to be in single arm planning.
        ...

    @classmethod
    @lru_cache
    def get_pr2(cls) -> PR2:
        pr2 = PR2(use_tight_joint_limit=False)
        pr2.reset_manip_pose()
        return pr2

    @classmethod
    @lru_cache
    def get_efkin(cls) -> ArticulatedEndEffectorKinematicsMap:
        config = cls.get_config()
        return config.get_endeffector_kin()

    @classmethod
    @lru_cache
    def get_colkin(cls) -> ArticulatedCollisionKinematicsMap:
        config = cls.get_config()
        return config.get_collision_kin()

    @classmethod
    @lru_cache
    def get_dof(cls) -> int:
        config = cls.get_config()
        names = config._get_control_joint_names()
        dof = len(names)
        if config.base_type == BaseType.PLANER:
            dof += 3
        elif config.base_type == BaseType.FLOATING:
            dof += 6
        else:
            assert False
        return dof


class CachedRArmFixedPR2ConstProvider(CachedPR2ConstProvider):
    @classmethod
    def get_config(cls) -> PR2Config:
        return PR2Config(base_type=BaseType.FIXED)

    @classmethod
    def get_collfree_const(cls, sdf: Callable[[np.ndarray], np.ndarray]) -> CollFreeConst:
        colfree = CollFreeConst(cls.get_colkin(), sdf, cls.get_pr2())
        return colfree


class CachedRArmPR2ConstProvider(CachedPR2ConstProvider):
    @classmethod
    def get_config(cls) -> PR2Config:
        return PR2Config(base_type=BaseType.PLANER)

    @classmethod
    def get_collfree_const(cls, sdf: Callable[[np.ndarray], np.ndarray]) -> CollFreeConst:
        colfree = CollFreeConst(cls.get_colkin(), sdf, cls.get_pr2())
        return colfree


class CachedDualArmPR2ConstProvider(CachedPR2ConstProvider):
    @classmethod
    def get_config(cls) -> PR2Config:
        return PR2Config(base_type=BaseType.PLANER, control_arm="dual")

    @classmethod
    def get_collfree_const(cls, sdf: Callable[[np.ndarray], np.ndarray]) -> IneqCompositeConst:
        colfree = CollFreeConst(cls.get_colkin(), sdf, cls.get_pr2())
        selcolfree = cls.get_config().get_neural_selcol_const(cls.get_pr2())
        return IneqCompositeConst([colfree, selcolfree])


ViewerT = TypeVar("ViewerT", bound=Union[TrimeshSceneViewer, SceneWrapper])


class VisualizableWorld(Protocol):
    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        ...


class VisualizableTask(Protocol):
    config_provider: ClassVar[Type[CachedPR2ConstProvider]]
    descriptions: List[Tuple[Coordinates, ...]]

    @property
    def world(self) -> VisualizableWorld:
        ...

    #  https://github.com/python/mypy/issues/7041


class TaskVisualizerBase(Generic[ViewerT], ABC):
    # TODO: this class actually take any Task if it has config provider
    task: VisualizableTask
    viewer: ViewerT
    robot_model: RobotModel
    _show_called: bool

    def __init__(self, task: VisualizableTask):
        viewer = self.viewer_type()()

        robot_config = task.config_provider()
        robot_model = robot_config.get_pr2()
        viewer.add(robot_model)

        task.world.visualize(viewer)
        for desc in task.descriptions:
            for co in desc:
                axis = Axis.from_coords(co)
                viewer.add(axis)

        self.task = task
        self.viewer = viewer
        self.robot_model = robot_model
        self._show_called = False

    def update_robot_state(self, q: np.ndarray) -> None:
        robot_config_provider = self.task.config_provider()
        config = robot_config_provider.get_config()
        set_robot_state(self.robot_model, config._get_control_joint_names(), q, config.base_type)

    @classmethod
    @abstractmethod
    def viewer_type(cls) -> Type[ViewerT]:
        ...


class InteractiveTaskVisualizer(TaskVisualizerBase[TrimeshSceneViewer]):
    def show(self) -> None:
        self.viewer.show()
        time.sleep(1.0)
        self._show_called = True

    def visualize_trajectory(self, trajectory: Trajectory, t_interval: float = 0.6) -> None:
        assert self._show_called
        robot_config_provider = self.task.config_provider()

        q_end = trajectory.numpy()[-1]
        self.update_robot_state(q_end)

        robot_model = robot_config_provider.get_pr2()
        config = robot_config_provider.get_config()

        for q in trajectory.numpy()[:-1]:
            set_robot_state(robot_model, config._get_control_joint_names(), q, config.base_type)
            self.viewer.redraw()
            time.sleep(t_interval)

        print("==> Press [q] to close window")
        while not self.viewer.has_exit:
            time.sleep(0.1)
            self.viewer.redraw()

    @classmethod
    def viewer_type(cls) -> Type[TrimeshSceneViewer]:
        return TrimeshSceneViewer


class StaticTaskVisualizer(TaskVisualizerBase[SceneWrapper]):
    @classmethod
    def viewer_type(cls) -> Type[SceneWrapper]:
        return SceneWrapper

    def save_image(self, path: Union[Path, str]) -> None:
        if isinstance(path, str):
            path = Path(path)
        png = self.viewer.save_image(resolution=[640, 480], visible=True)
        with path.open(mode="wb") as f:
            f.write(png)

    def save_trajectory_image(self, trajectory: Trajectory, path: Union[Path, str]) -> None:
        # self.set_robot_alpha(self.robot_model, 30)

        robot_config_provider = self.task.config_provider()
        robot_model = robot_config_provider.get_pr2()

        config = robot_config_provider.get_config()

        for q in trajectory.numpy():
            robot_model_copied = copy.deepcopy(robot_model)
            set_robot_state(
                robot_model_copied, config._get_control_joint_names(), q, config.base_type
            )
            self.set_robot_alpha(robot_model_copied, 30)
            self.viewer.add(robot_model_copied)

        robot_model_copied = copy.deepcopy(robot_model)
        set_robot_state(
            robot_model_copied,
            config._get_control_joint_names(),
            trajectory.numpy()[-1],
            config.base_type,
        )
        self.viewer.add(robot_model_copied)

        if isinstance(path, str):
            path = Path(path)
        png = self.viewer.save_image(resolution=[640, 480], visible=True)
        with path.open(mode="wb") as f:
            f.write(png)

    @staticmethod
    def set_robot_alpha(robot: RobotModel, alpha: int):
        assert alpha < 256
        for link in robot.link_list:
            visual_mesh = link.visual_mesh
            if isinstance(visual_mesh, list):
                for mesh in visual_mesh:
                    mesh.visual.face_colors[:, 3] = alpha
            else:
                visual_mesh.visual.face_colors[:, 3] = alpha