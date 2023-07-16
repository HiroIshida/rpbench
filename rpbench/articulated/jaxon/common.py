import copy
import tempfile
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import (
    ClassVar,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import imageio
import numpy as np
from skmp.constraint import BoxConst, COMStabilityConst, PoseConstraint, EqCompositeConst
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import set_robot_state
from skmp.trajectory import Trajectory
from skrobot.coordinates import Coordinates
from skrobot.model import RobotModel
from skrobot.model.primitives import Axis, Box
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType, RotationType

from rpbench.utils import SceneWrapper


class CachedJaxonConstProvider(ABC):
    @classmethod
    def get_config(cls) -> JaxonConfig:
        return JaxonConfig()

    @classmethod
    @lru_cache
    def get_jaxon(cls) -> Jaxon:
        jaxon = Jaxon()
        jaxon.reset_manip_pose()
        jaxon.translate([0.0, 0.0, 0.98])
        return jaxon

    @classmethod
    @lru_cache
    def get_box_const(cls) -> BoxConst:
        config = cls.get_config()
        return config.get_box_const()

    @classmethod
    def get_dual_legs_pose_const(
        cls,
        jaxon: Jaxon,
        co_rarm: Optional[Coordinates] = None,
        co_larm: Optional[Coordinates] = None,
        arm_rot_type: RotationType = RotationType.XYZW
    ) -> Union[PoseConstraint, EqCompositeConst]:
        config = cls.get_config()

        # TODO: for simplicity, split the kinematics solver to legs and arms
        # because leg rot-type are always XYZW but arm rot-type can be IGNORE

        leg_efkin = config.get_endeffector_kin(rleg=True, lleg=True, rarm=False, larm=False)
        leg_coords_list = [jaxon.rleg_end_coords, jaxon.lleg_end_coords]
        leg_const = PoseConstraint.from_skrobot_coords(leg_coords_list, leg_efkin, jaxon)  # type: ignore

        use_rarm = co_rarm is not None
        use_larm = co_larm is not None
        if not use_rarm and not use_larm:
            return leg_const

        arm_efkin = config.get_endeffector_kin(rleg=False, lleg=False, rarm=use_rarm, larm=use_larm, rot_type=arm_rot_type)
        arm_coords_list = []
        if use_rarm:
            arm_coords_list.append(co_rarm)
        if use_larm:
            arm_coords_list.append(co_larm)
        arm_const = PoseConstraint.from_skrobot_coords(arm_coords_list, arm_efkin, jaxon)  # type: ignore
        const = EqCompositeConst([leg_const, arm_const])
        return const

    @classmethod
    def get_com_const(cls, jaxon: Jaxon) -> COMStabilityConst:
        # TODO: the following com box computation assums that legs is aligned with x-axis
        # also, assumes that both legs has the same x coordinate
        ym = jaxon.rleg_end_coords.worldpos()[1]
        yp = jaxon.lleg_end_coords.worldpos()[1]
        com_box = Box([0.25, yp - ym + 0.14, 5.0], with_sdf=True)

        com_box.visual_mesh.visual.face_colors = [255, 0, 100, 100]
        config = cls.get_config()
        return config.get_com_stability_const(jaxon, com_box)


ViewerT = TypeVar("ViewerT", bound=Union[TrimeshSceneViewer, SceneWrapper])


class VisualizableWorld(Protocol):
    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        ...


class VisualizableTask(Protocol):
    config_provider: ClassVar[Type[CachedJaxonConstProvider]]
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
        robot_model = robot_config.get_jaxon()
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
        t = np.array(
            [
                [9.93795996e-01, -1.49563989e-02, -1.10208097e-01, 1.67460132e-04],
                [-1.10181461e-01, 2.59590094e-03, -9.93908098e-01, -4.49870487e00],
                [1.51513753e-02, 9.99884777e-01, 9.31878081e-04, 9.16509762e-01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        self.viewer.camera_transform = t

    def update_robot_state(self, q: np.ndarray) -> None:
        robot_config_provider = self.task.config_provider()
        config = robot_config_provider.get_config()
        set_robot_state(self.robot_model, config._get_control_joint_names(), q, BaseType.FLOATING)

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

        robot_model = robot_config_provider.get_jaxon()
        config = robot_config_provider.get_config()

        for q in trajectory.numpy():
            set_robot_state(robot_model, config._get_control_joint_names(), q, BaseType.FLOATING)
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

    def save_trajectory_gif(self, trajectory: Trajectory, path: Union[Path, str]) -> None:
        robot_config_provider = self.task.config_provider()
        robot_model = robot_config_provider.get_jaxon()

        config = robot_config_provider.get_config()

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            file_path_list = []

            for i, q in enumerate(trajectory.numpy()):
                print(q)
                set_robot_state(
                    robot_model, config._get_control_joint_names(), q, BaseType.FLOATING
                )
                self.viewer.redraw()
                time.sleep(0.5)
                file_path = td_path / "{}.png".format(i)
                file_path_list.append(file_path)
                self.save_image(file_path)

            images = []
            for file_path in file_path_list:
                images.append(imageio.imread(file_path))
            for _ in range(10):
                images.append(imageio.imread(file_path_list[-1]))
            imageio.mimsave(path, images)

    def save_trajectory_image(self, trajectory: Trajectory, path: Union[Path, str]) -> None:
        # self.set_robot_alpha(self.robot_model, 30)
        robot_config_provider = self.task.config_provider()
        robot_model = robot_config_provider.get_jaxon()

        config = robot_config_provider.get_config()

        for q in trajectory.numpy():
            robot_model_copied = copy.deepcopy(robot_model)
            set_robot_state(
                robot_model_copied, config._get_control_joint_names(), q, BaseType.FLOATING
            )
            self.set_robot_alpha(robot_model_copied, 30)
            self.viewer.add(robot_model_copied)

        robot_model_copied = copy.deepcopy(robot_model)
        set_robot_state(
            robot_model_copied,
            config._get_control_joint_names(),
            trajectory.numpy()[-1],
            BaseType.FLOATING,
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
