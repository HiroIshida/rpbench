import copy
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, List, Protocol, Tuple, Type, TypeVar, Union

import imageio
import numpy as np
from skmp.robot.utils import set_robot_state
from skmp.trajectory import Trajectory
from skrobot.coordinates import Coordinates
from skrobot.model import RobotModel
from skrobot.model.primitives import Axis
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from rpbench.articulated.jaxon.below_table import CachedJaxonConstProvider
from rpbench.utils import SceneWrapper

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

        for q in trajectory.numpy()[:-1]:
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
