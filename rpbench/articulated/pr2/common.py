from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Callable, List, Type, TypeVar

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
from skmp.visualization.solution_visualizer import (
    InteractiveSolutionVisualizer,
    SolutionVisualizerBase,
    StaticSolutionVisualizer,
)
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2
from tinyfk import BaseType

from rpbench.interface import TaskBase


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


t = np.array(
    [
        [-0.74452768, 0.59385861, -0.30497620, -0.28438419],
        [-0.66678662, -0.68392597, 0.29604201, 0.80949977],
        [-0.03277405, 0.42376552, 0.90517879, 3.65387983],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


PR2SolutionViewerT = TypeVar("PR2SolutionViewerT", bound="PR2SolutionViewerBase")


class PR2SolutionViewerBase(SolutionVisualizerBase):
    @classmethod
    def from_task(cls: Type[PR2SolutionViewerT], task: TaskBase) -> PR2SolutionViewerT:
        assert len(task.descriptions) == 1
        geometries = []
        for co in task.descriptions[0]:
            geometries.append(Axis.from_coords(co))

        config: PR2Config = task.config_provider.get_config()  # type: ignore[attr-defined]
        pr2 = task.config_provider.get_pr2()  # type: ignore[attr-defined]

        def robot_updator(robot, q):
            set_robot_state(pr2, config._get_control_joint_names(), q, config.base_type)

        obj = cls(pr2, geometry=geometries, visualizable=task.world, robot_updator=robot_updator)
        obj.viewer.camera_transform = t
        return obj


class PR2StaticTaskVisualizer(StaticSolutionVisualizer, PR2SolutionViewerBase):
    ...


class PR2InteractiveTaskVisualizer(InteractiveSolutionVisualizer, PR2SolutionViewerBase):
    ...