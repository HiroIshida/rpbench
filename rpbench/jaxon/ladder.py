from dataclasses import dataclass
from typing import List, Type, Union

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.sdf import UnionSDF
from skrobot.viewers import TrimeshSceneViewer
from voxbloxpy.core import Grid

from rpbench.utils import SceneWrapper


class LadderWorld:
    @dataclass
    class Parameter:
        ladder_width: float = 0.5
        ladder_step_interval: float = 0.3
        ladder_step_depth: float = 0.2
        ladder_step_x_translate: float = 0.05
        ladder_step_height: float = 0.03
        angle: float = -0.3

        @classmethod
        def create(cls, vector: np.ndarray) -> "LadderWorld.Parameter":
            assert len(vector) == 6
            return cls(*list(vector))

    param: Parameter
    poles: List[Box]
    steps: List[Box]

    def __init__(self, param: Parameter = Parameter()):
        length = 4.0
        poles = []
        pole_right = Box([0.05, 0.05, length], with_sdf=True)
        pole_right.translate([0, -0.5 * param.ladder_width, length * 0.3])
        pole_left = Box([0.05, 0.05, length], with_sdf=True)
        pole_left.translate([0, +0.5 * param.ladder_width, length * 0.3])
        pole_right.assoc(pole_left)
        poles = [pole_right, pole_left]

        # add steps
        steps = []

        step_dims = [param.ladder_step_depth, param.ladder_width, param.ladder_step_height]
        step = Box(step_dims, with_sdf=True)
        step.translate([param.ladder_step_x_translate, 0.0, 0.0])
        pole_right.assoc(step)
        steps.append(step)

        for i in range(9):
            step = Box(step_dims, with_sdf=True)
            step.translate(
                [param.ladder_step_x_translate, 0.0, param.ladder_step_interval * (i + 1)]
            )
            pole_right.assoc(step)
            steps.append(step)

        for i in range(2):
            step = Box(step_dims, with_sdf=True)
            step.translate(
                [param.ladder_step_x_translate, 0.0, -param.ladder_step_interval * (i + 1)]
            )
            pole_right.assoc(step)
            steps.append(step)

        pole_right.rotate(-param.angle, axis="y")
        for step in steps:
            step.rotate(+param.angle, axis="y", wrt="local")

        self.param = param
        self.poles = poles
        self.steps = steps

    @classmethod
    def sample(cls: Type["LadderWorld"], standard: bool = False) -> "LadderWorld":
        if standard:
            return cls()
        else:
            assert False

    def get_exact_sdf(self) -> UnionSDF:
        lst = []
        for pole in self.poles:
            lst.append(pole.sdf)
        for step in self.steps:
            lst.append(step.sdf)
        return UnionSDF(lst)

    def get_grid(self) -> Grid:
        raise NotImplementedError("girdsdf is not used")

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        # add origin
        for pole in self.poles:
            viewer.add(pole)
        for step in self.steps:
            viewer.add(step)

    def first_axes(self) -> List[Axis]:
        step = self.steps[0]
        depth, _, height = step._extents

        rleg_pose: Coordinates = step.copy_worldcoords()
        rleg_pose.translate([-depth * 0.5 + 0.01, -0.12, height * 0.5])
        ax1 = Axis.from_coords(rleg_pose)

        lleg_pose: Coordinates = step.copy_worldcoords()
        lleg_pose.translate([-depth * 0.5 + 0.01, 0.12, height * 0.5])
        ax2 = Axis.from_coords(lleg_pose)

        rarm_pose: Coordinates = self.poles[0].copy_worldcoords()
        rarm_pose.translate([0.0, 0.0, 0.3], wrt="local")

        larm_pose: Coordinates = rarm_pose.copy_worldcoords()
        larm_pose.translate([0.0, +self.param.ladder_width, 0.0], wrt="local")

        rarm_pose.rotate(np.pi * 0.5, "x", wrt="local")
        rarm_pose.rotate(-np.pi * 0.3, "y", wrt="local")
        ax3 = Axis.from_coords(rarm_pose)

        larm_pose.rotate(-np.pi * 0.5, "x", wrt="local")
        larm_pose.rotate(-np.pi * 0.3, "y", wrt="local")
        ax4 = Axis.from_coords(larm_pose)
        return [ax1, ax2, ax3, ax4]

    def post_first_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self.first_axes()
        pose_rleg = ax_rleg.copy_worldcoords()
        pose_rleg.translate([0, 0, 0.05])
        return [Axis.from_coords(pose_rleg), ax_lleg, ax_rarm, ax_larm]

    def second_axes(self) -> List[Axis]:
        desired_trans = self.desired_translation_by_step()
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self.first_axes()
        pose_rleg = ax_rleg.copy_worldcoords()
        pose_rleg.translate(desired_trans)
        return [Axis.from_coords(pose_rleg), ax_lleg, ax_rarm, ax_larm]

    def pre_second_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self.second_axes()
        pose_rleg = ax_rleg.copy_worldcoords()
        pose_rleg.translate([0, 0, 0.05])
        return [Axis.from_coords(pose_rleg), ax_lleg, ax_rarm, ax_larm]

    def post_second_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self.second_axes()
        pose_lleg = ax_lleg.copy_worldcoords()
        pose_lleg.translate([0, 0, 0.05])
        return [ax_rleg, Axis.from_coords(pose_lleg), ax_rarm, ax_larm]

    def third_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self.second_axes()
        pose_lleg = ax_lleg.copy_worldcoords()
        desired_trans = self.desired_translation_by_step()
        pose_lleg.translate(desired_trans)
        return [ax_rleg, Axis.from_coords(pose_lleg), ax_rarm, ax_larm]

    def pre_third_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self.third_axes()
        pose_lleg = ax_lleg.copy_worldcoords()
        pose_lleg.translate([0, 0, 0.05])
        return [ax_rleg, Axis.from_coords(pose_lleg), ax_rarm, ax_larm]

    def desired_translation_by_step(self) -> np.ndarray:
        step0 = self.steps[0]
        step1 = self.steps[1]
        diff = step1.worldpos() - step0.worldpos()
        return diff
