from dataclasses import dataclass
from typing import List, Literal, Tuple, Type, Union

import numpy as np
from skmp.constraint import (
    AbstractEqConst,
    AbstractIneqConst,
    CollFreeConst,
    IneqCompositeConst,
    PoseConstraint,
)
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.sdf import UnionSDF
from skrobot.utils.urdf import mesh_simplify_factor
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


@dataclass
class ConstraintSequence:
    world: LadderWorld

    def __getitem__(self, index):
        modes = ["first", "post_first", "pre_second", "second", "post_second", "pre_third", "third"]
        return self.get_constraint(modes[index])

    def get_constraint(
        self,
        jaxon: Jaxon,
        mode: Literal["satisfy", "motion_planning"],
        phase: Literal[
            "first", "post_first", "pre_second", "second", "post_second", "pre_third", "third"
        ],
    ) -> Tuple[AbstractEqConst, AbstractIneqConst]:

        # dirty hack to call e.g. self._first_axes
        method_name = "_" + phase + "_axes"
        axes: List[Coordinates] = getattr(self, method_name)()

        with mesh_simplify_factor(0.2):
            jaxon_config = JaxonConfig()

            # determine eq const
            if mode == "satisfy":
                efkin = jaxon_config.get_endeffector_kin()
            else:
                if phase == "pre_second":
                    efkin = jaxon_config.get_endeffector_kin(
                        rleg=False, lleg=True, rarm=True, larm=True
                    )
                    axes = axes[1:]
                elif phase == "pre_third":
                    efkin = jaxon_config.get_endeffector_kin(
                        rleg=True, lleg=False, rarm=True, larm=True
                    )
                    axes = [axes[0], axes[2], axes[3]]
                else:
                    assert False
            pose_const = PoseConstraint.from_skrobot_coords(axes, efkin, jaxon)
            eq_const = pose_const

            # determine ineq const
            if phase in ["first", "second", "third"]:
                colkin = jaxon_config.get_collision_kin(
                    rsole=False, lsole=False, rgripper=False, lgripper=False
                )
            elif phase in ["post_first", "pre_second"]:
                colkin = jaxon_config.get_collision_kin(
                    rsole=True, lsole=False, rgripper=False, lgripper=False
                )
            elif phase in ["post_second", "pre_third"]:
                colkin = jaxon_config.get_collision_kin(
                    rsole=False, lsole=True, rgripper=False, lgripper=False
                )
            else:
                assert False
            col_const = CollFreeConst(colkin, self.world.get_exact_sdf(), jaxon)
            selcol_const = jaxon_config.get_neural_selcol_const(jaxon)
            ineq_const = IneqCompositeConst([selcol_const, col_const])

            return eq_const, ineq_const

    def _first_axes(self) -> List[Axis]:
        step = self.world.steps[0]
        depth, _, height = step._extents

        rleg_pose: Coordinates = step.copy_worldcoords()
        rleg_pose.translate([-depth * 0.5 - 0.08, -0.12, height * 0.5])
        ax1 = Axis.from_coords(rleg_pose)

        lleg_pose: Coordinates = step.copy_worldcoords()
        lleg_pose.translate([-depth * 0.5 - 0.08, 0.12, height * 0.5])
        ax2 = Axis.from_coords(lleg_pose)

        rarm_pose: Coordinates = self.world.poles[0].copy_worldcoords()
        rarm_pose.translate([0.0, 0.0, 0.5], wrt="local")

        larm_pose: Coordinates = rarm_pose.copy_worldcoords()
        larm_pose.translate([0.0, +self.world.param.ladder_width, 0.0], wrt="local")

        rarm_pose.rotate(np.pi * 0.5, "x", wrt="local")
        rarm_pose.rotate(-np.pi * 0.3, "y", wrt="local")
        ax3 = Axis.from_coords(rarm_pose)

        larm_pose.rotate(-np.pi * 0.5, "x", wrt="local")
        larm_pose.rotate(-np.pi * 0.3, "y", wrt="local")
        ax4 = Axis.from_coords(larm_pose)
        return [ax1, ax2, ax3, ax4]

    def _post_first_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self._first_axes()
        pose_rleg = ax_rleg.copy_worldcoords()
        pose_rleg.translate([0, 0, 0.03])
        return [Axis.from_coords(pose_rleg), ax_lleg, ax_rarm, ax_larm]

    def _second_axes(self) -> List[Axis]:
        desired_trans = self._desired_translation_by_step()
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self._first_axes()
        pose_rleg = ax_rleg.copy_worldcoords()
        pose_rleg.translate(desired_trans)
        return [Axis.from_coords(pose_rleg), ax_lleg, ax_rarm, ax_larm]

    def _pre_second_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self._second_axes()
        pose_rleg = ax_rleg.copy_worldcoords()
        pose_rleg.translate([0, 0, 0.03])
        return [Axis.from_coords(pose_rleg), ax_lleg, ax_rarm, ax_larm]

    def _post_second_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self._second_axes()
        pose_lleg = ax_lleg.copy_worldcoords()
        pose_lleg.translate([0, 0, 0.03])
        return [ax_rleg, Axis.from_coords(pose_lleg), ax_rarm, ax_larm]

    def _third_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self._second_axes()
        pose_lleg = ax_lleg.copy_worldcoords()
        desired_trans = self._desired_translation_by_step()
        pose_lleg.translate(desired_trans)
        return [ax_rleg, Axis.from_coords(pose_lleg), ax_rarm, ax_larm]

    def _pre_third_axes(self) -> List[Axis]:
        ax_rleg, ax_lleg, ax_rarm, ax_larm = self._third_axes()
        pose_lleg = ax_lleg.copy_worldcoords()
        pose_lleg.translate([0, 0, 0.03])
        return [ax_rleg, Axis.from_coords(pose_lleg), ax_rarm, ax_larm]

    def _desired_translation_by_step(self) -> np.ndarray:
        step0 = self.world.steps[0]
        step1 = self.world.steps[1]
        diff = step1.worldpos() - step0.worldpos()
        return diff
