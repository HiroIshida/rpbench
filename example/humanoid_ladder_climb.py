import time

from skmp.constraint import CollFreeConst, IneqCompositeConst, PoseConstraint
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import set_robot_state
from skmp.satisfy import satisfy_by_optimization_with_budget
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from rpbench.jaxon.ladder import LadderWorld

if __name__ == "__main__":
    world = LadderWorld.sample(standard=True)
    axes = world.third_axes()

    with mesh_simplify_factor(0.2):
        jaxon = Jaxon()
        jaxon_config = JaxonConfig()
        efkin_legonly = jaxon_config.get_endeffector_kin(rleg=True, lleg=True, rarm=True, larm=True)
        colkin = jaxon_config.get_collision_kin(rsole=False, lsole=False)

        # eq const
        pose_const = PoseConstraint.from_skrobot_coords(axes, efkin_legonly, jaxon)
        eq_const = pose_const

        # ineq const
        col_const = CollFreeConst(colkin, world.get_exact_sdf(), jaxon)
        selcol_const = jaxon_config.get_neural_selcol_const(jaxon)
        ineq_const = IneqCompositeConst([col_const, selcol_const])

        # bounds
        bounds = jaxon_config.get_box_const()

    # solving initial IK
    print("start solving")
    result = satisfy_by_optimization_with_budget(
        eq_const, bounds, ineq_const, None, n_trial_budget=300
    )
    assert result.success

    set_robot_state(
        jaxon, jaxon_config._get_control_joint_names(), result.q, base_type=BaseType.FLOATING
    )

    vis = TrimeshSceneViewer()
    vis.add(jaxon)
    for ax in axes:
        vis.add(ax)
    world.visualize(vis)
    vis.show()
    time.sleep(100)
