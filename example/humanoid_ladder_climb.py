import time

from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import set_robot_state
from skmp.satisfy import satisfy_by_optimization_with_budget
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from rpbench.jaxon.ladder import ConstraintSequence, LadderWorld

if __name__ == "__main__":
    world = LadderWorld.sample(standard=True)
    with mesh_simplify_factor(0.2):
        jaxon = Jaxon()
    jaxon_config = JaxonConfig()
    cons_seq = ConstraintSequence(world)

    q_pre = None
    for mode in [
        "first",
        "post_first",
        "pre_second",
        "second",
        "pre_third",
        "third",
    ]:
        eq_const, ineq_const = cons_seq.get_constraint(jaxon, "satisfy", mode)  # type: ignore
        bounds = jaxon_config.get_close_box_const(q_pre)
        print("start solving")
        result = satisfy_by_optimization_with_budget(
            eq_const, bounds, ineq_const, q_pre, n_trial_budget=300
        )
        print(result)
        q_pre = result.q
        assert result.success

    set_robot_state(
        jaxon, jaxon_config._get_control_joint_names(), result.q, base_type=BaseType.FLOATING
    )

    vis = TrimeshSceneViewer()
    vis.add(jaxon)
    world.visualize(vis)
    vis.show()
    time.sleep(100)
