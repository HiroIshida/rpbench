import pickle
import time

import numpy as np
from skmp.constraint import CollFreeConst, IneqCompositeConst, PoseConstraint
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import SatisfactionConfig
from skmp.solver.interface import Problem
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver
from skmp.solver.nlp_solver import SQPBasedSolver, SQPBasedSolverConfig
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from rpbench.jaxon.below_table import CachedJaxonConstProvider, TableWorld

prov = CachedJaxonConstProvider()

world = TableWorld.sample(standard=True)

with mesh_simplify_factor(0.3):
    jaxon = Jaxon()
    jaxon.reset_manip_pose()
    jaxon.translate([0.0, 0.0, 0.98])

    jaxon_config = JaxonConfig()
    colkin = jaxon_config.get_collision_kin()
    colfree_const = CollFreeConst(colkin, world.get_exact_sdf(), jaxon, only_closest_feature=True)

    com_box = Box([0.25, 0.4, 5.0], with_sdf=True)
    com_box.visual_mesh.visual.face_colors = [255, 0, 100, 100]
    com_const = jaxon_config.get_com_stability_const(jaxon, com_box)

    ineq_const = IneqCompositeConst([colfree_const, com_const])

    leg_coords_list = [jaxon.rleg_end_coords, jaxon.lleg_end_coords]
    efkin_legs = jaxon_config.get_endeffector_kin(rarm=False, larm=False)
    global_eq_const = PoseConstraint.from_skrobot_coords(leg_coords_list, efkin_legs, jaxon)  # type: ignore

    goal_rarm_co = Coordinates([0.55, -0.6, 0.45], rot=[0, -0.5 * np.pi, 0])
    leg_coords_list = [jaxon.rleg_end_coords, jaxon.lleg_end_coords, goal_rarm_co]
    efkin_legs_rarm = jaxon_config.get_endeffector_kin(rarm=True, larm=False)
    goal_eq_const = PoseConstraint.from_skrobot_coords(leg_coords_list, efkin_legs_rarm, jaxon)

q_start = get_robot_state(jaxon, jaxon_config._get_control_joint_names(), BaseType.FLOATING)
problem = Problem(
    q_start,
    jaxon_config.get_box_const(),
    goal_eq_const,
    ineq_const,
    global_eq_const,
    motion_step_box_=jaxon_config.get_motion_step_box(),
)
problem = pickle.loads(pickle.dumps(problem))

rrt_conf = MyRRTConfig(500, satisfaction_conf=SatisfactionConfig(n_max_eval=50))
rrt = MyRRTConnectSolver.init(rrt_conf)
rrt.setup(problem)
print("start solving rrt")
result = rrt.parallel_solve(10)
assert result.traj is not None
print("time to plan: {}".format(result.time_elapsed))

print("smooth out the result")
solver = SQPBasedSolver.init(
    SQPBasedSolverConfig(
        n_wp=40,
        n_max_call=200,
        motion_step_satisfaction="explicit",
        verbose=True,
        ctol_eq=1e-3,
        ctol_ineq=1e-3,
        ineq_tighten_coef=0.0,
    )
)
solver.setup(problem)
smooth_result = solver.solve(result.traj)  # type: ignore
if smooth_result.traj is None:
    print("sqp: fail to smooth")
else:
    print("sqp: time to smooth: {}".format(smooth_result.time_elapsed))
    result = smooth_result  # type: ignore

vis = TrimeshSceneViewer()
world.visualize(vis)
axis = Axis.from_coords(goal_rarm_co)
vis.add(jaxon)
vis.add(axis)
vis.add(world.target_region)
vis.show()

time.sleep(4)
assert result.traj is not None
for q in result.traj:
    set_robot_state(jaxon, jaxon_config._get_control_joint_names(), q, base_type=BaseType.FLOATING)
    vis.redraw()
    time.sleep(0.5)
time.sleep(10)
