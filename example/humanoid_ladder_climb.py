import pickle
import time

from skmp.constraint import ConfigPointConst
from skmp.robot.jaxon import Jaxon, JaxonConfig
from skmp.robot.utils import set_robot_state
from skmp.satisfy import satisfy_by_optimization_with_budget
from skmp.solver.interface import Problem
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolver, SQPBasedSolverConfig
from skmp.visualization.collision_visualizer import CollisionSphereVisualizationManager
from skrobot.utils.urdf import mesh_simplify_factor
from skrobot.viewers import TrimeshSceneViewer
from tinyfk import BaseType

from rpbench.jaxon.ladder import ConstraintSequence, LadderWorld

# np.random.seed(0)

if __name__ == "__main__":
    world = LadderWorld.sample(standard=True)
    with mesh_simplify_factor(0.2):
        jaxon = Jaxon()
    jaxon_config = JaxonConfig()
    cons_seq = ConstraintSequence(world)

    modes = ["first", "post_first", "pre_second", "second", "post_second", "pre_third", "third"]

    use_cache: bool = False

    if not use_cache:
        q_pre = None
        qs = {}
        for mode in modes:
            eq_const, ineq_const = cons_seq.get_constraint(jaxon, "satisfy", mode)  # type: ignore
            if mode in ["post_first", "second", "post_second", "third"]:
                bounds = jaxon_config.get_close_box_const(
                    q_pre, joint_margin=0.1, base_pos_margin=0.1, base_rot_margin=0.1
                )
            else:
                bounds = jaxon_config.get_box_const()
            print("start solving {}".format(mode))
            result = satisfy_by_optimization_with_budget(
                eq_const, bounds, ineq_const, q_pre, n_trial_budget=300
            )
            q_pre = result.q
            assert result.success
            qs[mode] = result.q

        with open("/tmp/tmp.pkl", "wb") as f:
            pickle.dump(qs, f)

    with open("/tmp/tmp.pkl", "rb") as f:
        qs = pickle.load(f)

    tup1 = (qs["post_first"], qs["pre_second"], "pre_second")
    tup2 = (qs["post_second"], qs["pre_third"], "pre_third")
    trajs = []
    for q_start, q_goal, mode in [tup1, tup2]:
        eq_const, ineq_const = cons_seq.get_constraint(jaxon, "motion_planning", mode)  # type: ignore
        bounds = jaxon_config.get_box_const()

        problem = Problem(
            q_start,
            bounds,
            ConfigPointConst(q_goal),
            ineq_const,
            eq_const,
            motion_step_box_=0.1,
        )
        print("start solving rrt to mode {}".format(mode))
        conf = MyRRTConfig(10000)
        solver = MyRRTConnectSolver.init(conf)
        solver.setup(problem)
        res = solver.solve()
        print(res.time_elapsed)
        assert res.traj is not None

        print("start smoothing")
        smoother = SQPBasedSolver.init(
            SQPBasedSolverConfig(
                40,
                motion_step_satisfaction="debug_ignore",
                ineq_tighten_coef=0.0,
                verbose=True,
                ctol_eq=1e-3,
            )
        )
        smoother.setup(problem)
        smooth_res = smoother.solve(res.traj)
        assert smooth_res.traj is not None
        print("time to smooth: {}".format(smooth_res.time_elapsed))

        trajs.append(smooth_res.traj)

    q_whole = []
    q_whole.append(qs["first"])
    q_whole.append(qs["post_first"])
    q_whole.extend(list(trajs[0].resample(12).numpy()))
    q_whole.append(qs["second"])
    q_whole.extend(list(trajs[1].resample(12).numpy()))
    q_whole.append(qs["third"])

    vis = TrimeshSceneViewer()
    vis.add(jaxon)
    colkin = jaxon_config.get_collision_kin()
    colvis = CollisionSphereVisualizationManager(colkin, vis)
    world.visualize(vis)
    vis.show()

    time.sleep(10)
    for q in q_whole:
        set_robot_state(
            jaxon, jaxon_config._get_control_joint_names(), q, base_type=BaseType.FLOATING
        )
        colvis.update(jaxon)
        vis.redraw()
        time.sleep(0.5)
    time.sleep(10)
