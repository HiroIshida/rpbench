from typing import Dict, Type

from skmp.satisfy import SatisfactionConfig
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver
from skmp.solver.nlp_solver.memmo import NnMemmoSolver
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolverConfig
from skmp.solver.ompl_solver import OMPLSolver, OMPLSolverConfig

from rpbench.interface import (
    AbstractTaskSolver,
    DatadrivenTaskSolver,
    PlanningDataset,
    SkmpTaskSolver,
)
from rpbench.jaxon.below_table import HumanoidTableReachingTask
from rpbench.pr2.kivapod import KivapodEmptyReachingTask


class CompatibleSolvers:
    @classmethod
    def get_compatible_solvers(
        cls, task_type: Type[SkmpTaskSolver]
    ) -> Dict[str, AbstractTaskSolver]:
        method_name = "_" + task_type.__name__
        return getattr(cls, method_name)()

    @staticmethod
    def _KivapodEmptyReachingTask() -> Dict[str, AbstractTaskSolver]:
        task_type = KivapodEmptyReachingTask

        compat_solvers: Dict[str, AbstractTaskSolver] = {}

        trajlib_dataset = PlanningDataset.load(task_type)

        ompl_config = OMPLSolverConfig(n_max_call=3000, n_max_satisfaction_trial=30)
        sqp_config = SQPBasedSolverConfig(30, motion_step_satisfaction="explicit")

        # rrtconnect
        compat_solvers["rrtconnect"] = SkmpTaskSolver.init(OMPLSolver.init(ompl_config), task_type)

        # memmo
        compat_solvers["memmo_nn"] = DatadrivenTaskSolver.init(
            NnMemmoSolver, sqp_config, trajlib_dataset
        )

        return compat_solvers

    @staticmethod
    def _HumanoidTableReachingTask() -> Dict[str, AbstractTaskSolver]:
        task_type = HumanoidTableReachingTask

        compat_solvers: Dict[str, AbstractTaskSolver] = {}

        trajlib_dataset = PlanningDataset.load(task_type)

        myrrt_config = MyRRTConfig(3000, satisfaction_conf=SatisfactionConfig(n_max_eval=30))
        myrrt = MyRRTConnectSolver.init(myrrt_config)
        myrrt_parallel4 = myrrt.as_parallel_solver(n_process=4)
        myrrt_parallel8 = myrrt.as_parallel_solver(n_process=8)

        compat_solvers["rrtconnect"] = SkmpTaskSolver.init(myrrt, task_type)
        compat_solvers["rrtconnect4"] = SkmpTaskSolver.init(myrrt_parallel4, task_type)
        compat_solvers["rrtconnect8"] = SkmpTaskSolver.init(myrrt_parallel8, task_type)

        sqp_config = SQPBasedSolverConfig(30, motion_step_satisfaction="explicit")
        print("memmo nn 100")
        compat_solvers["memmo_nn100"] = DatadrivenTaskSolver.init(
            NnMemmoSolver,
            sqp_config,
            trajlib_dataset,
            n_data_use=100,
        )
        print("memmo nn 1000")
        compat_solvers["memmo_nn1000"] = DatadrivenTaskSolver.init(
            NnMemmoSolver,
            sqp_config,
            trajlib_dataset,
            n_data_use=1000,
        )
        print("memmo nn 10000")
        compat_solvers["memmo_nn10000"] = DatadrivenTaskSolver.init(
            NnMemmoSolver,
            sqp_config,
            trajlib_dataset,
            n_data_use=10000,
        )
        return compat_solvers
