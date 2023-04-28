from typing import Dict, Type

from skmp.satisfy import SatisfactionConfig
from skmp.solver.myrrt_solver import MyRRTConfig, MyRRTConnectSolver
from skmp.solver.nlp_solver.memmo import NnMemmoSolver
from skmp.solver.nlp_solver.sqp_based_solver import SQPBasedSolverConfig
from skmp.solver.ompl_solver import LightningSolver, OMPLSolver, OMPLSolverConfig

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

        # lightning
        compat_solvers["lightning"] = DatadrivenTaskSolver.init(
            LightningSolver, ompl_config, trajlib_dataset
        )

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
        compat_solvers["rrtconnect"] = SkmpTaskSolver.init(
            MyRRTConnectSolver.init(myrrt_config), task_type
        )

        sqp_config = SQPBasedSolverConfig(30, motion_step_satisfaction="explicit")
        compat_solvers["memmo_nn"] = DatadrivenTaskSolver.init(
            NnMemmoSolver, sqp_config, trajlib_dataset
        )

        return compat_solvers
