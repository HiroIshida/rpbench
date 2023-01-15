import tempfile
from pathlib import Path

from skmp.solver.ompl_solver import LightningSolver, OMPLSolverConfig

from rpbench.interface import DatadrivenTaskSolver, PlanningDataset
from rpbench.tabletop import TabletopBoxRightArmReachingTask


def test_dataset():
    task_type = TabletopBoxRightArmReachingTask
    solcon = OMPLSolverConfig()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).expanduser()
        dataset = PlanningDataset.create(task_type, 2)
        dataset.save(td_path)

        dataset_loaded = dataset.load(td_path, task_type)

        ddsolver = DatadrivenTaskSolver.init(LightningSolver, solcon, dataset_loaded)

    task = task_type.sample(1, True)
    ddsolver.setup(task)
    ddsolver.solve()


if __name__ == "__main__":
    test_dataset()
