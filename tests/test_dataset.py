import tempfile
from pathlib import Path

from skmp.solver.ompl_solver import OMPLDataDrivenSolver, OMPLSolverConfig

from rpbench.interface import DatadrivenTaskSolver, PlanningDataset
from rpbench.pr2.tabletop import TabletopBoxRightArmReachingTask


def test_dataset():
    task_type = TabletopBoxRightArmReachingTask
    solcon = OMPLSolverConfig()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td).expanduser()
        n_data = 5
        dataset = PlanningDataset.create(task_type, n_data, m_process=2)
        assert len(dataset.pairs) == n_data
        dataset.save(td_path)

        dataset_loaded = dataset.load(task_type, td_path)

        ddsolver = DatadrivenTaskSolver.init(OMPLDataDrivenSolver, solcon, dataset_loaded)

    task = task_type.sample(1, True)
    ddsolver.setup(task)
    ddsolver.solve()


if __name__ == "__main__":
    test_dataset()
