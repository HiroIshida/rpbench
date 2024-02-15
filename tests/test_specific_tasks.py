import numpy as np

from rpbench.articulated.jaxon.below_table import (
    HumanoidTableClutteredReachingIntrinsicTask2,
    HumanoidTableClutteredReachingTask,
    HumanoidTableClutteredReachingTask2,
    HumanoidTableReachingTask,
    HumanoidTableReachingTask2,
)


def test_vector_descriptions():
    test_table = {
        HumanoidTableReachingTask: ((2 + 5 + 6), False),
        HumanoidTableReachingTask2: ((2 + 5 + 3), False),
        HumanoidTableClutteredReachingTask: ((2 + 6), True),
        HumanoidTableClutteredReachingTask2: ((2 + 3), True),
        HumanoidTableClutteredReachingIntrinsicTask2: ((2 + 5 * 8 + 3), False),
    }

    for task_type, (desc_len, has_mesh) in test_table.items():
        descs = []
        for _ in range(10):
            task = task_type.sample(1)
            table = task.export_table()
            desc = table.get_vector_descs()[0]
            assert len(desc) == desc_len
            descs.append(desc)

            mesh = table.get_mesh()
            if has_mesh:
                assert mesh is not None
            else:
                assert mesh is None

        descs = np.array(descs)

    # check that all descs are different
    assert len(np.unique(descs, axis=0)) == len(descs)
