import numpy as np
from skrobot.sdf import UnionSDF

from rpbench.articulated.world.utils import BoxSkeleton, VoxelGrid, VoxelGridSkelton

np.random.seed(0)


def test_voxel_grid():
    box = BoxSkeleton([1, 0.8, 0.7])
    lb = -box.extents * 0.5
    ub = box.extents * 0.5
    mini_boxes = []
    for _ in range(10):
        mini_box = BoxSkeleton([0.1, 0.1, 0.2])
        pos = np.random.uniform(lb, ub)
        mini_box.translate(pos)
        box.assoc(mini_box)
        mini_boxes.append(mini_box)
    sdf = UnionSDF([box.sdf for box in mini_boxes])
    skelton = VoxelGridSkelton.from_box(box, (56, 56, 56))
    voxel_grid = VoxelGrid.from_sdf(sdf, skelton)

    # round trip correctness of serialization
    bytes1 = voxel_grid.serialize()
    bytes2 = VoxelGrid.deserialize(voxel_grid.serialize()).serialize()
    assert bytes1 == bytes2

    # round trip correctness of voxelization
    arr = voxel_grid.to_3darray()
    again = VoxelGrid.from_3darray(arr, skelton)
    bytes3 = again.serialize()
    assert bytes1 == bytes3


if __name__ == "__main__":
    test_voxel_grid()
