import numpy as np

from rpbench.articulated.world.heightmap import LocatedHeightmap
from rpbench.articulated.world.utils import BoxSkeleton, CylinderSkelton


def test_heightmap():
    box = BoxSkeleton([1, 1, 1])
    obs1 = BoxSkeleton([1.0, 1.0, 0.1], with_sdf=True, pos=[0.0, 0, 0.05])
    obs2 = BoxSkeleton([0.2, 0.2, 0.2], with_sdf=True, pos=[0.0, 0, 0.1])
    obs3 = CylinderSkelton(radius=0.2, height=0.4, pos=[-0.5, -0.5, 0.2], with_sdf=True)
    hmap = LocatedHeightmap.by_raymarching(box, [obs1, obs2, obs3])

    # obs2
    assert hmap.is_colliding(np.array([0.0, 0.0, 0.0]))
    assert hmap.is_colliding(np.array([0.0, 0.0, 0.19]))
    assert not hmap.is_colliding(np.array([0.0, 0.0, 0.21]))

    # obs3
    assert hmap.is_colliding(np.array([-0.4, -0.4, 0.0]))
    assert hmap.is_colliding(np.array([-0.4, -0.4, 0.39]))
    assert not hmap.is_colliding(np.array([-0.4, -0.4, 0.41]))
