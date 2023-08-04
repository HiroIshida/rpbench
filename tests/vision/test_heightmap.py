import numpy as np

from rpbench.articulated.vision import LocatedHeightmap
from rpbench.articulated.world.utils import BoxSkeleton, CylinderSkelton


def test_heightmap():
    box = BoxSkeleton([1, 1, 1])
    obs2 = BoxSkeleton([0.2, 0.2, 0.2], with_sdf=True, pos=[0.0, 0, -0.4])
    obs3 = CylinderSkelton(radius=0.2, height=0.4, pos=[-0.5, -0.5, -0.3], with_sdf=True)
    hmap = LocatedHeightmap.by_raymarching(box, [obs2, obs3], create_debug_points=True)

    np.testing.assert_almost_equal(hmap.get_max_height(np.array([0.0, 0.0, 0.0])), 0.2)
    np.testing.assert_almost_equal(hmap.get_max_height(np.array([-0.45, -0.45, 0.0])), 0.4)
    np.testing.assert_almost_equal(hmap.get_max_height(np.array([+0.45, +0.45, 0.0])), 0.0)
