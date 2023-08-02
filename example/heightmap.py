import numpy as np
from skrobot.model.primitives import PointCloudLink
from skrobot.viewers import TrimeshSceneViewer

from rpbench.articulated.world.heightmap import LocatedHeightmap
from rpbench.articulated.world.utils import BoxSkeleton, CylinderSkelton

if __name__ == "__main__":
    box = BoxSkeleton([1, 1, 1])
    obs1 = BoxSkeleton([1.0, 1.0, 0.1], with_sdf=True, pos=[0.0, 0, 0.05])
    obs2 = BoxSkeleton([0.0, 0.0, 0.2], with_sdf=True, pos=[0.0, 0, 0.1])
    obs3 = CylinderSkelton(radius=0.2, height=0.4, pos=[-0.5, -0.5, 0.1], with_sdf=True)
    hmap = LocatedHeightmap.by_raymarching(box, [obs1, obs2, obs3])
    points = box.sample_points(100000)
    points_collide = np.array([p for p in points if hmap.is_colliding(p)])
    link = PointCloudLink(points_collide)

    v = TrimeshSceneViewer()
    v.add(box.to_visualizable((255, 255, 255, 120)))
    v.add(link)
    v.show()
    import time

    time.sleep(1000)
