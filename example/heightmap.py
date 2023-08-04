import numpy as np
from skrobot.model.primitives import PointCloudLink
from skrobot.viewers import TrimeshSceneViewer

from rpbench.articulated.vision import LocatedHeightmap
from rpbench.articulated.world.utils import BoxSkeleton, CylinderSkelton

if __name__ == "__main__":
    box = BoxSkeleton([1, 1, 1])
    obs2 = BoxSkeleton([0.2, 0.2, 0.2], pos=[0.0, 0, -0.4])
    obs3 = CylinderSkelton(radius=0.2, height=0.4, pos=[-0.5, -0.5, -0.3])
    hmap = LocatedHeightmap.by_raymarching(box, [obs2, obs3])
    points = box.sample_points(100000)
    points_collide = np.array([p for p in points if hmap.is_colliding(p)])
    link = PointCloudLink(points_collide)

    v = TrimeshSceneViewer()
    v.add(box.to_visualizable((255, 255, 255, 120)))
    v.add(link)
    v.show()
    import time

    time.sleep(1000)
