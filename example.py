import time

from skrobot.viewers import TrimeshSceneViewer

from rpbench.world import TabletopBoxSingleArmReaching

prob = TabletopBoxSingleArmReaching.sample(1)
viewer = TrimeshSceneViewer()
prob.visualize(viewer)
viewer.show()
time.sleep(10)
