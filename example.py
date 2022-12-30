import time

from skrobot.viewers import TrimeshSceneViewer

from rpbench.world import TabletopBoxSingleArmReaching

task = TabletopBoxSingleArmReaching.sample(1)
viewer = TrimeshSceneViewer()
task.visualize(viewer)
viewer.show()
time.sleep(10)
