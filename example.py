import time

from skrobot.viewers import TrimeshSceneViewer

from rpbench.world import TabletopBoxRightArmReachingTask

task = TabletopBoxRightArmReachingTask.sample(1, True)
viewer = TrimeshSceneViewer()
task.visualize(viewer)
viewer.show()
time.sleep(10)
