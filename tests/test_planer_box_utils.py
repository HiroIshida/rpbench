import numpy as np

from rpbench.planer_box_utils import Box2d, PlanerCoords


def test_box2d_vertices():
    co = PlanerCoords.standard()
    for _ in range(20):
        co.angle = np.random.rand() * 10
        box = Box2d(np.random.rand(2), co)
        for edge in box.edges:
            detval = np.linalg.det(np.vstack([edge[0], edge[1]]).T)
            assert detval > 0
