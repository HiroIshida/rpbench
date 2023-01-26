import contextlib

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle


def skcoords_to_pose_vec(co: Coordinates) -> np.ndarray:
    pos = co.worldpos()
    rot = co.worldrot()
    ypr = rpy_angle(rot)[0]
    rpy = np.flip(ypr)
    return np.hstack((pos, rpy))


@contextlib.contextmanager
def temp_seed(seed, use_tempseed):
    if use_tempseed:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)
    else:
        yield
