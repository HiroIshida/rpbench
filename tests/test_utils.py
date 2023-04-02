import numpy as np

from rpbench.utils import pose_vec_to_skcoords, skcoords_to_pose_vec


def test_pose_skcoords_coversion():
    vec = np.random.randn(6)
    co = pose_vec_to_skcoords(vec)
    vec_again = skcoords_to_pose_vec(co)
    co_again = pose_vec_to_skcoords(vec_again)
    np.testing.assert_almost_equal(co.rotation, co_again.rotation)
