import pickle

from rpbench.articulated.world.ground import GroundClutteredWorld


def test_GroundClutteredWorld():
    w = GroundClutteredWorld.sample(True)
    assert w._heightmap is None
    w.heightmap()
    assert w._heightmap is not None

    w_again = pickle.loads(pickle.dumps(w))
    assert w_again._heightmap is None
