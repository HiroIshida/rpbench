from typing import List

from skmp.constraint import CollFreeConst
from skmp.kinematics import CollSphereKinematicsMap, EndEffectorKinematicsMap
from skmp.robot.fetch import FetchConfig
from skmp.robot.utils import get_robot_state
from skmp.utils import sksdf_to_cppsdf
from skrobot.models.fetch import Fetch
from skrobot.sdf.signed_distance_function import UnionSDF

from rpbench.articulated.world.utils import PrimitiveSkelton
from rpbench.utils import lru_cache_keeping_random_state


class CachedFetchConstProvider:
    # collection of classmethods with lru_cache

    @classmethod
    @lru_cache_keeping_random_state
    def _get_fetch(cls) -> Fetch:
        return Fetch()

    @classmethod
    def get_fetch(cls) -> Fetch:
        fetch = cls._get_fetch()
        fetch.reset_pose()
        return fetch

    @classmethod
    @lru_cache_keeping_random_state
    def get_efkin(cls) -> EndEffectorKinematicsMap:
        config = FetchConfig()
        return config.get_endeffector_kin()

    @classmethod
    @lru_cache_keeping_random_state
    def get_colkin(cls) -> CollSphereKinematicsMap:
        config = FetchConfig()
        return config.get_collision_kin()

    @classmethod
    def get_collfree_const(
        cls,
        primitive_list: List[PrimitiveSkelton],
    ) -> CollFreeConst:
        # take primitive instead of sdf (unlike PR2 and jaxon) is to append them to
        # self body obstacles's sdf and create psdf.UnionSDF
        self_body_sksdf_list = [p.sdf for p in FetchConfig().get_self_body_obstacles()]
        sksdf_total_list = self_body_sksdf_list + [p.sdf for p in primitive_list]
        sksdf_total = UnionSDF(sksdf_total_list)
        cppsdf = sksdf_to_cppsdf(sksdf_total)
        collfree_const = CollFreeConst(
            cls.get_colkin(), cppsdf, cls._get_fetch(), only_closest_feature=True
        )
        return collfree_const


if __name__ == "__main__":
    from skmp.visualization.collision_visualizer import (
        CollisionSphereVisualizationManager,
    )
    from skrobot.viewers import PyrenderViewer

    from rpbench.articulated.world.jsk_table import JskMessyTableWorld

    prov = CachedFetchConstProvider
    fetch = prov.get_fetch()
    world = JskMessyTableWorld.sample(standard=False)
    world.table.translate([0.6, 0.0, 0.0])
    obs_list = world.get_all_obstacles()
    colfree_const = prov.get_collfree_const(obs_list)
    colfree_const.reflect_skrobot_model(fetch)

    q = get_robot_state(fetch, FetchConfig().get_control_joint_names())
    ret = colfree_const.evaluate_single(q, False)
    print(ret)

    v = PyrenderViewer()
    man = CollisionSphereVisualizationManager(prov.get_colkin(), v, None)
    man.update(fetch)
    world.visualize(v)
    v.show()
    v.add(fetch)
    import time

    time.sleep(1000)
