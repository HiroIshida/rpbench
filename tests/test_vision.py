from typing import Callable, List, Tuple

import numpy as np
import tqdm
from skrobot.model.primitives import Sphere

try:
    from voxbloxpy.core import EsdfMap, IntegratorType

    VOXBLOX_INSTALLED = True
except ImportError:
    VOXBLOX_INSTALLED = False

from rpbench.articulated.vision import Camera, CameraConfig, RayMarchingConfig


def genrate_camera_pcloud_pairs(
    sdf: Callable[[np.ndarray], np.ndarray],
    camera_config: CameraConfig,
    rm_config: RayMarchingConfig,
    nx: int = 20,
    nz: int = 3,
) -> List[Tuple[Camera, np.ndarray]]:
    thetas = np.linspace(0, 2 * np.pi, nx)
    phis = np.linspace(-0.7 * np.pi, 0.7 * np.pi, nz)

    pairs = []
    for theta in thetas:
        for phi in phis:
            pos = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
            camera = Camera(pos, config=camera_config)
            camera.look_at(np.zeros(3), horizontal=True)
            pts_global = camera.generate_point_cloud(sdf, rm_config, hit_only=True)
            pairs.append((camera, pts_global))
    return pairs


def test_ray_marching():
    conf = RayMarchingConfig(max_dist=3.0)
    sphere = Sphere(1.0, with_sdf=True)

    pts_starts = np.random.normal(size=(100, 3))
    pts_starts = 2 * pts_starts / np.linalg.norm(pts_starts, axis=1)[:, np.newaxis]
    direction_arr_unit = -pts_starts / np.linalg.norm(pts_starts, axis=1)[:, np.newaxis]
    dists = Camera.ray_marching(pts_starts, direction_arr_unit, sphere.sdf, conf)
    np.testing.assert_almost_equal(dists, 1.0)


def test_generate_point_cloud():
    sphere = Sphere(0.2, pos=(0.2, 0.2, 0.2), with_sdf=True)
    assert sphere.sdf is not None
    pairs = genrate_camera_pcloud_pairs(
        sphere.sdf, CameraConfig(resolx=64, resoly=48), RayMarchingConfig(max_dist=3.0)
    )
    pts_concat = np.concatenate([p[1] for p in pairs])

    values = sphere.sdf(pts_concat)
    assert np.max(np.abs(values)) < 1e-2


def test_synthetic_esdf():
    if not VOXBLOX_INSTALLED:
        return
    radius = 0.4
    resol = 0.02
    sphere = Sphere(radius, with_sdf=True)
    assert sphere.sdf is not None
    pairs = genrate_camera_pcloud_pairs(
        sphere.sdf, CameraConfig(resolx=128, resoly=96), RayMarchingConfig(max_dist=3.0), nz=5
    )
    esdf = EsdfMap.create(resol, integrator_type=IntegratorType.SIMPLE)
    for camera, cloud_global in tqdm.tqdm(pairs):
        cloud_camera = camera.inverse_transform_vector(cloud_global)
        esdf.update(camera.get_voxbloxpy_camera_pose(), cloud_camera)

    def create_sphere_points(radius):
        pts = []
        for theta in np.linspace(0, 2 * np.pi, 100):
            for phi in np.linspace(-0.4 * np.pi, 0.4 * np.pi, 10):
                pos = (
                    np.array(
                        [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]
                    )
                    * radius
                )
                pts.append(pos)
        pts_sphere = np.array(pts)
        return pts_sphere

    def check(dist_from_surface: float, admissible_rate: float):
        pts_sphere = create_sphere_points(radius + dist_from_surface)
        sd_values = esdf.get_sd_batch(pts_sphere)
        success_rate = np.sum(np.abs(sd_values - dist_from_surface) < resol * 2) / len(pts_sphere)
        assert success_rate > admissible_rate

    check(-0.04, 0.99)
    check(0.0, 0.99)
    check(0.05, 0.95)
    check(0.1, 0.9)
    check(0.2, 0.7)
