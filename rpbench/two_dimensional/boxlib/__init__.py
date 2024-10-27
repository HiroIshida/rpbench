import ctypes
from os import path
from pathlib import Path
from typing import ClassVar

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.ctypeslib import ndpointer

this_source_path = Path(path.abspath(__file__))
lib_path = this_source_path.parent / "boxlib.so"
if not lib_path.exists():
    import subprocess

    cmd = f"g++ -shared -fPIC -O3 -o {lib_path} {this_source_path.parent}/boxlib.cpp"
    ret = subprocess.run(cmd, shell=True)
    assert ret.returncode == 0

lib = ctypes.CDLL(str(lib_path))

lib.make_boxes.argtypes = [
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]
lib.make_boxes.restype = ctypes.c_void_p

lib.create_parametric_maze_boxes.argtypes = [
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]
lib.create_parametric_maze_boxes.restype = ctypes.c_void_p

lib.create_parametric_maze_boxes_special.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]
lib.create_parametric_maze_boxes_special.restype = ctypes.c_void_p

lib.signed_distance_batch.argtypes = [
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_void_p,
]
lib.signed_distance_batch.restype = None
lib.delete_boxes.argtypes = [ctypes.c_void_p]
lib.delete_boxes.restype = None


class ParametricMazeBase:
    ptr: ctypes.c_void_p
    n: int
    param: np.ndarray
    y_length: float
    wall_thickness = 0.20
    holl_width = 0.12

    def __del__(self):
        lib.delete_boxes(self.ptr)

    def signed_distance(self, x, y):
        return lib.signed_distance(ctypes.c_double(x), ctypes.c_double(y), self.ptr)

    def signed_distance_batch(self, x, y):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(x)
        dist = np.empty(n, dtype=np.float64)
        lib.signed_distance_batch(x, y, dist, ctypes.c_int(n), self.ptr)
        return dist

    def visualize(self, fax):
        fig, ax = fax
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, self.y_length + 0.02)
        wall_ys = np.linspace(0, self.y_length, self.n + 2)[1:-1]
        holl_xs = self.param

        for i in range(self.n):
            wall_y = wall_ys[i]
            wall_y_min = wall_y - self.wall_thickness / 2
            holl_x_center = holl_xs[i]
            holl_x_min = holl_x_center - self.holl_width / 2
            holl_x_max = holl_x_center + self.holl_width / 2
            holl_x_min = max(holl_x_min, 0)
            holl_x_max = min(holl_x_max, 1)
            if holl_x_min > 0:
                left_wall = patches.Rectangle(
                    (0, wall_y_min), holl_x_min, self.wall_thickness, color="dimgray"
                )
                ax.add_patch(left_wall)
            if holl_x_max < 1:
                right_wall = patches.Rectangle(
                    (holl_x_max, wall_y_min), 1 - holl_x_max, self.wall_thickness, color="dimgray"
                )
                ax.add_patch(right_wall)
        boundary = patches.Rectangle(
            (0, 0), 1, self.y_length, fill=False, edgecolor="black", linewidth=2
        )

        ax.add_patch(boundary)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)


class ParametricMaze(ParametricMazeBase):
    def __init__(self, param: np.ndarray):
        y_length = (len(param) + 1) * 0.7
        self.ptr = lib.create_parametric_maze_boxes(
            param, len(param), self.wall_thickness, self.holl_width, y_length
        )
        self.n = len(param)
        self.param = param
        self.y_length = y_length

    @classmethod
    def sample(cls, n: int):
        x_min = cls.holl_width * 0.5
        x_max = 1.0 - cls.holl_width * 0.5
        param = np.random.uniform(x_min, x_max, n)
        return cls(param)


class ParametricMazeSpecial(ParametricMazeBase):
    def __init__(self, param: float):
        y_length = (4 + 1) * 0.7
        self.ptr = lib.create_parametric_maze_boxes_special(
            param, self.wall_thickness, self.holl_width, y_length
        )
        self.n = 1
        self.param = np.array([param])
        self.y_length = y_length

    @classmethod
    def sample(cls):
        x_min = cls.holl_width * 0.5
        x_max = 1.0 - cls.holl_width * 0.5
        param = np.random.uniform(x_min, x_max)
        return cls(param)


class ParametricCircles:
    ptr: ctypes.c_void_p
    param: np.ndarray
    obstacle_w: ClassVar[float] = 0.5
    obstacle_h: ClassVar[float] = 0.3

    def __del__(self):
        lib.delete_boxes(self.ptr)

    def __init__(self, params: np.ndarray):
        self.param = params
        self.y_length = (4 + 1) * 0.7
        self.ys = np.linspace(0, self.y_length, len(params) + 2)[1:-1]
        # create_voxes
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        for x, y in zip(params, self.ys):
            xmins.append(x - self.obstacle_w / 2)
            xmaxs.append(x + self.obstacle_w / 2)
            ymins.append(y - self.obstacle_h / 2)
            ymaxs.append(y + self.obstacle_h / 2)
        xmins = np.array(xmins)
        xmaxs = np.array(xmaxs)
        ymins = np.array(ymins)
        ymaxs = np.array(ymaxs)
        self.ptr = lib.make_boxes(xmins, xmaxs, ymins, ymaxs, len(params))

    @classmethod
    def sample(cls, n: int):
        x_min = 0.0
        x_max = 1.0
        params = np.random.uniform(x_min, x_max, n)
        return cls(params)

    def signed_distance(self, x, y):
        return lib.signed_distance(ctypes.c_double(x), ctypes.c_double(y), self.ptr)

    def signed_distance_batch(self, x, y):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(x)
        dist = np.empty(n, dtype=np.float64)
        lib.signed_distance_batch(x, y, dist, ctypes.c_int(n), self.ptr)
        return dist

    def visualize(self, fax):
        fig, ax = fax
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, self.y_length + 0.02)
        for i, y in enumerate(self.ys):
            x = self.param[i]
            x_min = x - self.obstacle_w / 2
            x + self.obstacle_w / 2
            y_min = y - self.obstacle_h / 2
            y + self.obstacle_h / 2
            ax.add_patch(
                patches.Rectangle(
                    (x_min, y_min),
                    self.obstacle_w,
                    self.obstacle_h,
                    fill=True,
                    color="dimgray",
                )
            )
        boundary = patches.Rectangle(
            (0, 0), 1, self.y_length, fill=False, edgecolor="black", linewidth=2
        )

        ax.add_patch(boundary)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)


if __name__ == "__main__":
    maze = ParametricCircles([0.4, 0.3, 0.8])
    fig, ax = plt.subplots()
    maze.visualize((fig, ax))
    plt.show()

    # maze = ParametricMaze(np.array([0.3, 0.5, 0.7]))
    # xlin = np.linspace(0.0, 1.0, 100)
    # ylin = np.linspace(0.0, maze.y_length, 100)
    # X, Y = np.meshgrid(xlin, ylin)
    # pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    # dist = maze.signed_distance_batch(pts[:, 0], pts[:, 1])
    # plt.imshow(dist.reshape(100, 100) < 0.0)
    # plt.show()
