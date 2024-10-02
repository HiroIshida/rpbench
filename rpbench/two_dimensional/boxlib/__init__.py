import ctypes
from os import path
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.ctypeslib import ndpointer

this_source_path = Path(path.abspath(__file__))
lib_path = this_source_path.parent / "boxlib.so"
lib = ctypes.CDLL(str(lib_path))
lib.create_parametric_maze_boxes.argtypes = [
    ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
]
lib.create_parametric_maze_boxes.restype = ctypes.c_void_p

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


class ParametricMaze:
    ptr: ctypes.c_void_p
    n: int
    param: np.ndarray
    wall_thickness = 0.1
    holl_width = 0.1
    # holl_width = 0.03

    def __init__(self, param: np.ndarray):
        self.ptr = lib.create_parametric_maze_boxes(
            param, len(param), self.wall_thickness, self.holl_width
        )
        self.n = len(param)
        self.param = param

    @classmethod
    def sample(cls, n: int):
        x_min = cls.holl_width * 0.5
        x_max = 1.0 - cls.holl_width * 0.5
        param = np.random.uniform(x_min, x_max, n)
        return cls(param)

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
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        wall_ys = np.linspace(0, 1, self.n + 2)[1:-1]
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
                    (0, wall_y_min), holl_x_min, self.wall_thickness, color="black"
                )
                ax.add_patch(left_wall)
            if holl_x_max < 1:
                right_wall = patches.Rectangle(
                    (holl_x_max, wall_y_min), 1 - holl_x_max, self.wall_thickness, color="black"
                )
                ax.add_patch(right_wall)
        boundary = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2)

        ax.add_patch(boundary)
        ax.set_aspect("equal")
        plt.title("Parametric Maze Visualization")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)


if __name__ == "__main__":
    maze = ParametricMaze(np.array([0.2]))
    xlin = np.linspace(0.0, 1.0, 100)
    ylin = np.linspace(0.0, 1.0, 100)
    X, Y = np.meshgrid(xlin, ylin)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dist = maze.signed_distance_batch(pts[:, 0], pts[:, 1])
    import matplotlib.pyplot as plt

    plt.imshow(dist.reshape(100, 100) < 0.0)
    plt.show()
