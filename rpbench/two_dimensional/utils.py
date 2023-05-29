from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class Grid2d:
    lb: np.ndarray
    ub: np.ndarray
    sizes: Tuple[int, int]


@dataclass
class Grid2dSDF:
    values: np.ndarray
    grid: Grid2d
    itp: RegularGridInterpolator

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.itp(x)
