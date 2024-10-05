import copy
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import disbmp
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csc_matrix


class BlockMatrix:
    inner_block_size: Tuple[int, int]
    outer_block_size: Tuple[int, int]
    mat: np.ndarray

    def __init__(self, inner_block_size: Tuple[int, int], outer_block_size: Tuple[int, int]):
        # crate block matrix
        mat = np.zeros(
            (outer_block_size[0] * inner_block_size[0], outer_block_size[1] * inner_block_size[1])
        )
        self.inner_block_size = inner_block_size
        self.outer_block_size = outer_block_size
        self.mat = mat

    def randomize(self):
        self.mat = np.random.rand(*self.mat.shape)

    def get_block(self, i: int, j: int) -> np.ndarray:
        if i >= self.outer_block_size[0] or j >= self.outer_block_size[1]:
            raise IndexError("Block index out of range")
        return self.mat[
            i * self.inner_block_size[0] : (i + 1) * self.inner_block_size[0],
            j * self.inner_block_size[1] : (j + 1) * self.inner_block_size[1],
        ]

    # implement index access to block matrix
    def __getitem__(self, key: Tuple[int, int]) -> np.ndarray:
        return self.get_block(*key)

    def __setitem__(self, key: Tuple[int, int], value: np.ndarray) -> None:
        # assert value.shape == self.inner_block_size
        mat = self.get_block(*key)
        mat[:, :] = value


@dataclass
class TrajectoryConfig:
    n_dim: int
    n_steps: int
    dt: float

    @property
    def total_dim(self) -> int:
        return self.n_dim * 2 * self.n_steps + self.n_dim * (self.n_steps - 1)


class HasTrajectoryConfigMixin:

    traj_conf: TrajectoryConfig

    @property
    def n_dim(self) -> int:
        return self.traj_conf.n_dim

    @property
    def n_steps(self) -> int:
        return self.traj_conf.n_steps

    @property
    def dt(self) -> float:
        return self.traj_conf.dt


@dataclass
class Trajectory(HasTrajectoryConfigMixin):
    traj_conf: TrajectoryConfig
    X: np.ndarray  # positions
    V: np.ndarray  # velocities
    U: np.ndarray  # controls

    @classmethod
    def from_disbmp_traj(cls, traj: disbmp.Trajectory, n_step, dt) -> "Trajectory":
        # 0 to traj.get_duration()
        times = np.linspace(0, traj.get_duration(), n_step)
        states = np.array([traj.interpolate(t) for t in times])
        X = states[:, :2]
        V = states[:, 2:]
        n_step = X.shape[0]
        traj_conf = TrajectoryConfig(2, n_step, dt)
        return cls.from_X_and_V(X, V, traj_conf)

    @classmethod
    def from_X_and_V(
        cls, X: np.ndarray, V: np.ndarray, traj_conf: TrajectoryConfig
    ) -> "Trajectory":
        n_dim = traj_conf.n_dim
        n_steps = traj_conf.n_steps
        dt = traj_conf.dt

        U = np.zeros((n_steps - 1, n_dim))
        for i in range(n_steps - 1):
            U[i] = (V[i + 1] - V[i]) / dt
        return cls(traj_conf, X, V, U)

    @classmethod
    def from_two_points(
        cls, p_start: np.ndarray, p_end: np.ndarray, traj_conf: TrajectoryConfig
    ) -> "Trajectory":
        n_dim = traj_conf.n_dim
        n_steps = traj_conf.n_steps
        dt = traj_conf.dt

        X = np.zeros((n_steps, n_dim))
        V = np.zeros((n_steps, n_dim))
        U = np.zeros((n_steps - 1, n_dim))

        diff = (p_end - p_start) / (n_steps - 1)
        for i in range(n_steps):
            X[i] = p_start + i * diff
            X[i] += np.random.randn(n_dim) * 0.05 * diff
            if i < n_steps - 1:
                V[i] = diff / n_steps / dt
        return cls(traj_conf, X, V, U)

    def to_array(self) -> np.ndarray:
        S = np.hstack([self.X, self.V])
        return np.hstack([S.ravel(), self.U.ravel()])

    @classmethod
    def from_array(cls, traj_array: np.ndarray, traj_conf: TrajectoryConfig):
        n_dim = traj_conf.n_dim
        n_steps = traj_conf.n_steps

        S = traj_array[: n_steps * n_dim * 2].reshape(n_steps, n_dim * 2)
        X = S[:, :n_dim]  # positions
        V = S[:, n_dim:]  # velocities
        U = traj_array[n_steps * n_dim * 2 :].reshape(n_steps - 1, n_dim)
        return cls(traj_conf, X, V, U)

    def get_length(self) -> float:
        return np.sum(np.linalg.norm(self.X[1:] - self.X[:-1], axis=1))


@dataclass
class TrajectoryBound:
    x_min: np.ndarray
    x_max: np.ndarray
    v_min: np.ndarray
    v_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray

    def lower_bound(self, n_step: int) -> np.ndarray:
        s_min = np.hstack([self.x_min, self.v_min])
        s_min_tile = np.tile(s_min, (n_step, 1))
        u_min_tile = np.tile(self.u_min, (n_step - 1, 1))
        return np.hstack([s_min_tile.ravel(), u_min_tile.ravel()])

    def upper_bound(self, n_step: int) -> np.ndarray:
        s_max = np.hstack([self.x_max, self.v_max])
        s_max_tile = np.tile(s_max, (n_step, 1))
        u_max_tile = np.tile(self.u_max, (n_step - 1, 1))
        return np.hstack([s_max_tile.ravel(), u_max_tile.ravel()])


class TrajectoryCostFunction(HasTrajectoryConfigMixin):
    traj_conf: TrajectoryConfig
    cost_matrix: csc_matrix

    def __init__(self, traj_conf: TrajectoryConfig):
        n_dim = traj_conf.n_dim
        n_steps = traj_conf.n_steps
        # note n_control = n_dim

        # traj is a concat of s_1:T and u_1:T-1
        # therefore the dimension of traj descriptor is (T * 2 * n_dim) + ((T - 1) * n_dim)
        # the former is the state and the latter is the control
        traj_dim = n_steps * 2 * n_dim + (n_steps - 1) * n_dim
        cost_matrix = np.zeros((traj_dim, traj_dim))

        # cost is only enforced fro the control input so
        cost_matrix[n_steps * 2 * n_dim :, n_steps * 2 * n_dim :] = np.eye((n_steps - 1) * n_dim)
        self.cost_matrix = sparse.csc_matrix(cost_matrix)
        self.traj_conf = traj_conf

    def __call__(self, traj: np.ndarray) -> Tuple[float, np.ndarray]:
        U = traj[self.n_steps * self.n_dim * 2 :].reshape(self.n_steps - 1, self.n_dim)
        cost = float(0.5 * U.ravel().T @ U.ravel())
        # compute gradient
        gradient = np.zeros(traj.size)
        gradient[self.n_steps * self.n_dim * 2 :] = U.ravel()
        return cost, gradient


class TrajectoryEndPointConstraint(HasTrajectoryConfigMixin):
    traj_conf: TrajectoryConfig
    start: np.ndarray
    goal: np.ndarray
    is_sparse: bool
    """
    An equality constraint that enforces the start and end point of the trajectory
    match the given start and end point
    """

    def __init__(
        self,
        traj_conf: TrajectoryConfig,
        start: np.ndarray,
        goal: np.ndarray,
        is_sparse: bool = True,
    ):
        self.traj_conf = traj_conf
        self.start = start
        self.goal = goal
        self.is_sparse = is_sparse

    def __call__(self, traj: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, csc_matrix]]:
        idx_goal_start = (self.n_steps - 1) * self.n_dim * 2

        start_actual = traj[: self.n_dim]
        goal_actual = traj[idx_goal_start : idx_goal_start + self.n_dim]
        start_vel_actual = traj[self.n_dim : self.n_dim * 2]
        goal_vel_actual = traj[idx_goal_start + self.n_dim : idx_goal_start + self.n_dim * 2]
        residual = np.hstack(
            [start_actual - self.start, goal_actual - self.goal, start_vel_actual, goal_vel_actual]
        )

        # jacobian
        jac = np.zeros((len(residual), len(traj)))
        jac[: self.n_dim, : self.n_dim] = np.eye(self.n_dim)  # start position
        jac[self.n_dim : self.n_dim * 2, idx_goal_start : idx_goal_start + self.n_dim] = np.eye(
            self.n_dim
        )  # goal position
        jac[self.n_dim * 2 : self.n_dim * 3, self.n_dim : self.n_dim * 2] = np.eye(
            self.n_dim
        )  # start velocity
        jac[
            self.n_dim * 3 : self.n_dim * 4,
            idx_goal_start + self.n_dim : idx_goal_start + self.n_dim * 2,
        ] = np.eye(
            self.n_dim
        )  # goal velocity

        if self.is_sparse:
            return residual, sparse.csc_matrix(jac)
        else:
            return residual, jac


class TrajectoryDifferentialConstraint(HasTrajectoryConfigMixin):
    traj_conf: TrajectoryConfig
    jac: Union[np.ndarray, csc_matrix]
    is_sparse: bool

    def __init__(self, traj_conf: TrajectoryConfig, is_sparse: bool = True):
        n_dim = traj_conf.n_dim
        n_steps = traj_conf.n_steps
        # note n_control = n_dim
        dt = traj_conf.dt

        # we have T - 1 constraints for each s_{t+1} = f(s_t, u_t)
        # define jacobian of these constraints wrt s_t
        jac_s = BlockMatrix((n_dim * 2, n_dim * 2), (n_steps - 1, n_steps))
        for i in range(n_steps - 1):
            jac_s[i, i] = -np.block(
                [[np.eye(n_dim), dt * np.eye(n_dim)], [np.zeros((n_dim, n_dim)), np.eye(n_dim)]]
            )
            jac_s[i, i + 1] = np.eye(n_dim * 2)

        # define jacobian of these constraints wrt u_t
        jac_u = BlockMatrix((n_dim * 2, n_dim), (n_steps - 1, n_steps - 1))
        for i in range(n_steps - 1):
            jac_u[i, i] = -np.block([[0.5 * dt**2 * np.eye(n_dim)], [dt * np.eye(n_dim)]])

        # sparsify the jacobians
        jac = np.hstack([jac_s.mat, jac_u.mat])
        if is_sparse:
            self.jac = sparse.csc_matrix(jac)
        else:
            self.jac = jac
        self.traj_conf = traj_conf
        self.is_sparse = is_sparse

    def __call__(self, traj: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, csc_matrix]]:
        # traj is a concat of s_1:T and u_1:T-1
        # therefore the dimension is (T * 2 * n_dim) + ((T - 1) * n_dim)
        # the former is the state and the latter is the control
        S = traj[: self.n_steps * self.n_dim * 2].reshape(self.n_steps, self.n_dim * 2)
        X = S[:, : self.n_dim]  # positions
        V = S[:, self.n_dim :]  # velocities
        U = traj[self.n_steps * self.n_dim * 2 :].reshape(self.n_steps - 1, self.n_dim)
        X_tm1 = X[:-1]
        V_tm1 = V[:-1]

        X_t_est = X_tm1 + V_tm1 * self.dt + 0.5 * U * self.dt**2
        V_t_est = V_tm1 + U * self.dt

        X_residual = X[1:] - X_t_est
        V_residual = V[1:] - V_t_est

        # compute residual such that [X_res1, V_res1, X_res2, V_res2, ...]
        residual = np.block([X_residual, V_residual]).ravel()
        return residual, self.jac


class TrajectoryObstacleAvoidanceConstraint(HasTrajectoryConfigMixin):
    traj_conf: TrajectoryConfig
    sdf: Callable[[np.ndarray], np.ndarray]  # signed distance function
    is_sparse: bool
    only_closest: bool

    def __init__(
        self,
        traj_conf: TrajectoryConfig,
        sdf: Callable[[np.ndarray], np.ndarray],
        is_sparse: bool = True,
        only_closest: bool = False,
    ):
        self.traj_conf = traj_conf
        self.sdf = sdf
        self.is_sparse = is_sparse
        self.only_closest = only_closest

    def __call__(self, traj: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, csc_matrix]]:
        if self.only_closest:
            return self._call_impl_only_closest(traj)
        else:
            return self._call_impl_default(traj)

    def _call_impl_only_closest(
        self, traj: np.ndarray
    ) -> Tuple[np.ndarray, Union[np.ndarray, csc_matrix]]:
        S = traj[: self.n_steps * self.n_dim * 2].reshape(self.n_steps, self.n_dim * 2)
        X = S[:, : self.n_dim]  # positions
        vals = self.sdf(X)
        min_idx = np.argmin(vals)
        x_min = X[min_idx]
        min_val = vals[min_idx]

        n_opt_dim = len(traj)
        grad = np.zeros((1, n_opt_dim))
        eps = 1e-6
        for i in range(self.n_dim):
            x_plus = copy.deepcopy(x_min)
            x_plus[i] += eps
            val_plus = self.sdf(np.array([x_plus]))[0]
            grad[0, self.n_dim * min_idx + i] = (val_plus - min_val) / eps
        grad_as_csc = sparse.csc_matrix(grad)
        return np.array([min_val]), grad_as_csc

    def _call_impl_default(
        self, traj: np.ndarray
    ) -> Tuple[np.ndarray, Union[np.ndarray, csc_matrix]]:
        S = traj[: self.n_steps * self.n_dim * 2].reshape(self.n_steps, self.n_dim * 2)
        X = S[:, : self.n_dim]  # positions

        val0 = self.sdf(X)
        eps = 1e-6
        # create X where only 0 the element is eps-added

        grads = np.zeros((self.n_steps, self.n_dim))
        for i in range(self.n_dim):
            X_plus = copy.deepcopy(X)
            X_plus[:, i] += eps
            val1 = self.sdf(X_plus)
            grad = (val1 - val0) / eps
            grads[:, i] = grad

        jac_s_block = BlockMatrix((1, self.n_dim * 2), (self.n_steps, self.n_steps))
        for i in range(self.n_steps):
            jac_s_block[i, i] = np.hstack([grads[i], np.zeros(self.n_dim)])
        jac_s = jac_s_block.mat

        jac_total = np.zeros((jac_s.shape[0], len(traj)))
        jac_total[:, : jac_s.shape[1]] = jac_s
        if self.is_sparse:
            return val0, sparse.csc_matrix(jac_total)
        else:
            return val0, jac_total
