import numpy as np
import scipy.constants
from numba import njit
from collections import deque
from typing import List


def classic_force_model(t, T, tau, pr=False):
    """
    Defines electromagnteic force of coil
    """
    if pr:
        print(f"tau {tau}")
    return (1.0 - 2.0 / np.pi * np.arctan((np.modf(t / T)[0] - tau) * 10.0e5)) / 2.0


class Capsubot:
    """
    basic implementation of the capsubot (capsule robot) agent mechanics

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dimensions:

    stiffness [N/m]
    M [kg]
    m [kg]
    N [N]
    dt [s]
    total_time [s]
    average_speed [m/s]
    state [m, m/s, m, m/s]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def __init__(self, dt: float, frame_skip: int, model: int = 0, log = False):
        if model == 0:
            self._stiffness = 256.23
            self._M = 0.193
            self._m = 0.074
            mu = 0.29  # coefficient of friction
            self._N = mu * (self._M + self._m) * scipy.constants.g
            self._force_max = 1.25
        elif model == 1:
            self._stiffness = 360.0
            self._M = 0.0213
            self._m = 0.0231
            self._N = 0.7
            self._force_max = 0.8
        else:
            raise Exception("Wrong model parameter.")

        self.model_type = model

        # Variables for dimmensionless comparison
        self.omega = np.sqrt(self._stiffness * (self._M + self._m) / self._M / self._m)
        self.L = self._force_max / self._stiffness

        self._dt = dt
        self._average_speed = 0.0
        self._frame_skip = frame_skip
        self._total_time = None
        self._state = None
        self.log = log
        self.total_time_buffer = deque()
        self.action_buffer = deque()
        self.x_buffer = deque()
        self.x_dot_buffer = deque()
        self.reset()

    def step(self, action) -> None:
        force = self._F_step(action, self._force_max)
        err_message = "you forgot to call the reset method!"
        assert self._state is not None, err_message

        for _ in range(self._frame_skip):
            x, x_dot, xi, xi_dot = self._state

            # Euler kinematic integration.
            dx = self._mechanical_model(self._state, force)
            self._state = [
                x + self._dt * dx[0],
                x_dot + self._dt * dx[1],
                xi + self._dt * dx[2],
                xi_dot + self._dt * dx[3],
            ]
            self._total_time = self._total_time + self._dt


            # uncomment this section only if you need to log hi rez values
            # it's very slow

            if self.log and _ % 5 == 0:
                self.total_time_buffer.append(self._total_time)
                self.action_buffer.append(action)
                self.x_buffer.append(x)
                self.x_dot_buffer.append(x_dot)


        self._average_speed = x / self._total_time

    def step_force(self, unit_force) -> None:
        """
        Args:
            unit_force: function with signgle parameter as time in seconds and range of fuction between 0 and 1.
        Returns:
            agent state as [time, body_position, body_velocity, inner_body_position, inner_body_velocity]
        """
        err_message = "you forgot to call the reset method!"
        assert self._state is not None, err_message

        for _ in range(self._frame_skip):
            x, x_dot, xi, xi_dot = self._state

            # Euler kinematic integration.
            dx = self._mechanical_model(self._state, self._force_max*unit_force(self._total_time))
            self._state = [
                x + self._dt * dx[0],
                x_dot + self._dt * dx[1],
                xi + self._dt * dx[2],
                xi_dot + self._dt * dx[3],
            ]
            self._total_time = self._total_time + self._dt

            # uncomment this section only if you need to log hi rez values
            # it's very slow

            if self.log and _ % 5 == 0:
                self.total_time_buffer.append(self._total_time)
                self.action_buffer.append(unit_force(self._total_time))
                self.x_buffer.append([x, x_dot, xi, xi_dot])
                self.x_dot_buffer.append(x_dot)

        self._average_speed = x / self._total_time

    def reset(self) -> None:
        self._average_speed = 0.0
        self._total_time = 0.0
        self._state = [0.0, 0.0, 0.0, 0.0]
        self.total_time_buffer.clear()
        self.action_buffer.clear()

    def set_state(self, state : List[float]) -> None:
        self._state = state.copy()

    def set_time(self, initial_time : float) -> None:
        self._total_time = initial_time

    @property
    def get_state(self) -> np.ndarray:
        return np.array(self._state).astype(np.float64)

    @property
    def get_dimless_state(self) -> np.ndarray:
        return np.ndarray(
            [
                self._state[0] / self.L,
                self._state[1] / self.L / self.omega,
                self._state[2] / self.L,
                self._state[3] / self.L / self.omega,
            ]
        )

    @property
    def get_dimless_total_time(self) -> float:
        return self._total_time * self.omega

    @property
    def get_average_speed(self) -> float:
        return self._average_speed

    @property
    def get_total_time(self) -> float:
        return self._total_time

    @property
    def get_total_time_buffer(self) -> deque:
        return self.total_time_buffer

    @property
    def get_action_buffer(self) -> deque:
        return self.action_buffer

    @property
    def get_x_buffer(self) -> deque:
        return self.x_buffer

    @property
    def get_x_dot_buffer(self) -> deque:
        return self.x_dot_buffer

    def _mechanical_model(self, obs_state, force):
        x, x_dot, xi, xi_dot = obs_state
        friction = self._friction_model(self._N, x_dot)
        x_acc = (self._stiffness * xi - force + friction) / self._M
        xi_acc = (-self._stiffness * xi + force) / self._m - x_acc
        return [x_dot, x_acc, xi_dot, xi_acc]

    @staticmethod
    @njit()
    def _F_step(action, force_max: float) -> float:
        return action * force_max

    @staticmethod
    @njit()
    def _friction_model(force: float, velocity) -> float:
        return -force * 2 / np.pi * np.arctan(velocity * 10e5)
