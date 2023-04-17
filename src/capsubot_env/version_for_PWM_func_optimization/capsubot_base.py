import numpy as np
from numba import njit
from collections import deque
from ..capsubot import Capsubot


class CapsubotMk2(Capsubot):
    """
    basic implementation of the capsubot (capsule robot) agent mechanics
    with new force model that represented PWM func

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dimensions:

    stiffness [N/m]
    M [kg]
    m [kg]
    friction_force [N]
    dt [s]
    total_time [s]
    average_speed [m/s]
    state [m, m/s, m, m/s]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    
    def __init__(self, dt: float, frame_skip: int, model: int = 0):
        super(CapsubotMk2, self).__init__(dt, frame_skip, model)
        self.T_buffer = deque()
        self.tau_buffer = deque()

    def step(self, action: np.ndarray) -> None:
        err_message = "you forgot to call the reset method!"
        assert self._state is not None, err_message

        for _ in range(self._frame_skip):
            force = self._F_step(t=self._total_time,
                                 force_max=self._force_max,
                                 T=action[0],
                                 tau=action[1],
                                 )

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

            """
            # uncomment this section only if you need to log hi rez values
            # it's very slow

            self.total_time_buffer.append(self._total_time)
            self.T_buffer.append(action[0])
            self.tau_buffer.append(action[1])
            self.x_buffer.append(x)
            self.x_dot_buffer.append(x_dot)
            """

        self._average_speed = x / self._total_time

    @property
    def get_T_buffer(self) -> deque:
        return self.T_buffer

    @property
    def get_tau_buffer(self) -> deque:
        return self.tau_buffer

    @staticmethod
    @njit()
    def _F_step(t: float, force_max: float, T: float = 0.1, tau: float = 0.1) -> float:
        part = t / T - t // T
        return force_max if part < tau else 0
