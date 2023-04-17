import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..capsubot_env import CapsubotEnv
from .capsubot_base import CapsubotMk2

MIN_VOLTAGE = 0.0  # Volts
MAX_VOLTAGE = 24.0

MIN_X = -10.0  # Meters
MAX_X = -MIN_X

MIN_XI = -1.0  # Meters
MAX_XI = -MIN_XI

MIN_DX = -10.0  # Meters per second
MAX_DX = -MIN_DX
MIN_DXI = MIN_DX
MAX_DXI = -MIN_DX

MAX_GOAL_POINT = 1.0  # Meters
MIN_GOAL_POINT = -0.5

MAX_DIST = 3.0  # Meters
MIN_DIST = -MAX_DIST

# PWM params
TAU_MIN = 0.1  # was 0.05
TAU_MAX = 1.0
T_MIN = 0.05   # was 0.008
T_MAX = 0.3


class CapsubotEnvMk2(CapsubotEnv):
    """
    In this version of the Capsubot env we implemented task in which agent should
    optimize PWM parameters like T (signal period) and mju (signal duty cycle)

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dimensions:

    average_speed [m/s]
    agent_state [m, m/s, m, m/s]
    min_period [s]
    dt [s]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This env supports only human mode rendering or none
    """

    def __init__(self, is_render=False, rendering_fps: int = 60, model: int = 0):
        super(CapsubotEnvMk2, self).__init__(is_render, rendering_fps, model)

        self.action_space = spaces.Box(low=np.array([T_MIN, TAU_MIN]),
                                       high=np.array([T_MAX, TAU_MAX]),
                                       dtype=np.float64,
                                       )

        self.agent = CapsubotMk2(self.dt, self.frame_skip, model)

    def step(self, action: np.ndarray):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        self.agent.step(action)
        self.agent_state = self.agent.get_state
        self.average_speed = self.agent.get_average_speed

        norm_state = self._normalize_state(self.agent_state)
        step_reward = self._calc_reward(self.average_speed, self.previous_average_speed)
        self.previous_average_speed = self.average_speed

        x = self.agent_state[0]

        if x >= self.right_termination_point:
            self.done = True
            step_reward += 200
        elif x <= self.left_termination_point:
            self.done = True
            step_reward -= 500

        return (
            norm_state,
            step_reward,
            self.done,
            {
                "average_speed": self.average_speed,
                "obs_state": self.agent_state,
                "total_time": self.agent.get_total_time,
                # uncomment this section only if you need to log hi rez values
                # it's very slow
                #
                # "total_time_deque": self.agent.get_total_time_buffer,
                # "T_deque": self.agent.get_T_buffer,
                # "tau_deque": self.agent.get_tau_buffer,
                # "x_deque": self.agent.get_x_buffer,
                # "x_dot_deque": self.agent.get_x_dot_buffer,
            },
        )
