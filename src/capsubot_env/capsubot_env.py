from collections import deque
from typing import List

import gym
import numpy as np
import scipy.constants
import scipy.integrate
from gym import spaces
from gym.utils import seeding
from numba import njit
from .capsubot_renderer import Renderer

MIN_VOLTAGE = 0.0
MAX_VOLTAGE = 24.0

MIN_X = -10.0
MAX_X = -MIN_X

MIN_XI = -1.0
MAX_XI = -MIN_XI

MIN_DX = -10.0
MAX_DX = -MIN_DX
MIN_DXI = MIN_DX
MAX_DXI = -MIN_DX


@njit()
def F_step(action, force_max: float) -> float:
    return action * force_max


@njit()
def friction_model(N: float, velocity) -> float:
    return -N * 2 / np.pi * np.arctan(velocity * 10e5)
    # return -np.sign(velocity) * N * self.mu


class CapsubotEnv(gym.Env):
    """A cabsubot with electromagnetic coil"""

    metadata = {"render.modes": ["live", "file", "none", "human"]}
    visualization = None

    def __init__(self):
        super(CapsubotEnv, self).__init__()

        self.total_time = None
        self.state = None
        self.average_speed = 0
        self.M = 0.193
        self.m = 0.074
        # Normal reaction for friction calculation
        self.N = (self.M + self.m) * scipy.constants.g
        self.stiffness = 256.23
        self.force_max = 1.25
        self.mu = 0.29  # Coefficient of friction.

        self.steps_in_period = 200
        self.min_period = 0.01
        self.dt = self.min_period / self.steps_in_period  # Action force discritization.
        # testing ver where the agent can't take an action more than one time per min_period
        self.frame_skip: int = self.steps_in_period
        self.previous_average_speed = 0.0
        self.done = False

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=np.array([MIN_X, MIN_DX, MIN_XI, MIN_DXI]),
            high=np.array([MAX_X, MAX_DX, MAX_XI, MAX_DXI]),
            dtype=np.float32,
        )

        self.viewer = None
        self.seed()
        # self.version_two = version_two  # ver where the agent must reach the endpoint and stop there

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_state(self, state: List[float]) -> np.ndarray:
        """
        Normalizing reward states because NN inside RL agent works better with normalized inputs.
        Because we can't know the true thresholds of the model states, we use that wierd interpolation.
        Thresholds were obtained experimentally.
        """
        state = np.array(state)
        norm_state = [
            np.interp(state[0], [-0.0007666844383611218, 0.07919885629789096], [-1, 1]),
            np.interp(state[1], [-0.18283166534706963, 0.24274101317295704], [-1, 1]),
            np.interp(state[2], [-0.01652187660136813, 0.019895362341591866], [-1, 1]),
            np.interp(state[3], [-1.1832780931204638, 1.1508305412787394], [-1, 1]),
        ]
        return np.array(norm_state)

    def mechanical_model(self, y, force):
        x, x_dot, xi, xi_dot = y
        friction = friction_model(self.N, x_dot)
        x_acc = (self.stiffness * xi - force + friction) / self.M
        xi_acc = (-self.stiffness * xi + force) / self.m - x_acc
        return [x_dot, x_acc, xi_dot, xi_acc]

    def calc_reward(self, av_speed, prev_av_speed, scale_factor=5000) -> float:
        return (av_speed - prev_av_speed) * scale_factor

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        force = F_step(action, self.force_max)

        for _ in range(self.frame_skip):
            x, x_dot, xi, xi_dot = self.state

            # Euler kinematic integration.
            dx = self.mechanical_model(self.state, force)
            self.state = [
                x + self.dt * dx[0],
                x_dot + self.dt * dx[1],
                xi + self.dt * dx[2],
                xi_dot + self.dt * dx[3],
            ]

            self.total_time = self.total_time + self.dt
        self.average_speed = x / self.total_time

        norm_state = self.normalize_state(self.state)
        # TODO: normalize reward
        step_reward = self.calc_reward(self.average_speed, self.previous_average_speed)
        self.previous_average_speed = self.average_speed

        if x >= 0.2:
            self.done = True
            step_reward += 200
        elif x <= -0.05:
            self.done = True
            step_reward -= 500

        return (
            norm_state,
            step_reward,
            self.done,
            {"average_speed": self.average_speed},
        )

    def reset(self):
        self.state = (0, 0, 0, 0)
        self.total_time = 0.0
        self.average_speed = 0.0
        self.previous_average_speed = 0.0
        self.done = False
        self.viewer = Renderer(
            name=f"{self.__class__.__name__}", render_target_region=False
        )
        return np.array(self.state).astype(np.float32)

    def render(self, mode="human"):
        return self.viewer.render(self.state)

    def close(self):
        if self.viewer:
            self.viewer.quit()
            self.viewer = None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class CapsubotEnvToPoint(CapsubotEnv):
    """
    ### Observation Space

    - The position of the center of mass (x)
    - The velocity of the center of mass (x_dot)
    - The position of the inner body (xi)
    - The velocity of the inner body (xi_dot)
    - The coordinate of the goal point (gp_coord)
    - The distance between the centre of mass and the goal point (dist)
    """

    def __init__(
        self,
        goal_point: float = 0.3,  # x point where the robot should stop
        tolerance: float = 0.1,  # tolerance to stop point in percents of goal_point distance
        maxlen_counting_speed: int = 10,  # len of velocities buffer
        left_termination_point: float = -0.08,
    ):
        super(CapsubotEnvToPoint, self).__init__()
        self.goal_point = goal_point
        self.tolerance = goal_point * tolerance
        self.buffer = deque(maxlen=maxlen_counting_speed)
        self.left_termination_point = left_termination_point
        self.right_termination_point = self.goal_point * 1.5
        # FIXME!
        self.observation_space = spaces.Box(
            low=np.array([MIN_X, MIN_DX, MIN_XI, MIN_DXI]),
            high=np.array([MAX_X, MAX_DX, MAX_XI, MAX_DXI]),
            dtype=np.float32,
        )

        # normalize state limit values
        self.left_lim_value = self.goal_point - self.right_termination_point
        self.right_lim_value = self.goal_point - self.left_termination_point

    def normalize_state(self, state: List[float]) -> np.ndarray:
        """
        Normalizing reward states because NN inside RL agent works better with normalized inputs.
        Because we can't know the true thresholds of the model states, we use that wierd interpolation.
        Thresholds were obtained experimentally. Except state[5].
        """
        state = np.array(state)
        norm_state = [
            np.interp(state[0], [-0.0007666844383611218, 0.07919885629789096], [-1, 1]),
            np.interp(state[1], [-0.18283166534706963, 0.24274101317295704], [-1, 1]),
            np.interp(state[2], [-0.01652187660136813, 0.019895362341591866], [-1, 1]),
            np.interp(state[3], [-1.1832780931204638, 1.1508305412787394], [-1, 1]),
            state[4],
            np.interp(state[5], [self.left_lim_value, self.right_lim_value], [-1, 1]),
        ]
        return np.array(norm_state)

    def calc_reward(
        self, current_pos: float, velocity: float, scale_factor: int = 10
    ) -> int:  # FIXME
        # we don't want situation when 1speed + (-2speed) + ... = 0. Using abs to avoid it
        self.buffer.append(abs(velocity))

        # if inside target region
        if self.goal_point <= current_pos <= self.goal_point + self.tolerance:
            reward = 1
            # if velocity is small enough
            if sum(self.buffer) <= 1e-3:
                reward = 5000
            # needs to decrease velocity to reach positive reward
            else:
                reward -= sum(self.buffer) * scale_factor
        # if the robot goes backward or to far from the target region
        elif (
            current_pos <= self.left_termination_point
            or current_pos >= self.right_termination_point
        ):
            reward = -3000
        # if still not inside the target region
        else:
            reward = -0.5
        return reward

    def termination(self, current_pos) -> bool:
        if (self.goal_point <= current_pos <= self.goal_point + self.tolerance) and (
            sum(self.buffer) <= 1e-3
        ):
            self.done = True
        elif (
            current_pos < self.left_termination_point
            or current_pos > self.right_termination_point
        ):
            self.done = True
        return self.done

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        force = F_step(action, self.force_max)

        for _ in range(self.frame_skip):
            x, x_dot, xi, xi_dot, gp_coord, dist = self.state

            # Euler kinematic integration.
            dx = self.mechanical_model(self.state, force)
            self.state = [
                x + self.dt * dx[0],
                x_dot + self.dt * dx[1],
                xi + self.dt * dx[2],
                xi_dot + self.dt * dx[3],
                gp_coord,
                gp_coord - x,
            ]

            self.total_time = self.total_time + self.dt
        self.average_speed = x / self.total_time

        norm_state = self.normalize_state(self.state)
        step_reward = self.calc_reward(current_pos=x, velocity=x_dot)
        done = self.termination(current_pos=x)
        info = {
            "average_speed": self.average_speed,
            "current_pos": x,
            "center_mass_velocity": x_dot,
            "dones": int(self.done),
        }

        return norm_state, step_reward, done, info

    def reset(self):
        self.state = (0, 0, 0, 0, self.goal_point, self.goal_point)
        self.total_time = 0.0
        self.average_speed = 0.0
        self.previous_average_speed = 0.0
        self.done = False
        self.viewer = Renderer(
            name=f"{self.__class__.__name__}", render_target_region=False
        )
        return np.array(self.state).astype(np.float32)

    def render(self, mode="human"):
        return self.viewer.render(self.state)
