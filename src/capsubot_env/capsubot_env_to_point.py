from collections import deque
from typing import List, Dict, Optional

import numpy as np
from gym import spaces

from .capsubot import Capsubot
from .capsubot_env import CapsubotEnv
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

MAX_GOAL_POINT = 1.0
MIN_GOAL_POINT = -0.5

MAX_DIST = 3.0
MIN_DIST = -MAX_DIST


class CapsubotEnvToPoint(CapsubotEnv):
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Observation Space

    # agent (state)
    - The position of the center of mass (x)
    - The velocity of the center of mass (x_dot)
    - The position of the inner body (xi)
    - The velocity of the inner body (xi_dot)

    # target (_target_state)
    - The coordinate of the goal point
    - The distance between the centre of mass and the goal point
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dimensions:

    goal_point [m]
    target_state [m, m]
    agent_state [m, m/s, m, m/s]
    average_speed [m/s]
    tolerance [m]
    total_time [s]
    dt [s]
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def __init__(
        self,
        tolerance: float = 0.02,  # tolerance to stop point (2 cm)
        maxlen_counting_speed: int = 15,  # len of velocities buffer
        termination_distance: float = 0.08,
        is_render: bool = False,
        rendering_fps: int = 60,
        model: int = 0,
        goal_point: Optional[float] = None,
        is_inference: bool = False,
    ):
        super(CapsubotEnvToPoint, self).__init__(is_render, rendering_fps, model)
        self.goal_point: Optional[float] = goal_point
        self.goal_point_buffer: List[float] = []
        self.right_termination_point = None
        self.left_lim: Optional[float] = None
        self.right_lim: Optional[float] = None
        self._target_state: Optional[np.ndarray] = None
        self.observation: Optional[Dict[str, np.ndarray]] = None
        self.tolerance = tolerance
        self.velocity_buffer = deque(maxlen=maxlen_counting_speed)
        self.termination_distance = termination_distance
        self.left_termination_point = -termination_distance
        self.random_generator = self._seed()
        self.is_inference = is_inference

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([MIN_X, MIN_DX, MIN_XI, MIN_DXI]),
                    high=np.array([MAX_X, MAX_DX, MAX_XI, MAX_DXI]),
                    dtype=np.float64,
                ),
                "target": spaces.Box(
                    low=np.array([MIN_GOAL_POINT, MAX_DIST]),
                    high=np.array([MAX_GOAL_POINT, MIN_DIST]),
                    dtype=np.float64,
                ),
            }
        )

        self.agent = Capsubot(self.dt, self.frame_skip)
        if is_render:
            self.viewer = Renderer(f"{self.__class__.__name__}", True)
        else:
            self.viewer = None

        # calling reset method to set the goal_point and values that depends on it
        self.reset()

        # TODO: implement version for training with random goal point and version for validation
        # TODO: add checking acceleration of the center of mass to end episode

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        self.agent.step(action)
        self.agent_state = self.agent.get_state
        self.average_speed = self.agent.get_average_speed

        x = self.agent_state[0]
        x_dot = self.agent_state[1]
        self._target_state = np.array([self.goal_point, self.goal_point - x])
        self.observation = self._get_observation

        norm_observation = self._normalize_obs(self.observation)
        step_reward = self._calc_reward(current_pos=x, velocity=x_dot)
        self._termination_checker(x_pose=x)

        info = {
            "obs_state": self.observation,
            "average_speed": self.average_speed,
            "total_time": self.agent.get_total_time,
            "goal_point": self.goal_point,
            # uncomment this section only if you need to log hi rez values
            # it's very slow
            #
            # "total_time_deque": self.agent.get_total_time_buffer,
            # "action_deque": self.agent.get_action_buffer,
            # "x_deque": self.agent.get_x_buffer,
            # "x_dot_deque": self.agent.get_x_dot_buffer,
        }

        return norm_observation, step_reward, self.done, info

    def reset(self):
        if not self.is_inference:
            # randomly spawns the goal_point in interval [0.2, 1.5)
            self.goal_point = self.random_generator.uniform(0.2, 1.5)
            self.goal_point_buffer.append(self.goal_point)

        assert self.goal_point != 0.0, "target point is 0"
        self.right_termination_point = self.goal_point + self.termination_distance

        # normalize state limit values
        self.left_lim = self.left_termination_point
        self.right_lim = self.right_termination_point

        self.agent.reset()
        self.agent_state = self.agent.get_state
        self._target_state = np.array([self.goal_point, self.goal_point])
        self.observation = self._get_observation
        self.previous_average_speed = 0.0
        self.done = False

        if self.viewer:
            self.viewer.goal_point = self.goal_point
            self.viewer.tolerance = self.tolerance

        return self.observation

    def render(self, mode="human"):
        if self.viewer:
            return self.viewer.render(
                self.agent.get_total_time,
                self.agent_state,
                self.rendering_fps,
                self.goal_point,
            )

    def _is_inside_target_region(self, current_pos: float) -> bool:
        # if the agent inside the target region
        return (
            self.goal_point - self.tolerance / 2
            <= current_pos
            <= self.goal_point + self.tolerance / 2
        )

    def _is_outside_of_working_zone(self, current_pos: float) -> bool:
        # if the agent goes backward or too far from the target region
        return (current_pos <= self.left_termination_point) or (
            current_pos >= self.right_termination_point
        )

    def _normalize_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalizing reward states because NN inside RL agent works better with normalized inputs.
        Because we can't know the true thresholds of the model states, we use that wierd interpolation.
        Thresholds were obtained experimentally, except target_state[0].
        """
        state = obs.get("agent")
        target_state = obs.get("target")

        assert self.left_lim and self.right_lim is not None, "norm lims are None!"

        norm_agent_state = [
            np.interp(
                state[0],
                [self.left_termination_point, self.right_termination_point],
                [-1.0, 1.0],
            ),
            np.interp(state[1], [-0.36, 0.48], [-1.0, 1.0]),
            np.interp(state[2], [-0.033, 0.04], [-1.0, 1.0]),
            np.interp(state[3], [-2.36, 2.3], [-1.0, 1.0]),
        ]
        norm_target_state = [
            target_state[0],
            np.interp(target_state[1], [self.left_lim, self.right_lim], [-1.0, 1.0]),
        ]
        return {
            "agent": np.array(norm_agent_state),
            "target": np.array(norm_target_state),
        }

    def _calc_reward(
        self, current_pos: float, velocity: float, reward_scale_factor: int = 5
    ) -> int:
        # we don't want situation when 1speed + (-2speed) + ... = 0. Using abs to avoid it
        self.velocity_buffer.append(abs(velocity))
        mean_velocity = np.mean(self.velocity_buffer)

        if self._is_inside_target_region(current_pos):
            # if velocity is small enough
            if mean_velocity <= 1e-3:
                reward = 2000
            # needs to decrease velocity to reach positive reward
            else:
                reward = -mean_velocity * reward_scale_factor

        elif self._is_outside_of_working_zone(current_pos):
            reward = -2000
        # if still not inside the target region
        else:
            current_pos = np.clip(current_pos, 0.05, self.right_termination_point)
            temp_reward = np.clip((self.goal_point / current_pos), 1.0, 24.0)

            right_boundary_of_norm = self.goal_point / 0.05
            reward = -(temp_reward - 1.0) / (right_boundary_of_norm - 1.0)

        # if still not inside the target region
        """
        else:
            reward = np.clip(current_pos / self.goal_point, -0.5, 1.0)
        """

        return reward

    def _termination_checker(self, x_pose: float) -> None:
        if (self._is_inside_target_region(x_pose)) and (
            np.mean(self.velocity_buffer) <= 1e-3
        ):
            self.done = True

        elif self._is_outside_of_working_zone(x_pose):
            self.done = True

    @property
    def _get_observation(self) -> Dict[str, np.ndarray]:
        return {"agent": self.agent_state, "target": self._target_state}

    @staticmethod
    def _seed(seed=42):
        return np.random.default_rng(seed)


class CapsubotEnvToPoint2(CapsubotEnvToPoint):
    """
    It differs from the original CapsubotEnvToPoint by observation space
    and other funcs that depends on observation space.
    """

    def __init__(
        self,
        tolerance: float = 0.02,  # tolerance to stop point (2 cm)
        maxlen_counting_speed: int = 15,  # len of velocities buffer
        termination_distance: float = 0.08,
        is_render: bool = False,
        rendering_fps: int = 60,
        model: int = 0,
        goal_point: Optional[float] = None,
        is_inference: bool = False,
    ):
        super(CapsubotEnvToPoint2, self).__init__(tolerance,
                                                  maxlen_counting_speed,
                                                  termination_distance,
                                                  is_render,
                                                  rendering_fps,
                                                  model,
                                                  goal_point,
                                                  is_inference,
                                                  )
        self.velocity_buffer = deque(maxlen=maxlen_counting_speed)
        self.termination_distance = termination_distance
        self.left_termination_point = -termination_distance

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([MIN_X, MIN_DX, MIN_XI, MIN_DXI]),
                    high=np.array([MAX_X, MAX_DX, MAX_XI, MAX_DXI]),
                    dtype=np.float64,
                ),
                "target": spaces.Box(
                    low=np.array([MIN_GOAL_POINT]),
                    high=np.array([MAX_GOAL_POINT]),
                    dtype=np.float64,
                ),
            }
        )

        self.agent = Capsubot(self.dt, self.frame_skip)
        if is_render:
            self.viewer = Renderer(f"{self.__class__.__name__}", True)
        else:
            self.viewer = None

        # calling reset method to set the goal_point and values that depends on it
        self.reset()

        # TODO: implement version for training with random goal point and version for validation
        # TODO: add checking acceleration of the center of mass to end episode

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        self.agent.step(action)
        self.agent_state = self.agent.get_state
        self.average_speed = self.agent.get_average_speed

        x = self.agent_state[0]
        x_dot = self.agent_state[1]
        self._target_state = np.array([self.goal_point])
        self.observation = self._get_observation

        norm_observation = self._normalize_obs(self.observation)
        step_reward = self._calc_reward(current_pos=x, velocity=x_dot)
        self._termination_checker(x_pose=x)

        info = {
            "obs_state": self.observation,
            "average_speed": self.average_speed,
            "total_time": self.agent.get_total_time,
            "goal_point": self.goal_point,
            # uncomment this section only if you need to log hi rez values
            # it's very slow
            #
            # "total_time_deque": self.agent.get_total_time_buffer,
            # "action_deque": self.agent.get_action_buffer,
            # "x_deque": self.agent.get_x_buffer,
            # "x_dot_deque": self.agent.get_x_dot_buffer,
        }

        return norm_observation, step_reward, self.done, info

    def reset(self):
        if not self.is_inference:
            # randomly spawns the goal_point in interval [0.2, 1.5)
            self.goal_point = self.random_generator.uniform(0.2, 1.5)
            self.goal_point_buffer.append(self.goal_point)

        assert self.goal_point != 0.0, "target point is 0"
        self.right_termination_point = self.goal_point + self.termination_distance

        self.agent.reset()
        self.agent_state = self.agent.get_state
        self._target_state = np.array([self.goal_point])
        self.observation = self._get_observation
        self.previous_average_speed = 0.0
        self.done = False

        if self.viewer:
            self.viewer.goal_point = self.goal_point
            self.viewer.tolerance = self.tolerance

        return self.observation

    def _normalize_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalizing reward states because NN inside RL agent works better with normalized inputs.
        Because we can't know the true thresholds of the model states, we use that wierd interpolation.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        NOTE: We don't normalize the target state in this version of the env
        """
        state = obs.get("agent")

        norm_agent_state = [
            np.interp(
                state[0],
                [self.left_termination_point, self.right_termination_point],
                [-1.0, 1.0],
            ),
            np.interp(state[1], [-0.36, 0.48], [-1.0, 1.0]),
            np.interp(state[2], [-0.033, 0.04], [-1.0, 1.0]),
            np.interp(state[3], [-2.36, 2.3], [-1.0, 1.0]),
        ]

        return {
            "agent": np.array(norm_agent_state),
            "target": obs.get("target"),
        }
