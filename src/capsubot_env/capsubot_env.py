import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .capsubot import Capsubot
from .capsubot_renderer import Renderer

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


class CapsubotEnv(gym.Env):
    """
    A cabsubot with electromagnetic coil

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
        super(CapsubotEnv, self).__init__()
        if model == 0:
            # self.steps_in_period = 200
            # self.min_period = 0.001  # maybe it could be smaller (was 0.01)
            # self.dt = self.min_period / self.steps_in_period  # Action force discritization.
            # testing ver where the agent can't take an action more than one time per min_period
            # self.frame_skip: int = 200  # should be like min_period / n. Mb smaller than now

            self.steps_in_period = 200
            self.min_period = 0.01
            self.dt = self.min_period / self.steps_in_period
            self.frame_skip: int = 200
        elif model == 1:
            # signal changes no more frequently than
            self.min_period = 0.01  # was 0.0005
            # number of steps performing by model with chosen signal params
            self.steps_in_period = 1000
            self.frame_skip: int = 1000
            # integrator resolution
            self.dt = self.min_period / self.steps_in_period
        else:
            raise Exception("Wrong model parameter.")

        self.average_speed = 0
        self.agent_state = None
        self.previous_average_speed = 0.0
        self.done = False
        self.left_termination_point: float = -0.1
        self.right_termination_point: float = 0.3
        self.rendering_fps = rendering_fps

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([MIN_X, MIN_DX, MIN_XI, MIN_DXI]),
            high=np.array([MAX_X, MAX_DX, MAX_XI, MAX_DXI]),
            dtype=np.float64,
        )

        if is_render:
            self.viewer = Renderer(
                name=f"{self.__class__.__name__}", render_target_region=False
            )
        else:
            self.viewer = None

        self.agent = Capsubot(self.dt, self.frame_skip, model)
        self.seed()

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def _normalize_state(state: np.ndarray) -> np.ndarray:
        
        # Normalizing reward states because NN inside RL agent works better with normalized inputs.
        # Normalizing reward states because NN inside RL agent works better with normalized inputs.
        # Because we can't know the true thresholds of the model states, we use that wierd interpolation.
        # Thresholds were obtained experimentally.
        
        norm_state = [
            np.interp(state[0], [-0.1, 0.3], [-1.0, 1.0]),
            np.interp(state[1], [-0.36, 0.48], [-1.0, 1.0]),
            np.interp(state[2], [-0.033, 0.04], [-1.0, 1.0]),
            np.interp(state[3], [-2.36, 2.3], [-1.0, 1.0]),
        ]

        return np.array(norm_state)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        self.agent.step(action)
        self.agent_state = self.agent.get_state
        self.average_speed = self.agent.get_average_speed

        x = self.agent_state[0]
        self._termination_checker(x_pose=x)

        norm_state = self._normalize_state(self.agent_state)
        step_reward = self._reward_calculator(x_pose=x)
        self.previous_average_speed = self.average_speed

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
                # "action_deque": self.agent.get_action_buffer,
                # "x_deque": self.agent.get_x_buffer,
                # "x_dot_deque": self.agent.get_x_dot_buffer,
            },
        )

    def reset(self):
        self.previous_average_speed = 0.0
        self.done = False
        self.agent.reset()
        self.agent_state = self.agent.get_state
        return self.agent_state

    def render(self):
        if self.viewer:
            return self.viewer.render(
                time=self.agent.get_total_time,
                state=self.agent_state,
                fps=self.rendering_fps,
            )

    def _termination_checker(self, x_pose: float) -> None:
        if x_pose >= self.right_termination_point:
            self.done = True
        elif x_pose <= self.left_termination_point:
            self.done = True

    def _reward_calculator(self, x_pose, scale_factor: int = 5000) -> float:
        reward = (self.average_speed - self.previous_average_speed) * scale_factor

        if x_pose >= self.right_termination_point:
            reward = 1000
        elif x_pose <= self.left_termination_point:
            reward = -1000
        return reward

    def close(self):
        if self.viewer:
            self.viewer.quit()
            self.viewer = None

    @property
    def model_logs(self):
        mechanical_params: dict[str, float] = self.agent.get_mechanical_params
        env_params: dict[str, float] = {"steps_in_period": self.steps_in_period,
                                        "min_period": self.min_period,
                                        "dt": self.dt,
                                        "frame_skip": self.frame_skip,
                                        }
        return {"mechanical_params": mechanical_params,
                "env_params": env_params,
                }
