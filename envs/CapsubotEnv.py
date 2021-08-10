import random
import json
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import numpy as np
from scipy.constants import g as GRAVITY_CONSTANT
import scipy.integrate

MIN_VOLTAGE = 0.0
MAX_VOLTAGE = 24.0

MIN_X = -10.0
MAX_X = -MIN_X

MIN_XI = -10.0
MAX_XI = -MIN_XI

MIN_DX = -1.0
MAX_DX = -MIN_DX
MIN_DXI = MIN_DX
MAX_DXI = -MIN_DX


class CapsubotEnv(gym.Env):
    """A cabsubot with electromagnetic coil"""

    metadata = {"render.modes": ["live", "file", "none"]}
    visualization = None

    def __init__(self):
        super(CapsubotEnv, self).__init__()

        # self.is_right_movement = True
        self.total_time = None
        self.state = None
        self.M = 0.193
        self.m = 0.074
        self.stiffness = 256.23
        self.force_max = 1.25
        self.mu = 0.29 # Coefficient of friction.
        self.dt = 0.01 # Action force discritization.

        self.action_space = spaces.Box(
            low=MIN_VOLTAGE, high=MAX_VOLTAGE, shape=(1, 1)  # mb need to cast to nparray?
        )

        self.observation_space = spaces.Box(
            low=np.array([MIN_X, MIN_DX, MIN_XI, MIN_DXI]),
            high=np.array([MAX_X, MAX_DX, MAX_XI, MAX_DXI]),
            shape=(4,),
            dtype=np.float16,
        )


        self.viewer = None
        self.seed()
        self.reset()

        # add physical parameters

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def force_model(self, voltage):
        return voltage

    def friction_model(self, velocity):
        N = (self.M + self.m) * GRAVITY_CONSTANT
        return -np.sign(velocity) * N * self.mu

    def mechanical_model(self, y, t, force):
        x, x_dot, xi, xi_dot = y
        friction = self.friction_model(x_dot)
        x_acc = (self.stiffness * xi - force + friction) / self.M
        xi_acc = (-self.stiffness * xi + force) / self.m - x_acc
        return [x_dot, x_acc, xi_dot, xi_acc]

    def step(self, action, integrator="ode"):
        x, x_dot, xi, xi_dot = self.state
        force = self.force_model(action)

        if (integrator=="ode"):
            y0 = [x, x_dot, xi, xi_dot]
            t = [0, self.dt]
            sol = scipy.integrate.odeint(self.mechanical_model, y0, t, args=(force,)) # Need to investigate why euler integration didn't work
            self.state = [item for item in sol[-1]]
        else:
            # Euler kinematic integration.
            dx = self.mechanical_model(self.state, 0, force)
            self.state = [
                x + self.dt * dx[0],
                x_dot + self.dt * dx[1],
                xi + self.dt * dx[2],
                xi_dot + self.dt * dx[3],
            ]

        self.total_time = self.total_time + self.dt
        average_speed = x / self.total_time

        reward = 1 if average_speed > 0 and x_dot > 0 else 0  # FIXME
        done = x < -1 or x > 5  # FIXME

        return self.state, reward, done, {}

    def reset(self):
        self.state = (0, 0, 0, 0)
        self.total_time = 0.0
        return self.state

    def render(self, mode="human"):
        screen_width = 1280
        screen_height = 400

        capsule_length = 100.0
        capsule_height = 30.0

        world_width = 5.0
        scale = screen_width / world_width

        inner_body_length = capsule_length / 2.0
        inner_body_height = capsule_height
        inner_body_y = capsule_height

        ground_level = 200

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Capsule polygon.
            l, r, t, b = (
                -capsule_length / 2,
                capsule_length / 2,
                capsule_height / 2,
                -capsule_height / 2,
            )
            capsule = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.capsule_transform = rendering.Transform()
            capsule.add_attr(self.capsule_transform)
            self.viewer.add_geom(capsule)

            # Inner body polygon
            l, r, t, b = (
                -inner_body_length / 2,
                inner_body_length / 2,
                inner_body_height / 2,
                -inner_body_height / 2,
            )
            inner_body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.inner_body_transform = rendering.Transform()
            inner_body.add_attr(self.inner_body_transform)
            inner_body.add_attr(self.capsule_transform)
            inner_body.set_color(0.8, 0.6, 0.4)
            self.viewer.add_geom(inner_body)

            # Ground surface
            self.track = rendering.Line((0, ground_level), (screen_width, ground_level))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x, x_dot, xi, xi_dot = self.state
        capsule_x = x * scale + screen_width / 2.0  # MIDDLE OF CART
        capsule_y = ground_level + capsule_height / 2.0  # MIDDLE OF CART
        self.capsule_transform.set_translation(capsule_x, capsule_y)

        inner_body_x = xi * scale  # MIDDLE OF CART
        self.inner_body_transform.set_translation(inner_body_x, inner_body_y)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
