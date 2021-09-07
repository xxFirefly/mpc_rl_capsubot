import random
import json
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import numpy as np
import scipy.constants
import scipy.integrate
import pyglet
import math

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


class CapsubotEnv(gym.Env):
    """A cabsubot with electromagnetic coil"""

    metadata = {"render.modes": ["live", "file", "none"]}
    visualization = None

    def __init__(self, force="trivial"):
        super(CapsubotEnv, self).__init__()
        self.force = force

        # self.is_right_movement = True
        self.total_time = None
        self.state = None
        self.average_speed = 0
        self.M = 0.193
        self.m = 0.074
        self.stiffness = 256.23
        self.force_max = 1.25
        self.mu = 0.29  # Coefficient of friction.

        steps_in_period = 200
        min_period = 0.01
        self.dt = min_period / steps_in_period  # Action force discritization.
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=np.array([MIN_X, MIN_DX, MIN_XI, MIN_DXI]),
            high=np.array([MAX_X, MAX_DX, MAX_XI, MAX_DXI]),
            dtype=np.float32,
        )

        self.viewer = None
        self.seed()

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def force_model(self, voltage, force="trivial"):
        if force == "trivial":
            return voltage
        if force == "step":
            return 1.25 if voltage == 1 else 0.0
        if force == "em":
            # Force model from Nikita.
            v = voltage
            alpha = math.log10(mu_r)
            # Where is R? Active resistance of the coil.
            r_a = 1  # Radius of of the coil winding
            r_0 = 0.75  # Inner radius of the coil
            lamda = rho / alpha  # Or here is a?
            part1 = -v * v * mu_r * mu_0 / 8 / math.pi / lamda / lamda / l / l
            part2 = (r0 / r_a) * (r0 / r_a) * alpha * np.exp(-alpha / l * x)
            return part1 * part2

    def friction_model(self, velocity):
        N = (self.M + self.m) * scipy.constants.g
        return -N * 2 / np.pi * np.arctan(velocity * 10e5)
        # return -np.sign(velocity) * N * self.mu

    def mechanical_model(self, y, t, force):
        x, x_dot, xi, xi_dot = y
        friction = self.friction_model(x_dot)
        x_acc = (self.stiffness * xi - force + friction) / self.M
        xi_acc = (-self.stiffness * xi + force) / self.m - x_acc
        return [x_dot, x_acc, xi_dot, xi_acc]

    def step(self, action, integrator="euler"):
        x, x_dot, xi, xi_dot = self.state
        force = self.force_model(action, force=self.force)  # Choose right force model.

        if integrator == "ode":
            y0 = [x, x_dot, xi, xi_dot]
            t = [0, self.dt]
            sol = scipy.integrate.odeint(
                self.mechanical_model, y0, t, args=(force,)
            )  # Need to investigate why euler integration didn't work
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
        self.average_speed = x / self.total_time

        reward = 1 if self.average_speed > 0 and x_dot > 0 else 0  # FIXME
        done = x < -1 or x > 5  # FIXME

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = (0, 0, 0, 0)
        self.total_time = 0.0
        return np.array(self.state).astype(np.float32)

    def render(self, mode="human"):
        screen_width = 1280
        screen_height = 400

        capsule_length = 100.0
        capsule_height = 30.0

        world_width = 1.0
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

            # Score
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=screen_width * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )

        if self.state is None:
            return None

        x, x_dot, xi, xi_dot = self.state
        capsule_x = x * scale + screen_width / 2.0  # MIDDLE OF CART
        capsule_y = ground_level + capsule_height / 2.0  # MIDDLE OF CART
        self.capsule_transform.set_translation(capsule_x, capsule_y)

        inner_body_x = xi * scale  # MIDDLE OF CART
        self.inner_body_transform.set_translation(inner_body_x, inner_body_y)

        self.score_label.text = "%04i" % self.average_speed
        self.score_label.draw()

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
