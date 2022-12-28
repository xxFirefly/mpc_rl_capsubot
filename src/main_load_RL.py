import os

import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import pandas as pd

from src.capsubot_env.capsubot_env import CapsubotEnv

from src.capsubot_env.capsubot_env_to_point import CapsubotEnvToPoint

# insert into model_path the path to the model *.zip
# it can't be hardcoded because of using datetime module
models_dir: str = os.path.join("..", "RL_WIP", "RL_data_store", "models")
model_path: str = os.path.join(
    models_dir,
    "CapsubotEnvToPoint",
    "PPO_envs-1_LR-0003_steps-4096_epochs-10_MultiInputPolicy_17-12-2022-08",
    "3500000",
)


def printer(array: list) -> None:
    array = np.array(array)
    print(f"min_value = {np.amin(array)}, max_value = {np.amax(array)}")
    print("-----------------------------------------------------------")


model = PPO.load(model_path)

env = CapsubotEnvToPoint(is_render=True)
obs = env.reset()
n_steps = int(5.0 / env.dt)
rewards = []
xs = []
x_dots = []
xis = []
xi_dots = []
total_times = []
actions = []
states = {"x": xs, "x_dot": x_dots, "xi": xis, "xi_dot": xi_dots}
for step in range(8000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    #xs.append(info.get("obs_state").get("agent")[0])
    #x_dots.append(info.get("obs_state").get("agent")[1])
    #xis.append(info.get("obs_state").get("agent")[2])
    #xi_dots.append(info.get("obs_state").get("agent")[3])
    total_times.append(info.get("total_time"))

    rewards.append(reward)
    actions.append(action)

    env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print(
            f"Goal reached! reward={reward}, "
            f"at time={info.get('total_time')}, "
            f"x_pos = {info.get('obs_state')[0]},"
            # f"x_pos = {info.get('obs_state').get('agent')[0]},"
            f"average_speed = {info.get('average_speed')}"
        )

        """
        # only for logging hi rez values
        actions.extend(info.get("action_deque"))
        total_times.extend(info.get("total_time_deque"))
        xs.extend(info.get("x_deque"))
        x_dots.extend(info.get("x_dot_deque"))
        """

        print("x is ")
        printer(states.get("x"))
        print("x_dot is ")
        printer(states.get("x_dot"))
        print("xi is ")
        printer(states.get("xi"))
        print("xi_dot is ")
        printer(states.get("xi_dot"))
        print(actions)

        break

"""
# only for logging hi rez values
data_to_point_dict = {"actions": actions,
                      "total_times": total_times,
                      "positions": xs,
                      "speeds": x_dots
                      }

data_frame = pd.DataFrame(data_to_point_dict)
data_frame.to_csv("data_max_speed.csv")
"""

env.close()


