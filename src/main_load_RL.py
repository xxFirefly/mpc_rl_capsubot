import os

import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import pandas as pd

from src.capsubot_env.capsubot_env import CapsubotEnv
from src.capsubot_env.version_for_PWM_func_optimization.capsubot_env_max_speed import CapsubotEnvMk2

from src.capsubot_env.capsubot_env_to_point import (
    CapsubotEnvToPoint,
    CapsubotEnvToPoint2,
)

# insert into model_path the path to the model *.zip
# it can't be hardcoded because of using datetime module
models_dir: str = os.path.join("..", "RL_WIP", "RL_data_store", "models")
model_path: str = os.path.join(
    models_dir,
    "CapsubotEnvToPoint2",
    "PPO_envs-1_LR-0002_steps-8192_epochs-10_MultiInputPolicy_13-04-2023-22",
    "3600000",
)


def printer(array: list) -> None:
    array = np.array(array)
    print(f"min_value = {np.amin(array)}, max_value = {np.amax(array)}")
    print("-----------------------------------------------------------")


model = PPO.load(model_path)

env = CapsubotEnvToPoint2(is_render=True,
                          rendering_fps=250,
                          model=0,
                          goal_point=0.5,
                          is_inference=True,
                          )
obs = env.reset()
rewards = []
xs = []
x_dots = []
xis = []
xi_dots = []
total_times = []
actions = []
# Ts = []
# taus = []
states = {"x": xs, "x_dot": x_dots, "xi": xis, "xi_dot": xi_dots}
for step in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # xs.append(info.get("obs_state").get("agent")[0])
    # x_dots.append(info.get("obs_state").get("agent")[1])
    # xis.append(info.get("obs_state").get("agent")[2])
    # xi_dots.append(info.get("obs_state").get("agent")[3])
    # total_times.append(info.get("total_time"))

    # rewards.append(reward)
    # actions.append(action)

    env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print(
            f"Goal reached! reward={reward}, "
            f"at time={info.get('total_time')}, "
            f"x_pos = {info.get('obs_state')},"
            # f"x_pos = {info.get('obs_state').get('agent')[0]},"
            f"average_speed = {info.get('average_speed')}"
        )

        try:
            print(f"{info.get('goal_point')}")
        except:
            print("It's max speed version. Here is no goal point")

        # only for logging hi rez values
        actions.extend(info.get("action_deque"))
        total_times.extend(info.get("total_time_deque"))
        xs.extend(info.get("x_deque"))
        x_dots.extend(info.get("x_dot_deque"))
        # Ts.extend(info.get("T_deque"))
        # taus.extend(info.get("tau_deque"))

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
        """

        break


# only for logging hi rez values
RL_controled_CapsubotEnv_data = {
    "actions": actions,
    "total_times": total_times,
    "positions": xs,
    "speeds": x_dots,
    # "Ts": Ts,
    # "tau": taus,
}

file_name: str = "3600k_LR-0002_steps-8192_epochs-10_MultiInput_13-04-2023-22.csv"
path_to_file = os.path.join("..", "RL_WIP", "RL_data_store", "csv_infos", "CapsubotToPoint2", file_name)

data_frame = pd.DataFrame(RL_controled_CapsubotEnv_data)
data_frame.to_csv(path_to_file)


env.close()
