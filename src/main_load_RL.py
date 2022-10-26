import os

from stable_baselines3 import PPO

from src.capsubot_env.capsubot_env import CapsubotEnv, CapsubotEnvToPoint

# insert into model_path the path to the model *.zip
# it can't be hardcoded because of using datetime module
models_dir: str = os.path.join("..", "RL_WIP", "RL_data_store", "models")
model_path: str = os.path.join(
    models_dir,
    "to_point",
    "TO_POINT_PPO-n_envs_4_LR_0003_Nsteps_4096_Nepochs_1026_10_2022-02",
    "1000000",
)

model = PPO.load(model_path)

env = CapsubotEnvToPoint()
obs = env.reset()
n_steps = int(5.0 / env.dt)
rewards = []
states = [obs[0]]
actions = [0]
for step in range(2000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    rewards.append(reward)
    states.append(obs[0])
    actions.append(action)
    env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print(
            f"Goal reached! reward={reward}, at time={env.total_time}, x_pos = {info.get('current_pos')}"
        )
        break
env.close()
