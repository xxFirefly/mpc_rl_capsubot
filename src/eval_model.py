import os

from stable_baselines3 import PPO

from src.capsubot_env.capsubot_env import CapsubotEnv
from src.capsubot_env.version_for_PWM_func_optimization.capsubot_env_max_speed import CapsubotEnvMk2
from src.capsubot_env.capsubot_env_to_point import CapsubotEnvToPoint, CapsubotEnvToPoint2


def load_and_eval_model(directory: str, env, env_render: bool = False):
    """
    loading and evaluating models in the specified directory
    :param env: test gym env
    :param env_render: whether to render the env or not (it takes render time)
    :param directory: specify the desired dir here
    :return: dict of good models (models that reached done point with positive reward), dict of bad models and info
    """
    good_models = {}
    fail_models = {}
    eval_info = {}
    for saved_model_name in range(1, 80):
        model_path: str = os.path.join(
            "..",
            "RL_WIP",
            "RL_data_store",
            "models",
            "CapsubotEnvToPoint2",
            f"{directory}",
            f"{int(saved_model_name * 1e5)}",
        )
        try:
            model = PPO.load(model_path)
        except:
            print("No such file. You have less models than you are trying to load")
            break

        good_models, fail_models, eval_info = eval_model(
            env=env,
            model=model,
            good_models=good_models,
            fail_models=fail_models,
            eval_info=eval_info,
            env_render=env_render,
            saved_model_name=saved_model_name,
        )
    return good_models, fail_models, eval_info


def eval_model(
    env,
    model,
    good_models: dict,
    fail_models: dict,
    eval_info: dict,
    env_render: bool,
    saved_model_name: int,
):
    """
    :param env: test gym env
    :param model: RL model
    :param saved_model_name: name of the model to eval
    :return: dict of good models (models that reached done point with positive reward), dict of bad models and info
    """
    obs = env.reset()
    rewards = []
    actions = [0]
    for _ in range(7000):  # for ToPoint you must use more timesteps
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if env_render:
            env.render()
        if done:
            if (
                reward > 0.0
            ):  # dict looks like {"model_name": time when done is reached}
                good_models[f"{int(saved_model_name * 1e5)}"] = info.get("total_time")
            else:
                fail_models[f"{int(saved_model_name * 1e5)}"] = info.get("total_time")
            break
        else:  # Such models that don't reach any done points for 3k timesteps are bad too
            fail_models[f"{int(saved_model_name * 1e5)}"] = info.get("total_time")
    env.close()
    eval_info.update(
        [
            (f"rewards_{int(saved_model_name * 1e5)}", rewards),
            (f"actions_{int(saved_model_name * 1e5)}", actions)
        ]
    )
    return good_models, fail_models, eval_info


def choose_best(models_dict: dict) -> None:
    """
    :returns best model's time and name in terms of average speed
    """
    try:
        key_with_min_value = min(models_dict, key=lambda k: models_dict[k])
        time = models_dict.get(key_with_min_value)
        print(f"Best model: {key_with_min_value}, with time: {time}")
    except:
        print("There is no good models in the run")


def main() -> None:
    env = CapsubotEnvToPoint2(model=0, goal_point=0.5, is_inference=True)
    good_models, fail_models, eval_info = load_and_eval_model(
        directory="PPO_envs-1_LR-0002_steps-8192_epochs-10_MultiInputPolicy_14-04-2023-07", env=env
    )
    print(f"good_models_dict{good_models}")
    print("__________________________________________")
    print(f"bad_models_dict{fail_models}")
    print("__________________________________________")
    choose_best(good_models)


if __name__ == "__main__":
    main()
