import cProfile
import datetime
import os
import pstats

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.capsubot_env.capsubot_env import CapsubotEnv
from src.capsubot_env.capsubot_env_to_point import CapsubotEnvToPoint, CapsubotEnvToPoint2
from src.capsubot_env.version_for_PWM_func_optimization.capsubot_env_max_speed import CapsubotEnvMk2
from src.capsubot_env.custom_logger import SummaryWriterCallback

# HYPERPARAMS
# TODO: implement LR schedule
# TODO: to play with hyperparams more
N_ENVS: int = 1
N_STEPS = 4096
# BATCH_SIZE =
N_EPOCHS = 10
# GAMMA =
# GAE_LAMBDA =

# POLICIES
# POLICY: str = "MlpPolicy"
# POLICY: str = "MultiInputPolicy"


class AgentTrainer:
    def __init__(
        self,
        class_name: CapsubotEnv | CapsubotEnvToPoint | CapsubotEnvToPoint2 | CapsubotEnvMk2,
        policy: str,
        sub_dir: str,
        rl_model_name: str = "PPO",
        is_multithreading: bool = False,
        lr: float = float(2.0e-4),
        training_reps: int = 30,
        timesteps: float = 1e5,
        batch_size: int = 64,
        n_steps: int = 4096,
    ):
        self.class_name = class_name
        self.sub_dir = sub_dir
        self.rl_model_name: str = rl_model_name
        self.is_multithreading: bool = is_multithreading
        self.learning_rate = lr
        self.training_reps = training_reps
        self.timesteps = timesteps
        self.policy = policy
        self.batch_size = batch_size
        self.n_steps = n_steps

        err_msg = "you forgot to choose policy"
        assert self.policy is not None, err_msg

        self.env = self._pick_version()
        self.tensorboard_log_name: str = self._pick_tb_log_name()
        self._create_dirs_paths()
        self._create_dirs()
        self.model = self._pick_model()

    def train(self):
        """
        uncomment lines if you want to profile the code
        """
        # pr = cProfile.Profile()

        self._create_log_file()

        for i in range(1, self.training_reps):
            # pr.enable()
            self.model.learn(
                total_timesteps=int(self.timesteps),
                reset_num_timesteps=False,
                tb_log_name=self.tensorboard_log_name,
                callback=SummaryWriterCallback(),
            )
            # pr.disable()
            self.model.save(os.path.join(self.models_dir, f"{int(self.timesteps * i)}"))
            # ps = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
            # ps.print_stats(50)

    def _create_dirs_paths(self):
        additional_info_str: str = f"_LR-{str(self.learning_rate)[2:]}_" + datetime.datetime.now().strftime(
            "%d-%m-%Y-%H"
        )

        self.models_dir: str = os.path.join(
            "..", "RL_WIP", "RL_data_store", "models", self.env_name, self.sub_dir, self.rl_model_name
        ) + additional_info_str

        self.logdir: str = os.path.join(
            "..", "RL_WIP", "RL_data_store", "logs", self.env_name, self.sub_dir, self.rl_model_name
        ) + additional_info_str

    def _create_dirs(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def _pick_version(self):
        env = self.class_name

        if self.is_multithreading:
            # TODO: implement multithreading version
            pass
        else:
            if isinstance(self.class_name, (CapsubotEnvToPoint, CapsubotEnvToPoint2)):
                self.env_name: str = env.__class__.__name__

                err_msg: str = "You must use MultiInputPolicy with CapsubotEnvToPoint()"
                assert self.policy == "MultiInputPolicy", err_msg

                # right_term_point is random for this env
                self.left_term_point: float = env.left_termination_point

            elif isinstance(self.class_name, (CapsubotEnv, CapsubotEnvMk2)):
                self.env_name: str = env.__class__.__name__

                err_msg: str = "You can't use MultiInputPolicy with CapsubotEnv()"
                assert self.policy != "MultiInputPolicy", err_msg

                self.left_term_point: float = env.left_termination_point
                self.right_term_point: float = env.right_termination_point

        return make_vec_env(lambda: env, n_envs=N_ENVS)

    def _pick_tb_log_name(self) -> str:
        if isinstance(self.class_name, (CapsubotEnvToPoint, CapsubotEnvToPoint2)):
            if self.left_term_point < 0.0:
                left_term_point: str = "-" + str(self.left_term_point)[3:]
            else:
                left_term_point: str = str(self.left_term_point)[2:]

            tensorboard_log_name: str = f"PPO_left_tp({left_term_point})"
        else:
            if self.left_term_point < 0.0:  # -0.XXX[3:] -> "-XXX"
                left_term_point: str = "-" + str(self.left_term_point)[3:]
            else:
                left_term_point: str = str(self.left_term_point)[2:]
            right_term_point: str = str(self.right_term_point)[2:]

            tensorboard_log_name: str = (
                f"PPO_left_tp({left_term_point})_right_tp({right_term_point})"
            )

        return tensorboard_log_name

    def _create_log_file(self):
        f = open(f"{os.path.join(self.logdir, 'log_file.txt')}", "w+")
        logs_info = self.class_name.model_logs

        f.write(f"batch_size={self.batch_size},"
                f" n_steps={self.n_steps},"
                f" n_envs={N_ENVS},"
                f" lr={self.learning_rate},"
                f" policy={self.policy},"
                f" mechanical model=({logs_info.get('mechanical_params')}),"
                f" environment params=({logs_info.get('env_params')})")
        f.close()

    def _pick_model(self):
        model = PPO(
            self.policy,
            self.env,
            verbose=1,
            learning_rate=self.learning_rate,
            tensorboard_log=self.logdir,
            n_steps=self.n_steps,
            n_epochs=N_EPOCHS,
            batch_size=self.batch_size,
        )
        return model


if __name__ == "__main__":
    # AgentTrainer(class_name="CapsubotEnvToPoint2").train()
    # AgentTrainer(class_name="CapsubotEnvToPoint").train()
    AgentTrainer(policy="MlpPolicy", class_name=CapsubotEnv(model=0), sub_dir="standart_reward_func", batch_size=64, n_steps=8192, lr=float(1.0e-4),
                 training_reps=40).train()
    AgentTrainer(policy="MlpPolicy", class_name=CapsubotEnv(model=0), sub_dir="standart_reward_func", batch_size=64, n_steps=8192,
                 lr=float(2.0e-4),
                 training_reps=40).train()
    AgentTrainer(policy="MlpPolicy", class_name=CapsubotEnv(model=0), sub_dir="standart_reward_func", batch_size=256, n_steps=8192, lr=float(1.0e-4),
                 training_reps=40).train()
    AgentTrainer(policy="MlpPolicy", class_name=CapsubotEnv(model=0), sub_dir="standart_reward_func", batch_size=64, n_steps=2048, lr=float(1.0e-4),
                 training_reps=40).train()
