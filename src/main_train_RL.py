import cProfile
import datetime
import os
import pstats

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.capsubot_env.capsubot_env import CapsubotEnv
from src.capsubot_env.capsubot_env_to_point import CapsubotEnvToPoint, CapsubotEnvToPoint2
from src.capsubot_env.custom_logger import SummaryWriterCallback

# HYPERPARAMS
# TODO: implement LR schedule
# TODO: to play with hyperparams more
N_ENVS: int = 1
LEARNING_RATE: float = float(3.2e-4)
TIMESTEPS: float = 1e5
TRAINING_REPS: int = 70
N_STEPS = 4096
# BATCH_SIZE =
N_EPOCHS = 10
# GAMMA =
# GAE_LAMBDA =

# POLICIES
# POLICY: str = "MlpPolicy"
POLICY: str = "MultiInputPolicy"


class AgentTrainer:
    def __init__(
        self,
        class_name: str,
        rl_model_name: str = "PPO",
        is_multithreading: bool = False,
    ):
        self.class_name: str = class_name
        self.rl_model_name: str = rl_model_name
        self.is_multithreading: bool = is_multithreading
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

        for i in range(1, TRAINING_REPS):
            # pr.enable()
            self.model.learn(
                total_timesteps=int(TIMESTEPS),
                reset_num_timesteps=False,
                tb_log_name=self.tensorboard_log_name,
                callback=SummaryWriterCallback(),
            )
            # pr.disable()
            self.model.save(os.path.join(self.models_dir, f"{int(TIMESTEPS * i)}"))
            # ps = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
            # ps.print_stats(50)

    def _create_dirs_paths(self):
        additional_info_str: str = f"_envs-{N_ENVS}_LR-{str(LEARNING_RATE)[2:]}_steps-{N_STEPS}_epochs-{N_EPOCHS}_{POLICY}_" + datetime.datetime.now().strftime(
            "%d-%m-%Y-%H"
        )

        self.models_dir: str = os.path.join(
            "..", "RL_WIP", "RL_data_store", "models", self.sub_dir, self.rl_model_name
        ) + additional_info_str

        self.logdir: str = os.path.join(
            "..", "RL_WIP", "RL_data_store", "logs", self.sub_dir, self.rl_model_name
        ) + additional_info_str

    def _create_dirs(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def _pick_version(self):
        if self.is_multithreading:
            # TODO: implement multithreading version
            pass
        else:
            if self.class_name == "CapsubotEnv":
                env = CapsubotEnv()
                self.sub_dir: str = env.__class__.__name__

                err_msg: str = "You can't use MultiInputPolicy with CapsubotEnv()"
                assert POLICY != "MultiInputPolicy", err_msg

                self.left_term_point: float = env.left_termination_point
                self.right_term_point: float = env.right_termination_point
            elif self.class_name == "CapsubotEnvToPoint":
                env = CapsubotEnvToPoint()
                self.sub_dir: str = env.__class__.__name__

                err_msg: str = "You must use MultiInputPolicy with CapsubotEnvToPoint()"
                assert POLICY == "MultiInputPolicy", err_msg

                # right_term_point is random for this env
                self.left_term_point: float = env.left_termination_point
            else:
                env = CapsubotEnvToPoint2()
                self.sub_dir: str = env.__class__.__name__

                err_msg: str = "You must use MultiInputPolicy with CapsubotEnvToPoint()"
                assert POLICY == "MultiInputPolicy", err_msg

                # right_term_point is random for this env
                self.left_term_point: float = env.left_termination_point

        return make_vec_env(lambda: env, n_envs=N_ENVS)

    def _pick_tb_log_name(self) -> str:
        if self.class_name == "CapsubotEnv":
            if self.left_term_point < 0.0:  # -0.XXX[3:] -> "-XXX"
                left_term_point: str = "-" + str(self.left_term_point)[3:]
            else:
                left_term_point: str = str(self.left_term_point)[2:]
            right_term_point: str = str(self.right_term_point)[2:]

            tensorboard_log_name: str = (
                f"PPO_left_tp({left_term_point})_right_tp({right_term_point})"
            )
        else:
            if self.left_term_point < 0.0:
                left_term_point: str = "-" + str(self.left_term_point)[3:]
            else:
                left_term_point: str = str(self.left_term_point)[2:]

            tensorboard_log_name: str = f"PPO_left_tp({left_term_point})"

        return tensorboard_log_name

    def _pick_model(self):
        model = PPO(
            POLICY,
            self.env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            tensorboard_log=self.logdir,
            n_steps=N_STEPS,
            n_epochs=N_EPOCHS,
        )
        return model


if __name__ == "__main__":
    AgentTrainer(class_name="CapsubotEnvToPoint2").train()
    AgentTrainer(class_name="CapsubotEnvToPoint").train()
