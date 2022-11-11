from typing import List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


def dict_to_list(item):
    item = item.get("obs_state")
    if isinstance(item, dict):
        return item.get("agent")
    return item


class SummaryWriterCallback(BaseCallback):
    def _on_training_start(self) -> None:
        self._log_freq = 100  # log every 100 calls

        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            list_of_av_speeds = [
                item.get("average_speed") for item in self.locals.get("infos")
            ]
            list_of_x_pos = [dict_to_list(item)[0] for item in self.locals.get("infos")]
            list_of_x_velocities = [
                dict_to_list(item)[1] for item in self.locals.get("infos")
            ]
            list_of_t_times = [
                item.get("total_time") for item in self.locals.get("infos")
            ]
            list_of_goal_points = [
                item.get("goal_point") for item in self.locals.get("infos")
            ]

            average_speed = np.array(sum(list_of_av_speeds) / len(list_of_av_speeds))
            self.tb_formatter.writer.add_scalar(
                "average_speed_train/average_speed_mean",
                average_speed,
                self.num_timesteps,
            )

            self._tb_writer(
                list_of_av_speeds, "average_speed_train", "average_speed_env"
            )
            self._tb_writer(list_of_x_pos, "positions", "x_pos_of")
            self._tb_writer(list_of_x_velocities, "velocities", "x_velocity")
            self._tb_writer(list_of_t_times, "rollout", "total_time")
            if all(list_of_goal_points):
                self._tb_writer(list_of_goal_points, "positions", "goal_point")

            self.tb_formatter.writer.flush()
            return True

    def _tb_writer(
        self, list_of_values: List[float], section_name: str, plot_name: str
    ):
        for i in range(len(list_of_values)):
            self.tb_formatter.writer.add_scalar(
                f"{section_name}/{plot_name}_{i + 1}",
                list_of_values[i],
                self.num_timesteps,
            )
