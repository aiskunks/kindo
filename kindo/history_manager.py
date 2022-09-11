import json
from typing import List

import matplotlib.pyplot as plt

from kindo.paths import get_trained_model_paths
from kindo.utils import chunks


class HistoryManager:
    def __init__(self, env_dir_name):
        self.env_name = env_dir_name
        self.model_paths = get_trained_model_paths(env_dir_name)
        self.model_names = [model_path.split("/")[2] for model_path in self.model_paths]
        self.model_histories = {}

        for model_path, model_name in zip(self.model_paths, self.model_names):
            with open(f"{model_path}/history.json", "r") as f:
                model_history = json.load(f)

            self.model_histories[model_name] = model_history

    def get_history_overall_stats(self, stat_key):
        return [self.model_histories[model_name][stat_key] for model_name in self.model_names]

    @staticmethod
    def _plot_history_bars(
        title: str,
        values: List,
        y_label: str,
        x_label: str,
        x_ticks: List[str],
        rotate_x_ticks: bool,
        title_font_size: int,
        fig=None,
        ax=None,
    ):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ind = list(range(len(values)))
        ax.bar(ind, values)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_title(title, fontsize=title_font_size)
        ax.set_xticks(ind)
        rotation = 45 if rotate_x_ticks else 0
        ax.set_xticklabels(x_ticks, fontsize=8, rotation=rotation)

    def plot_mean_rewards(self, fig=None, ax=None, title_font_size=14, rotate_x_ticks=False):
        self._plot_history_bars(
            title=f"Mean 100 rewards over last 100 episodes for {self.env_name}",
            values=self.get_history_overall_stats("mean_100_episodes_reward"),
            y_label="Mean Rewards",
            x_label="Models",
            x_ticks=self.model_names,
            fig=fig,
            ax=ax,
            title_font_size=title_font_size,
            rotate_x_ticks=rotate_x_ticks,
        )

    def plot_mean_regrets(self, fig=None, ax=None, title_font_size=14, rotate_x_ticks=False):
        mean_regrets = self.get_history_overall_stats("mean_100_episodes_regret")
        if len(mean_regrets) == 0:
            print(
                "Regrets were not calculated for this environment. To calculate regrets while "
                "training it is required to proved `maximum_episode_reward` when call "
                "`train_a_model` or `train_a_couple_of_models functions"
            )
            return

        self._plot_history_bars(
            title=f"Mean 100 regrets over last 100 episodes for {self.env_name}",
            values=mean_regrets,
            y_label="Mean Regrets",
            x_label="Models",
            x_ticks=self.model_names,
            fig=fig,
            ax=ax,
            title_font_size=title_font_size,
            rotate_x_ticks=rotate_x_ticks,
        )

    def plot_time_spent_on_training(
        self, fig=None, ax=None, title_font_size=14, rotate_x_ticks=False
    ):
        self._plot_history_bars(
            title=f"Time models spent on training {self.env_name}",
            values=self.get_history_overall_stats("time_spent_on_training"),
            y_label="Time in seconds",
            x_label="Models",
            x_ticks=self.model_names,
            title_font_size=title_font_size,
            rotate_x_ticks=rotate_x_ticks,
            fig=fig,
            ax=ax,
        )

    def plot_episode_rewards(self, fig=None, ax=None, mean_step=30):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        episode_rewards = [
            self.model_histories[model_name]["rewards_per_episode"]
            for model_name in self.model_names
        ]

        mean_rewards = []
        for ep_rew in episode_rewards:
            mean_rewards.append([sum(chunk) / len(chunk) for chunk in chunks(ep_rew, mean_step)])

        maximum_number_of_episode_means = max(len(ep_rews) for ep_rews in mean_rewards)
        maximum_number_of_episodes = max(len(ep_rews) for ep_rews in episode_rewards)

        timesteps = list(
            range(
                0,
                max(len(ep_rews) for ep_rews in episode_rewards),
                int(maximum_number_of_episodes / maximum_number_of_episode_means),
            )
        )

        for i, ep_rew in enumerate(mean_rewards):
            ax.plot(timesteps[: len(ep_rew)], ep_rew, label=self.model_names[i])

        ax.set_title("Rewards per episode", fontsize=14)
        ax.set_xlabel("Episodes", fontsize=10)
        ax.set_ylabel("Reward", fontsize=10)
        ax.legend()

    def plot_episode_regrets(self, fig=None, ax=None, mean_step=30):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        episode_regrets = [
            self.model_histories[model_name]["regret_per_episode"]
            for model_name in self.model_names
        ]

        if len(episode_regrets) < 0:
            print(
                "Regrets were not calculated for this environment. To calculate regrets while "
                "training it is required to proved `maximum_episode_reward` when call "
                "`train_a_model` or `train_a_couple_of_models functions"
            )
            return

        mean_regrets = []
        for ep_rew in episode_regrets:
            mean_regrets.append([sum(chunk) / len(chunk) for chunk in chunks(ep_rew, mean_step)])

        maximum_number_of_episode_means = max(len(ep_rews) for ep_rews in mean_regrets)
        maximum_number_of_episodes = max(len(ep_rews) for ep_rews in episode_regrets)

        timesteps = list(
            range(
                0,
                max(len(ep_rews) for ep_rews in episode_regrets),
                int(maximum_number_of_episodes / maximum_number_of_episode_means),
            )
        )

        for i, ep_rew in enumerate(mean_regrets):
            ax.plot(timesteps[: len(ep_rew)], ep_rew, label=self.model_names[i])

        ax.set_title("Regrets per episode", fontsize=14)
        ax.set_xlabel("Episodes", fontsize=10)
        ax.set_ylabel("Regret", fontsize=10)
        ax.legend()
