import json
import time
from typing import Optional, Union

from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.environments.tf_py_environment import TFPyEnvironment


class BaseKindoRLCallback(BaseCallback):
    def init_callback(
        self,
        model: Union[BaseAlgorithm, TFAgent],
        train_env: Optional[Union[Env, TFPyEnvironment]] = None,
    ) -> None:
        if isinstance(model, BaseAlgorithm):
            super().init_callback(model)
            self.training_framework = "baselines"
        elif isinstance(model, TFAgent):
            self.training_env = train_env
            self.training_framework = "tf_agents"
        else:
            raise ValueError(
                f"{self.__class__.__name__} does not support "
                f"model of type `{model.__class__.__name__}`"
            )

    def on_step(self) -> bool:
        if isinstance(self.model, BaseAlgorithm):
            return super().on_step()
        else:
            self.n_calls += 1
            self.num_timesteps = self.n_calls

            return self._on_step()

    def on_steps(self, num_steps) -> bool:
        self.n_calls += num_steps
        self.num_timesteps = self.n_calls
        return self._on_step()


class HistorySavingCallback(BaseKindoRLCallback, EventCallback):
    """
    A callback enabling to save training history such as epsilon and reward per
    each episode. After model is trained, the callback will save historical data
    in history directory. This callback is an extension to `EvalCallback` thus
    all the evaluation logic is remained untouched
    """

    def __init__(
        self,
        total_timesteps: int,
        history_save_dir: str,
        stop_callback=None,
        maximum_episode_reward: int = None,
        verbose=0,
    ):
        """
        :param total_timesteps: Number of total timesteps to tran a model
        """
        self.history_save_dir = history_save_dir
        self.maximum_episode_reward = maximum_episode_reward
        self.total_timesteps = total_timesteps
        self.episode_regrets = []
        self._start_time = None
        # Overall stats:
        self._current_timestep = 0
        self.number_of_episodes_spent = 0
        self.mean_100_episodes_reward = 0
        self.mean_100_episodes_regret = None
        self.time_spent_on_training = None
        # For models trained with VecEnv
        self._curr_ep_rewards = None
        self._episode_rewards = None
        self.need_to_calculate_rewards = False  # True when model is trained with VecEnv
        super().__init__(callback=stop_callback, verbose=verbose)

    @property
    def episode_rewards(self):
        if self.need_to_calculate_rewards:
            return self._episode_rewards
        else:
            return self.locals["episode_rewards"]

    @property
    def history_file_path(self):
        return f"{self.history_save_dir}/history.json"

    def calculate_overall_stats(self):
        self.time_spent_on_training = time.time() - self._start_time
        self.number_of_episodes_spent = len(self.episode_rewards)
        self.mean_100_episodes_reward = sum(self.episode_rewards[-100:]) / len(
            self.episode_rewards[-100:]
        )

        if self.maximum_episode_reward:
            self.mean_100_episodes_regret = sum(self.episode_regrets[-100:]) / len(
                self.episode_regrets[-100:]
            )

    def calculate_regret(self):
        if self.maximum_episode_reward:
            self.episode_regrets = [
                self.maximum_episode_reward - episode_reward
                for episode_reward in self.episode_rewards
            ]

    def get_all_history(self):
        return {
            "rewards_per_episode": self.episode_rewards,
            "regret_per_episode": self.episode_regrets,
            "mean_100_episodes_reward": self.mean_100_episodes_reward,
            "mean_100_episodes_regret": self.mean_100_episodes_regret,
            "time_spent_on_training": self.time_spent_on_training,
            "number_of_episodes": self.number_of_episodes_spent,
        }

    def save_history(self):
        with open(self.history_file_path, "w") as f:
            json.dump(self.get_all_history(), f)

    def upgrade_stats_and_save(self):
        self.calculate_regret()
        self.calculate_overall_stats()
        self.save_history()

    def _on_training_start(self):
        if self.training_framework == "baselines":
            if isinstance(self.model.env, DummyVecEnv):
                self.need_to_calculate_rewards = True
                self._curr_ep_rewards = []
                self._episode_rewards = []

    def _on_step(self) -> bool:
        continue_training = self._on_event()

        curr_step = self.n_calls if hasattr(self, "n_calls") else self.num_timesteps

        is_last_timestep = curr_step >= self.total_timesteps

        if self.training_framework == "baselines":
            if self.need_to_calculate_rewards:
                try:
                    reward = self.locals["reward"][0]
                    done = self.locals["done"][0]
                except KeyError:
                    reward = self.locals["rewards"][0]
                    done = self.locals["dones"][0]

                self._curr_ep_rewards.append(reward)

                if done:
                    self._episode_rewards.append(sum(self._curr_ep_rewards))
                    self._curr_ep_rewards = []

                if curr_step < 9000:
                    # A weird behaviour of the framework here:
                    # when it is a VecEnv it stops earlier than it has to stop.
                    # It stops earlier from 96% of total timesteps to 99.85%.
                    # Numbers were calculated empirically.
                    is_last_timestep = curr_step > 0.96 * self.total_timesteps
                elif 9000 < curr_step < 200000:
                    is_last_timestep = curr_step > 0.994 * self.total_timesteps
                else:
                    is_last_timestep = curr_step > 0.9985 * self.total_timesteps

        if self._start_time is None:
            self._start_time = time.time()

        if is_last_timestep or not continue_training:
            self.upgrade_stats_and_save()

        return continue_training


class StopTrainingWhenMean100EpReward(BaseKindoRLCallback):
    def __init__(
        self,
        reward_threshold: float,
        timestep_activation_threshold: int = 5000,
        verbose: int = 1,
    ):
        """
        :param timestep_activation_threshold: A number of timesteps
        after which the callback is executed. You may need to use it
        when the reward threshold is negative and the algorithm starts
        with a number of rewards of zero. In this case  The algorithm
        will always start training with a greater number of mean rewards
        than a reward threshold and will be terminated after the first
        episode.
        """
        self.timestep_activation_threshold = timestep_activation_threshold
        self.reward_threshold = reward_threshold
        super().__init__(verbose)

    def _on_step(self) -> bool:
        continue_training = True
        if hasattr(self, "n_calls"):
            num_timesteps = self.n_calls
        else:
            num_timesteps = self.num_timesteps

        if num_timesteps > self.timestep_activation_threshold:
            if num_timesteps % 1000 == 0:
                if self.parent is not None:
                    last_100_rewards = self.parent.episode_rewards[-100:]
                else:
                    last_100_rewards = self.locals["episode_rewards"][-100:]
                mean_100_ep_reward = sum(last_100_rewards) / len(last_100_rewards)
                continue_training = mean_100_ep_reward <= self.reward_threshold

        return continue_training
