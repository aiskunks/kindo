import logging
from abc import ABCMeta
from pathlib import Path
from typing import Optional, Type, Union

from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from kindo.callbacks import HistorySavingCallback, StopTrainingWhenMean100EpReward
from kindo.paths import abs_path, save_path

logger = logging.getLogger(__name__)


def initialize_stable_baselines_model(model_class: ABCMeta, train_env: Env) -> BaseAlgorithm:
    return model_class(policy="MlpPolicy", env=train_env)


def train_baselines_model(
    model: Union[BaseAlgorithm, Type[BaseAlgorithm]],
    env: Env,
    model_name: Optional[str],
    total_timesteps: int = 100000,
    log_interval: int = 10,
    maximum_episode_reward=None,
    stop_training_threshold=195,
):
    model_dir = f"{save_path}/{env.__class__.__name__}/{model_name}"

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    stop_callback = StopTrainingWhenMean100EpReward(
        reward_threshold=stop_training_threshold, timestep_activation_threshold=5000
    )
    history_saving_callback = HistorySavingCallback(
        total_timesteps=total_timesteps,
        history_save_dir=model_dir,
        stop_callback=stop_callback,
        maximum_episode_reward=maximum_episode_reward,
    )

    if isinstance(model, ABCMeta):
        model = initialize_stable_baselines_model(model_class=model, train_env=env)

    model = model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        callback=history_saving_callback,
    )
    model.save(abs_path(f"{model_dir}/model"))
    model.env.close()
    logger.info(f"Model {model.__class__.__name__} is trained and saved to {model_dir}")
