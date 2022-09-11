from abc import ABCMeta
from typing import List, Optional, Type, Union

from gym import Env
from gym.wrappers import TimeLimit
from stable_baselines3.common.base_class import BaseAlgorithm
from tf_agents.agents.tf_agent import TFAgent

from kindo import utils
from kindo.stable_baselines_api.stable_baselines_trainer import train_baselines_model
from kindo.tf_agents_api.tf_agents_trainer import train_tf_agent


def train(
    model: Union[BaseAlgorithm, TFAgent, Type[BaseAlgorithm], Type[TFAgent]],
    env: Union[Env, TimeLimit],
    total_timesteps: int,
    stop_threshold: int,
    model_name: Optional[str] = None,
    maximum_episode_reward: Optional[int] = None,
):
    env = env.env if isinstance(env, TimeLimit) else env
    model_name = model_name or utils.compile_random_model_name(model)
    train_kwargs = {
        "model": model,
        "env": env,
        "total_timesteps": total_timesteps,
        "model_name": model_name,
        "maximum_episode_reward": maximum_episode_reward,
        "stop_training_threshold": stop_threshold,
    }
    # Check instances:
    if isinstance(model, BaseAlgorithm):
        train_baselines_model(**train_kwargs)
    elif isinstance(model, TFAgent):
        train_tf_agent(**train_kwargs)
    # Check classes:
    elif issubclass(model, BaseAlgorithm):
        train_baselines_model(**train_kwargs)
    elif issubclass(model, TFAgent):
        train_tf_agent(**train_kwargs)
    else:
        raise ValueError(f"Model of class `{model.__class__.__name__}` is not supported")


def train_multiple(
    models: List[Union[BaseAlgorithm, TFAgent, ABCMeta]],
    env: Env,
    total_timesteps: int,
    stop_threshold: int,
    model_names: Optional[List[str]] = None,
    maximum_episode_reward: int = None,
):
    if model_names is not None:
        assert len(models) == len(model_names), (
            "The length of the `model_names` list should be the " "same as `models` list"
        )
    else:
        model_names = [utils.compile_random_model_name(model) for model in models]

    for model, model_name in zip(models, model_names):
        train(
            model=model,
            env=env,
            total_timesteps=total_timesteps,
            model_name=model_name,
            maximum_episode_reward=maximum_episode_reward,
            stop_threshold=stop_threshold,
        )
