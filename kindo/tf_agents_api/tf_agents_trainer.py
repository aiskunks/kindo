import typing
from abc import ABCMeta
from pathlib import Path

import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tf_agents import agents
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.agents.dqn.examples.v2.train_eval import create_feedforward_network
from tf_agents.agents.sac.tanh_normal_projection_network import TanhNormalProjectionNetwork
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import kindo
import kindo.paths
from kindo import callbacks, environment_converter
from kindo.tf_agents_api import utils


class WrongModelError(Exception):
    pass


def train_off_policy_tf_agent(
    model: TFAgent,
    train_env: TFPyEnvironment,
    total_timesteps: int,
    callback: callbacks.BaseKindoRLCallback = None,
):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        model.collect_data_spec, batch_size=train_env.batch_size, max_length=100000
    )
    collect_policy = model.collect_policy

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=64, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    if callback is not None:
        callback.init_callback(model=model, train_env=train_env)

    # Metrics over all the episodes
    locals_ = {"episode_rewards": [], "episode_losses": [], "episode_lengths": []}
    # Metrics for current episode
    curr_episode_losses, curr_episode_rewards, curr_episode_length = [], [], 0

    utils.step(environment=train_env, policy=collect_policy, replay_buffer=replay_buffer)

    if callback is not None:
        callback.on_training_start(locals_=locals_, globals_={})

    for _ in range(total_timesteps):
        reward, done = utils.step(
            environment=train_env, policy=collect_policy, replay_buffer=replay_buffer
        )
        experience, unused_info = next(iterator)
        train_loss = model.train(experience).loss.numpy()

        curr_episode_losses.append(train_loss)
        curr_episode_rewards.append(reward)
        curr_episode_length += 1

        if done:
            locals_["episode_rewards"].append(sum(curr_episode_rewards))
            locals_["episode_losses"].append(sum(curr_episode_losses))
            locals_["episode_lengths"].append(curr_episode_length)

            curr_episode_rewards = []
            curr_episode_losses = []
            curr_episode_length = 0

        if callback is not None:
            callback.update_locals(locals_)
            continue_training = callback.on_step()

            if not continue_training:
                break

    if callback is not None:
        callback.on_training_end()


def train_on_policy_tf_agent(
    model: TFAgent,
    train_env: TFPyEnvironment,
    total_timesteps: int,
    callback: callbacks.BaseKindoRLCallback = None,
):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        model.collect_data_spec, batch_size=train_env.batch_size, max_length=100000
    )

    if callback is not None:
        callback.init_callback(model, train_env=train_env)

    collect_policy = model.collect_policy
    locals_ = {"episode_rewards": [], "episode_losses": [], "episode_lengths": []}
    passed_timesteps = 0

    if callback is not None:
        callback.on_training_start(locals_, {})

    while passed_timesteps < total_timesteps:
        episode_reward, episode_length = utils.step_episode(
            train_env, collect_policy, replay_buffer
        )
        passed_timesteps += episode_length
        locals_["episode_rewards"].append(episode_reward)
        locals_["episode_lengths"].append(episode_length)

        experience = replay_buffer.gather_all()
        train_loss = model.train(experience).loss.numpy()
        locals_["episode_losses"].append(train_loss)
        replay_buffer.clear()

        if callback is not None:
            callback.update_locals(locals_)
            continue_training = callback.on_steps(num_steps=episode_length)
            if not continue_training:
                break

    if callback is not None:
        callback.on_training_end()


def initialize_tf_agent(model_class: ABCMeta, train_env: TFPyEnvironment) -> TFAgent:
    optimizer = Adam(learning_rate=1e-3)

    if model_class in [agents.PPOAgent]:
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=(200, 100),
            activation_fn=tf.keras.activations.tanh,
        )
        value_net = value_network.ValueNetwork(
            train_env.observation_spec(),
            fc_layer_params=(200, 100),
            activation_fn=tf.keras.activations.tanh,
        )
        model = model_class(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            actor_net=actor_net,
            value_net=value_net,
            optimizer=optimizer,
        )
    elif model_class in [agents.DqnAgent]:
        action_spec = train_env.action_spec()
        num_actions = action_spec.maximum - action_spec.minimum + 1
        q_network = create_feedforward_network(fc_layer_units=(100,), num_actions=num_actions)
        model = model_class(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            q_network=q_network,
            optimizer=optimizer,
        )
    elif model_class in [agents.ReinforceAgent]:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.time_step_spec().observation, train_env.action_spec(), fc_layer_params=(100,)
        )
        model = model_class(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            actor_network=actor_net,
            optimizer=optimizer,
        )
    elif model_class in [agents.SacAgent]:
        time_step_spec = train_env.time_step_spec()
        observation_spec = time_step_spec.observation
        action_spec = train_env.action_spec()
        critic_joint_fc_layers = (256, 256)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=(256, 256),
            continuous_projection_net=TanhNormalProjectionNetwork,
        )
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            joint_fc_layer_params=critic_joint_fc_layers,
            kernel_initializer="glorot_uniform",
            last_kernel_initializer="glorot_uniform",
        )
        model = agents.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(3e-4),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(3e-4),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(3e-4),
        )
    else:
        raise ValueError(f"Class of class `{model_class.__name__}` is not supported")
    model.initialize()
    return model


def train_tf_agent(
    model: typing.Union[TFAgent, typing.Type[TFAgent]],
    env: gym.Env,
    total_timesteps: int,
    model_name: typing.Optional[str] = None,
    maximum_episode_reward: int = 200,
    stop_training_threshold: int = 195,
):
    train_env = environment_converter.gym_to_tf(env)
    environment_name = env.__class__.__name__
    model_dir = f"{kindo.paths.save_path}/{environment_name}/{model_name}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    stop_training_callback = callbacks.StopTrainingWhenMean100EpReward(
        reward_threshold=stop_training_threshold
    )
    history_saving_callback = callbacks.HistorySavingCallback(
        total_timesteps=total_timesteps,
        history_save_dir=model_dir,
        maximum_episode_reward=maximum_episode_reward,
        stop_callback=stop_training_callback,
    )

    if isinstance(model, ABCMeta):
        model = initialize_tf_agent(model_class=model, train_env=train_env)

    if model.__class__ in [
        agents.DqnAgent,
        DdqnAgent,
        agents.DdpgAgent,
        agents.SacAgent,
    ]:
        train_off_policy_tf_agent(model, train_env, total_timesteps, history_saving_callback)
    elif model.__class__ in [agents.PPOAgent, agents.ReinforceAgent, agents.Td3Agent]:
        train_on_policy_tf_agent(model, train_env, total_timesteps, history_saving_callback)
    else:
        raise WrongModelError(
            f"Model of class `{model.__class__.__name__}` is not supported by Kindo API"
        )

    collect_policy = model.collect_policy
    saver = PolicySaver(collect_policy, batch_size=None)
    saver.save(f"{model_dir}/model")
