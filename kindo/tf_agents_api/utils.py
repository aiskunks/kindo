import typing

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.trajectories import trajectory


"""
It was made a decision not to use drivers and metrics provided by tf_agents as it looks
inconvenient to use them. They also dont fit to kindo common format.
"""


def step(
    environment: TFPyEnvironment, policy: tf_policy.TFPolicy, replay_buffer: ReplayBuffer
) -> typing.Tuple[float, bool]:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)
    return next_time_step.reward.numpy()[0], next_time_step.is_last()


def step_episode(
    environment: TFPyEnvironment, policy: tf_policy.TFPolicy, replay_buffer: ReplayBuffer
) -> typing.Tuple[int, int]:
    done = False
    environment.reset()

    curr_episode_rewards = []
    episode_reward = 0
    episode_length = 0

    while not done:
        reward, done = step(environment, policy, replay_buffer)
        curr_episode_rewards.append(reward)
        episode_length += 1

        if done:
            episode_reward = sum(curr_episode_rewards)

    return episode_reward, episode_length
