from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies import tf_policy


def compute_mean_reward(
    environment: TFPyEnvironment, policy: tf_policy.Base, num_episodes=10
) -> float:
    """
    Evaluate mean reward over `num_episodes`
    Implementation is taken from Tensorflow official documentation tutorial:
    https://www.tensorflow.org/agents/tutorials/6_reinforce_tutorial#metrics_and_evaluation
    """
    total_reward = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_reward = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_reward += time_step.reward

        total_reward += episode_reward

    avg_rewards = total_reward / num_episodes
    return avg_rewards.numpy()[0]
