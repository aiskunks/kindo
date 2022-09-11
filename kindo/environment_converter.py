import gym
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.environments.tf_py_environment import TFPyEnvironment


def gym_to_tf(env: gym.Env) -> TFPyEnvironment:
    """
    Convert gym environment to tf_agents compilable environment
    """
    py_env = wrap_env(env)
    return TFPyEnvironment(py_env)
