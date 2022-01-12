import gym


def create_env(config):
    env = gym.make(config.env_name)
    return env

def scaler(state):
    return state