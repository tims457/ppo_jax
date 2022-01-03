import gym


def create_env(game):
    env = gym.make(game)
    return env
