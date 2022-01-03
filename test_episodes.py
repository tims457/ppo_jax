"""Test policy by playing a full game."""

import itertools
from typing import Any, Callable
import flax
import distrax
import numpy as np
import jax

import env_utils
import agent
import models


def policy_test(n_episodes: int, apply_fn: Callable[..., Any],
                params: flax.core.frozen_dict.FrozenDict, game: str):
    """Perform a test of the policy in Atari environment.

  Args:
    n_episodes: number of full Atari episodes to test on
    apply_fn: the actor-critic apply function
    params: actor-critic model parameters, they define the policy being tested
    game: defines the Atari game to test on

  Returns:
    total_reward: obtained score
  """
    test_env = env_utils.create_env(game)
    key = jax.random.PRNGKey(1234)
    for _ in range(n_episodes):
        obs = test_env.reset()
        state = obs[None, ...]  # add batch dimension
        total_reward = 0.0
        for t in itertools.count():
            key, subkey = jax.random.split(key)
            mu, sigma, _ = agent.policy_action(apply_fn, params, state)
            dist = distrax.MultivariateNormalDiag(mu, sigma)
            action = dist.sample(seed=subkey)
            # probs = np.exp(np.array(log_probs, dtype=np.float32))
            # probabilities = probs[0] / probs[0].sum()
            # action = np.random.choice(probs.shape[1], p=probabilities)
            obs, reward, done, _ = test_env.step(action.tolist()[0])
            total_reward += reward
            next_state = obs[None, ...] if not done else None
            state = next_state
            if done:
                break
    return total_reward
