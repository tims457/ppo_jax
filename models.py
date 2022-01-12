import distrax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class ActorCritic(nn.Module):
    """Class defining the actor-critic model with shared input layers"""
    layers: Sequence[int]
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x.astype(dtype)
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer,
                         name=f'fc{i}',
                         kernel_init=nn.initializers.orthogonal(),
                         dtype=dtype)(x)
            x = nn.tanh(x)
        out = nn.Dense(self.num_outputs * 2, name='logits',
                       kernel_init=nn.initializers.orthogonal(), dtype=dtype)(x)
        mu, sigma = out[..., :self.num_outputs], nn.softplus(
            out[..., self.num_outputs:])

        value = nn.Dense(
            1, name='value', kernel_init=nn.initializers.orthogonal(), dtype=dtype)(x)
        return mu, sigma, value


class Actor(nn.Module):
    """Class defining the actor model"""
    layers: Sequence[int]
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x.astype(dtype)
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer,
                         name=f'fc{i}',
                         kernel_init=nn.initializers.orthogonal(),
                         dtype=dtype)(x)
            x = nn.tanh(x)
        out = nn.Dense(self.num_outputs * 2, name='logits',
                       kernel_init=nn.initializers.orthogonal(), dtype=dtype)(x)
        mu, sigma = out[..., :self.num_outputs], nn.softplus(
            out[..., self.num_outputs:])
        return mu, sigma


class Critic(nn.Module):
    """Class defining the critic model """
    layers: Sequence[int]
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32
        x.astype(dtype)
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer,
                         name=f'fc{i}',
                         kernel_init=nn.initializers.orthogonal(),
                         dtype=dtype)(x)
            x = nn.tanh(x)

        value = nn.Dense(
            1, name='value', kernel_init=nn.initializers.orthogonal(), dtype=dtype)(x)
        return value
