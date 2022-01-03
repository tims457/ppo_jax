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
        out = nn.Dense(self.num_outputs * 2, name='logits', dtype=dtype)(x)
        mu, sigma = out[..., :self.num_outputs], nn.softplus(
            out[..., self.num_outputs:])
        value = nn.Dense(1, name='value', dtype=dtype)(x)
        return mu, sigma, value
