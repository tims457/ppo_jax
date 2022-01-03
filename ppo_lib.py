# PPO for continuous action space based on example code from Flax
# https://github.com/google/flax


import jax
import flax
import time
import optax
import distrax
import functools
import ml_collections

import jax.random
import numpy as np
import jax.numpy as jnp

from absl import logging
from flax import linen as nn
from numpy.random import seed
from datetime import datetime
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.core.scope import Collection

from typing import Any, Callable, Tuple, List
from flax.training.train_state import TrainState

import agent
import models
import env_utils
import test_episodes
# TODO allow discrete and continuous actions
# TODO fix checkpointing and tensorboard
# TODO update documentation
# TODO all input types
# TODO actor and critic models


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(
    rewards: np.ndarray,
    terminal_masks: np.ndarray,
    values: np.ndarray,
    discount: float,
    gae_param: float,
) -> jnp.array:
    """Use Generalized Advantage Estimation (GAE) to compute advantages.

    As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
    key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.

    Args:
        rewards: array shaped (actor_steps, num_agents), rewards from the environment
        terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
                        and ones for non-terminal states
        values: array shaped (actor_steps, num_agents), values estimated by critic
        discount: RL discount usually denoted with gamma
        gae_param: GAE parameter usually denoted with lambda

    Returns:
        advantages: calculated advantages shaped (actor_steps, num_agents)
    """
    assert rewards.shape[0] + 1 == values.shape[0], (
        "One more value needed; Eq. "
        "(12) in PPO paper requires "
        "V(s_{t+1}) for delta_t")
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal states.
        value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
        delta = rewards[t] + value_diff
        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + discount * gae_param * terminal_masks[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)


@functools.partial(jax.jit, static_argnums=(1,))
def loss_fn(
    params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    minibatch: Tuple,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
    key,
) -> float:
    """Evaluate the loss function.

    Compute loss as a sum of three components: the negative of the PPO clipped
    surrogate objective, the value function loss and the negative of the entropy
    bonus.

    Args:
        params: the parameters of the actor-critic model
        apply_fn: the actor-critic model's apply function
        minibatch: Tuple of five elements forming one experience batch:
                obs: shape (batch_size, 84, 84, 4)
                actions: shape (batch_size, 84, 84, 4)
                old_log_probs: shape (batch_size,)
                returns: shape (batch_size,)
                advantages: shape (batch_size,)
        clip_param: the PPO clipping parameter used to clamp ratios in loss function
        vf_coeff: weighs value function loss in total loss
        entropy_coeff: weighs entropy bonus in the total loss

    Returns:
        loss: the PPO loss, scalar quantity
    """
    obs, actions, old_log_probs, returns, advantages = minibatch

    mus, sigmas, values = agent.policy_action(apply_fn, params, obs)
    dist = distrax.MultivariateNormalDiag(mus, sigmas)
    _, log_probs = dist.sample_and_log_prob(seed=key)
    # probs = jnp.exp(log_probs)

    values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).

    value_loss = jnp.mean(jnp.square(returns - values), axis=0)

    # https://math.stackexchange.com/questions/2029707/entropy-of-the-multivariate-gaussian
    # entropy = jnp.sum(-probs * log_probs).mean()

    entropy = dist.entropy().mean()

    # log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    log_probs_act_taken = dist.log_prob(actions)
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    # Advantage normalization (following the OpenAI baselines).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(1.0 - clip_param, ratios,
                                              1.0 + clip_param)
    ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)

    return ppo_loss + vf_coeff * value_loss - entropy_coeff * entropy


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(
    state: TrainState,
    trajectories: Tuple,
    batch_size: int,
    *,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
    key,
) -> Tuple:
    """Compilable train step.

    Runs an entire epoch of training (i.e. the loop over minibatches within
    an epoch is included here for performance reasons).

    Args:
        state: the train state
        trajectories: Tuple of the following five elements forming the experience:
                    obs: shape (steps_per_agent*num_agents, obs_size)
                    actions: shape (steps_per_agent*num_agents, output_size)
                    old_log_probs: shape (steps_per_agent*num_agents, )
                    returns: shape (steps_per_agent*num_agents, )
                    advantages: (steps_per_agent*num_agents, )
        batch_size: the minibatch size, static argument
        clip_param: the PPO clipping parameter used to clamp ratios in loss function
        vf_coeff: weighs value function loss in total loss
        entropy_coeff: weighs entropy bonus in the total loss

    Returns:
        optimizer: new optimizer after the parameters update
        loss: loss summed over training steps
    """
    iterations = trajectories[0].shape[0] // batch_size
    trajectories = jax.tree_map(
        lambda x: x.reshape((iterations, batch_size) + x.shape[1:]),
        trajectories)
    loss = 0.0
    for batch in zip(*trajectories):
        key, subkey = jax.random.split(key)
        # loss_fn(state.params, state.apply_fn, batch, clip_param, vf_coeff,
        # entropy_coeff, subkey)
        grad_fn = jax.value_and_grad(loss_fn)
        l, grads = grad_fn(state.params, state.apply_fn, batch, clip_param,
                           vf_coeff, entropy_coeff, subkey)
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss


def get_experience(
    key,
    train_state: TrainState,
    simulators: List[agent.RemoteSimulator],
    steps_per_actor: int,
) -> List[List[agent.ExpTuple]]:
    """Collect experience from agents.

    Runs `steps_per_actor` time steps of the env for each of the `simulators`.
    """
    all_experience = []
    # Range up to steps_per_actor + 1 to get one more value needed for GAE.
    for _ in range(steps_per_actor + 1):
        key, subkey = jax.random.split(key)
        all_obs = []
        for sim in simulators:
            obs = sim.conn.recv()
            all_obs.append(obs)
        all_obs = np.concatenate(all_obs, axis=0)
        mus, sigmas, values = agent.policy_action(
            train_state.apply_fn,
            train_state.params,
            all_obs,
        )

        mus, sigmas, values = jax.device_get((mus, sigmas, values))
        dist = distrax.MultivariateNormalDiag(mus, sigmas)
        actions = dist.sample(seed=subkey)

        for i, sim in enumerate(simulators):
            sim.conn.send(actions[i])
        experiences = []
        for i, sim in enumerate(simulators):
            obs, action, reward, done = sim.conn.recv()
            value = values[i, 0]
            log_prob = dist[i].log_prob(action)
            sample = agent.ExpTuple(
                obs,
                action,
                reward,
                value,
                log_prob,
                done,
            )
            experiences.append(sample)
        all_experience.append(experiences)
    return all_experience


def get_experience_single(
    env_name,
    key,
    train_state: TrainState,
    steps_per_actor: int,
) -> List[List[agent.ExpTuple]]:
    all_experience = []
    env = env_utils.create_env(env_name)
    obs = env.reset()

    # Range up to steps_per_actor + 1 to get one more value needed for GAE.
    for _ in range(steps_per_actor + 1):
        key, subkey = jax.random.split(key)
        mu, sigma, value = agent.policy_action(
            train_state.apply_fn,
            train_state.params,
            obs,
        )

        # mu, sigma, value = jax.device_get((mu, sigma, value))
        dist = distrax.MultivariateNormalDiag(mu, sigma)

        action, log_prob = dist.sample_and_log_prob(seed=subkey)
        action, log_prob, value = jax.device_get((action, log_prob, value))

        next_obs, reward, done, _ = env.step(action.tolist())

        sample = agent.ExpTuple(
            obs,
            action,
            reward,
            value,
            log_prob,
            done,
        )
        all_experience.append([sample])

        if done:
            obs = env.reset()
        else:
            obs = next_obs

    return all_experience


# TODO review
def process_experience(
    experience: List[List[agent.ExpTuple]],
    actor_steps: int,
    num_agents: int,
    gamma: float,
    lambda_: float,
    input_dims: Tuple,
    output_dims: int,
) -> Tuple:
    """Process experience for training, including advantage estimation.

    Args:
        experience: collected from agents in the form of nested lists/namedtuple
        actor_steps: number of steps each agent has completed
        num_agents: number of agents that collected experience
        gamma: dicount parameter
        lambda_: GAE parameter

    Returns:
        trajectories: trajectories readily accessible for `train_step()` function
    """
    exp_dims = (actor_steps, num_agents)
    values_dims = (actor_steps + 1, num_agents)
    obs = np.zeros(exp_dims + input_dims, dtype=np.float32)
    actions = np.zeros(exp_dims + (output_dims,), dtype=np.float32)
    rewards = np.zeros(exp_dims, dtype=np.float32)
    values = np.zeros(values_dims, dtype=np.float32)
    log_probs = np.zeros(exp_dims, dtype=np.float32)
    dones = np.zeros(exp_dims, dtype=np.float32)

    for t in range(len(experience) - 1):  # experience[-1] only for next_values
        for agent_id, exp_agent in enumerate(experience[t]):
            obs[t, agent_id, ...] = exp_agent.state
            actions[t, agent_id] = exp_agent.action
            rewards[t, agent_id] = exp_agent.reward
            values[t, agent_id] = exp_agent.value
            log_probs[t, agent_id] = exp_agent.log_prob
            # Dones need to be 0 for terminal states.
            dones[t, agent_id] = float(not exp_agent.done)
    for a in range(num_agents):
        values[-1, a] = experience[-1][a].value
    advantages = gae_advantages(rewards, dones, values, gamma, lambda_)
    returns = advantages + values[:-1, :]
    # After preprocessing, concatenate data from all agents.
    trajectories = (obs, actions, log_probs, returns, advantages)
    trajectory_len = num_agents * actor_steps
    trajectories = tuple(
        map(lambda x: np.reshape(x, (trajectory_len,) + x.shape[2:]),
            trajectories))
    return trajectories


@functools.partial(jax.jit, static_argnums=(1, 2))
def get_initial_params(key: np.ndarray, model: nn.Module, input_dims: tuple):
    init_shape = jnp.ones((1, *input_dims), jnp.float32)
    initial_params = model.init(key, init_shape)["params"]
    return initial_params


def create_train_state(
    params,
    model: nn.Module,
    config: ml_collections.ConfigDict,
    train_steps: int,
) -> TrainState:
    if config.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=train_steps,
        )
    else:
        lr = config.learning_rate
    tx = optax.chain(optax.clip(config.clip_grad), optax.adam(lr))
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return train_state


def train(
    model: models.ActorCritic,
    config: ml_collections.ConfigDict,
    model_dir: str,
) -> TrainState:
    """Main training loop.

    Args:
        model: the actor-critic model
        config: object holding hyperparameters and the training information
        model_dir: path to dictionary where checkpoints and logging info are stored

    Returns:
        optimizer: the trained optimizer
    """

    env = config.env
    if config.parallel:
        simulators = [
            agent.RemoteSimulator(env) for _ in range(config.num_agents)
        ]
    summary_writer = tensorboard.SummaryWriter(
        model_dir + f"_{datetime.now().strftime('%d%b%y_%H.%M')}")
    summary_writer.hparams(dict(config))
    loop_steps = int(config.total_steps //
                     (config.num_agents * config.actor_steps))
    # train_step does multiple steps per call for better performance
    # compute number of steps per call here to convert between the number of
    # train steps and the inner number of optimizer steps
    iterations_per_step = config.num_agents * config.actor_steps // config.batch_size

    initial_params = get_initial_params(jax.random.PRNGKey(0), model,
                                        config.input_dims)
    train_state = create_train_state(
        initial_params,
        model,
        config,
        loop_steps * config.num_epochs * iterations_per_step,
    )
    del initial_params

    # train_state = checkpoints.restore_checkpoint(model_dir, train_state)
    # number of train iterations done by each train_step
    start_step = int(train_state.step // config.num_epochs //
                     iterations_per_step)
    logging.info("Start training from step: %s", start_step)
    key = jax.random.PRNGKey(config.seed)

    for step in range(start_step, loop_steps):
        key, subkey = jax.random.split(key)

        # Bookkeeping and testing.
        if step % config.log_freq == 0:
            score = test_episodes.policy_test(
                1,
                train_state.apply_fn,
                train_state.params,
                env,
            )
            frames = step * config.num_agents * config.actor_steps
            summary_writer.scalar("game_score", score, int(frames))
            logging.info(
                "Step %s:\nsteps seen %s\nscore %s\n\n",
                step,
                frames,
                score,
            )

        # Core training code.
        alpha = 1.0 - step / loop_steps if config.decaying_lr_and_clip_param else 1.0

        # with jax.profiler.trace(model_dir):
        # tic = time.perf_counter()
        if config.parallel:
            all_experiences = get_experience(subkey, train_state, simulators,
                                             config.actor_steps)
        else:
            all_experiences = get_experience_single(config.env, subkey,
                                                    train_state,
                                                    config.actor_steps)
        # toc = time.perf_counter()
        # print(f"get experience {toc - tic:0.4f} seconds")

        # tic = time.perf_counter()
        trajectories = process_experience(
            all_experiences,
            config.actor_steps,
            config.num_agents,
            config.gamma,
            config.lambda_,
            config.input_dims,
            config.output_dims,
        )
        # toc = time.perf_counter()
        # print(f"process experience {toc - tic:0.4f} seconds")

        clip_param = config.clip_param * alpha
        total_loss = 0.0
        # tic = time.perf_counter()
        for _ in range(config.num_epochs):
            permutation = np.random.permutation(config.num_agents *
                                                config.actor_steps)
            trajectories = tuple(x[permutation] for x in trajectories)
            key, subkey = jax.random.split(key)
            train_state, loss = train_step(
                train_state,
                trajectories,
                config.batch_size,
                clip_param=clip_param,
                vf_coeff=config.vf_coeff,
                entropy_coeff=config.entropy_coeff,
                key=subkey,
            )
            total_loss += loss
        # toc = time.perf_counter()
        # print(f"train {toc - tic:0.4f} seconds")

        print(f"Step {step} loss: {total_loss:.3f}\t")
        summary_writer.scalar("loss", total_loss,
                              step * config.num_agents * config.actor_steps)

        if (step + 1) % config.checkpoint_freq == 0:
            checkpoints.save_checkpoint(model_dir, train_state, step + 1)
    return train_state
