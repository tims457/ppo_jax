"""Definitions of default hyperparameters."""

import ml_collections


def get_config():
    """Get the default configuration.

    The default hyperparameters originate from PPO paper arXiv:1707.06347
    and openAI baselines 2::
    https://github.com/openai/baselines/blob/master/baselines/ppo2/defaults.py
    """
    # initialization and seed
    config = ml_collections.ConfigDict()
    config.seed = 1234
    # Number of agents playing in parallel.
    config.parallel = False
    config.num_agents = 1
    """checkpoint and logging"""
    config.log_freq = 10
    config.checkpoint_freq = 500
    """environment"""
    config.env = "LunarLanderContinuous-v2"
    config.input_dims = (8,)
    config.output_dims = 4
    """model"""
    config.layers = (64, 64, 64)
    """ppo specific"""
    # RL discount parameter.
    config.gamma = 0.99
    # Generalized Advantage Estimation parameter.
    config.lambda_ = 0.95
    # The PPO clipping parameter used to clamp ratios in loss function.
    config.clip_param = 0.2
    # Weight of value function loss in the total loss.
    config.vf_coeff = 0.5
    # Weight of entropy bonus in the total loss.
    config.entropy_coeff = 0.001
    # Number of steps each agent performs in one policy unroll.
    config.actor_steps = 1024 * 2
    # Total number of steps to train for.
    config.total_steps = config.actor_steps * 100
    """optimizer"""
    config.learning_rate = 2.5e-4
    # Batch size used in training.
    config.batch_size = 256
    # Linearly decay lr and clipping parameter to zero during
    # the training.
    config.decaying_lr_and_clip_param = False
    # Number of training epochs per each unroll of the policy.
    config.num_epochs = 10
    # gradient clipping
    config.clip_grad = 0.5

    return config
