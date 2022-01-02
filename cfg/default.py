"""Definitions of default hyperparameters."""

import ml_collections


def get_config():
    """Get the default configuration.

  The default hyperparameters originate from PPO paper arXiv:1707.06347
  and openAI baselines 2::
  https://github.com/openai/baselines/blob/master/baselines/ppo2/defaults.py
  """
    config = ml_collections.ConfigDict()
    config.layers = (32, 32, 32)

    config.env = "LunarLanderContinuous-v2"
    config.input_dims = (18,)
    config.output_dims = 4
    # Total number of frames seen during training.
    config.total_steps = 10  # TODO change
    
    config.log_freq = 40
    config.checkpoint_freq = 500
    
    
    # The learning rate for the Adam optimizer.
    config.learning_rate = 2.5e-4
    # Batch size used in training.
    config.batch_size = 256
    # Number of agents playing in parallel.
    config.num_agents = 1
    # Number of steps each agent performs in one policy unroll.
    config.actor_steps = 128
    # Number of training epochs per each unroll of the policy.
    config.num_epochs = 3
    # RL discount parameter.
    config.gamma = 0.99
    # Generalized Advantage Estimation parameter.
    config.lambda_ = 0.95
    # The PPO clipping parameter used to clamp ratios in loss function.
    config.clip_param = 0.1
    # Weight of value function loss in the total loss.
    config.vf_coeff = 0.5
    # Weight of entropy bonus in the total loss.
    config.entropy_coeff = 0.01
    # Linearly decay learning rate and clipping parameter to zero during
    # the training.
    config.decaying_lr_and_clip_param = True
    return config
