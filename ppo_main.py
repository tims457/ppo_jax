
import os
from absl import app
from absl import flags
import jax
import jax.random
# import tensorflow as tf
from ml_collections import config_flags

import agent
import models
import ppo_lib
# import env_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'workdir',
    default='/tmp/ppo_training',
    help=('Directory to save checkpoints and logging info.'))

config_flags.DEFINE_config_file(
    'config',
    "cfg/default.py",
    'File path to the default configuration file.',
    lock_config=True)


def main(argv):
  # Make sure tf does not allocate gpu memory.
  # tf.config.experimental.set_visible_devices([], 'GPU')
  config = FLAGS.config
  env = config.env
  model = models.ActorCritic(layers = config.layers, num_outputs=config.output_dims)
  ppo_lib.train(model, config, FLAGS.workdir)

if __name__ == '__main__':
  app.run(main)
