#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# From
# https://github.com/google-research/google-research/tree/master/multi_game_dt

#@title Imports
from typing import Any, Callable, Mapping, Optional, Text, Tuple, Union
import jax
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from dopamine.discrete_domains import atari_lib

# 'TPU_DRIVER_MODE' in globals()
# TPU_DRIVER_MODE = 1

# get the latest JAX and jaxlib
# !pip install --upgrade -q jax jaxlib

# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

# !pip install -U dm-haiku
import haiku as hk
import optax

from loguru import logger
from atari.env_wrappers import build_env_fn, _batch_rollout
from madt_model_functions import optimal_action
from atari.madt_atari_env2 import atari_env


# -----------------------------------------------------------------------------
def build_env_fn2(game_name):
  logger.info(f"build_env_fn() game_name:{game_name}")
  """Returns env constructor fn."""

  def env_fn2():
    logger.info("env_fn2()")
    # Mod by Tim:
    env = atari_env("{}".format(args.env), env_conf, args)
    env = SequenceEnvironmentWrapper(env, 4)
    return env

  return env_fn



# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Main() - Start")
    ptitle('Mask A3C Eval')
    # parser = argparse.ArgumentParser(description='Mask-A3C_EVAL')
    # parser.add_argument(
    #     '--convlstm',
    #     action='store_true',
    #     help='Using convLSTM')
    # parser.add_argument(
    #     '--mask_double',
    #     action='store_true',
    #     help='Using mask a3c double')
    # parser.add_argument(
    #     '--mask_single_p',
    #     action='store_true',
    #     help='Using mask a3c single policy')
    # parser.add_argument(
    #     '--mask_single_v',
    #     action='store_true',
    #     help='Using mask a3c single value')
    # parser.add_argument(
    #     '--image',
    #     action='store_true',
    #     help='Using save image')
    # parser.add_argument(
    #     '--env',
    #     default='PongNoFrameskip-v4',
    #     metavar='ENV',
    #     help='environment to train on (default: PongNoFrameskip-v4)')
    # parser.add_argument(
    #     '--env-config',
    #     default='config.json',
    #     metavar='EC',
    #     help='environment to crop and resize info (default: config.json)')
    # parser.add_argument(
    #     '--num-episodes',
    #     type=int,
    #     default=100,
    #     metavar='NE',
    #     help='how many episodes in evaluation (default: 100)')
    # parser.add_argument(
    #     '--load-model-dir',
    #     default='trained_models/',
    #     metavar='LMD',
    #     help='folder to load trained models from')
    # parser.add_argument(
    #     '--load-model',
    #     default='BreakoutNoFrameskip-v4',
    #     metavar='LMN',
    #     help='name to load trained models from')
    # parser.add_argument(
    #     '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
    # parser.add_argument(
    #     '--render',
    #     action='store_true',
    #     help='Watch game as it being played')
    # parser.add_argument(
    #     '--max-episode-length',
    #     type=int,
    #     default=10000,
    #     metavar='M',
    #     help='maximum length of an episode (default: 100000)')
    # parser.add_argument(
    #     '--gpu-ids',
    #     type=int,
    #     default=-1,
    #     help='GPU to use [-1 CPU only] (default: -1)')
    # parser.add_argument(
    #     '--skip-rate',
    #     type=int,
    #     default=4,
    #     metavar='SR',
    #     help='frame skip rate (default: 4)')
    # parser.add_argument(
    #     '--seed',
    #     type=int,
    #     default=1,
    #     metavar='S',
    #     help='random seed (default: 1)')

    # args = parser.parse_args()
    # print(args.load_model)
    # print(args)


    # Select the first game from evaluation config. Feel free to change.
    game_name = 'Breakout'  
    # game_name = 'Asterix'
    # num_envs = 16  # @param
    num_envs = 1  # @param
    
    # Mod by Tim:
    # env_fn = build_env_fn(game_name)
    env_fn = build_env_fn2(game_name)


    # Create a batch of environments to evaluate.
    env_batch = [env_fn() for i in range(num_envs)]

    rng = jax.random.PRNGKey(0)
    # NOTE: the evaluation num_steps is shorter than what is used for paper experiments for speed.
    # rew_sum, frames, rng = _batch_rollout(rng, env_batch, optimal_action, num_steps=5000, log_interval=10)
    rew_sum, frames, rng = _batch_rollout(rng, env_batch, optimal_action, num_steps=500, log_interval=1)
    # rew_sum, frames, rng = _batch_rollout(rng, env_batch, optimal_action, num_steps=5, log_interval=1)

    print('scores:', rew_sum, 'average score:', np.mean(rew_sum))

    print(f'total score: mean: {np.mean(rew_sum)} std: {np.std(rew_sum)} max: {np.max(rew_sum)}')

    # @title Plot scores

    ## plt.ion()
    plt.plot(rew_sum, 'o')
    plt.title(f'Game scores for {game_name}')
    plt.xlabel('trial index')
    plt.ylabel('score')
    plt.show()

    logger.info("Main() End")
     


     

