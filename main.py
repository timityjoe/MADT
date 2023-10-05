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
# import functools
# import collections
# import getpass
import jax
# import jax.numpy as jnp
# from jax import tree_util
import numpy as np
# import scipy.linalg
# from typing import Generator, Mapping, Sequence, Text
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
# import time
# import gym
import matplotlib.pyplot as plt

# import pickle

# @title Install necessary packages.
# !pip install -U dopamine-rl
# !pip install --upgrade gym
# !pip install gym[atari,accept-rom-license]

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
from env_wrappers import build_env_fn, _batch_rollout
from madt_model_functions import optimal_action

# @title Load model checkpoint - Moved to madt_model_functions.py
# See 
# https://offline-rl.github.io/
# Follow steps for gsutil installation, then
#       gsutil -m cp -R gs://atari-replay-datasets/dqn ./
#       gsutil -m cp -R gs://rl-infra-public/multi_game_dt/checkpoint_38274228.pkl ./
# file_path = 'gs://rl-infra-public/multi_game_dt/checkpoint_38274228.pkl'
# print('loading checkpoint from:', file_path)
# with tf.io.gfile.GFile(file_path, 'rb') as f:
#   model_params, model_state = pickle.load(f)

# model_param_count = sum(x.size for x in jax.tree_util.tree_leaves(model_params))
# print('Number of model parameters: %.2e' % model_param_count)



if __name__ == "__main__":
    logger.info("Main() - Start")

    # Select the first game from evaluation config. Feel free to change.
    game_name = 'Breakout'  # @param
    num_envs = 16  # @param
    env_fn = build_env_fn(game_name)
    # Create a batch of environments to evaluate.
    env_batch = [env_fn() for i in range(num_envs)]

    rng = jax.random.PRNGKey(0)
    # NOTE: the evaluation num_steps is shorter than what is used for paper experiments for speed.
    #rew_sum, frames, rng = _batch_rollout(rng, env_batch, optimal_action, num_steps=5000, log_interval=100)
    rew_sum, frames, rng = _batch_rollout(rng, env_batch, optimal_action, num_steps=5, log_interval=1)

    print('scores:', rew_sum, 'average score:', np.mean(rew_sum))

    print(f'total score: mean: {np.mean(rew_sum)} std: {np.std(rew_sum)} max: {np.max(rew_sum)}')

    # @title Plot scores

    # plt.ion()
    plt.plot(rew_sum, 'o')
    plt.title(f'Game scores for {game_name}')
    plt.xlabel('trial index')
    plt.ylabel('score')
    plt.show()

    logger.info("Main() End")
     


     

