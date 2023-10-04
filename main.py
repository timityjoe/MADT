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
import functools
import collections
import getpass
import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np
import scipy.linalg
from typing import Generator, Mapping, Sequence, Text
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import time
import gym
import matplotlib.pyplot as plt

import pickle

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

# @title Load model checkpoint

file_path = 'gs://rl-infra-public/multi_game_dt/checkpoint_38274228.pkl'
print('loading checkpoint from:', file_path)
with tf.io.gfile.GFile(file_path, 'rb') as f:
  model_params, model_state = pickle.load(f)

model_param_count = sum(x.size for x in jax.tree_util.tree_leaves(model_params))
print('Number of model parameters: %.2e' % model_param_count)

# @title Utilities

def cross_entropy(logits, labels):
  """Applies sparse cross entropy loss between logits and target labels."""
  labels = jax.nn.one_hot(labels, logits.shape[-1], dtype=logits.dtype)
  loss = -labels * jax.nn.log_softmax(logits)
  return jnp.mean(loss)


def accuracy(logits, labels):
  """Applies sparse cross entropy loss between logits and target labels."""
  predicted_label = jnp.argmax(logits, axis=-1)
  acc = jnp.equal(predicted_label, labels).astype(jnp.float32)
  return jnp.mean(acc)


def add_position_embedding(tokens: jnp.array) -> jnp.array:
  """Add position embedding to a token sequence."""
  assert len(tokens.shape) == 3
  seq_length = tokens.shape[1]
  dim_tokens = tokens.shape[2]
  embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
  pos_emb = hk.get_parameter('positional_embeddings', [seq_length, dim_tokens], init=embed_init)
  tokens = tokens + pos_emb
  return tokens


def image_embedding(
    image: jnp.ndarray,
    output_dim: int,
    is_training: bool,
    output_conv_channels: Optional[int] = 128,
    patch_size: Optional[Tuple[int, int]] = (14, 14),
):
  """Embed [B x T x W x H x C] images to tokens [B x T x output_dim] tokens.

  Args:
    image: [B x T x W x H x C] image to embed.
    output_dim: Output embedding dimensionality.
    is_training: Whether we're training or not.
    output_conv_channels: channel dimensionality of convolution layers (only
      for convoluation networks).
    patch_size: a tuple (patch_height, patch_width), only for patches.

  Returns:
    Image embedding of shape [B x T x output_dim] or [B x T x _ x output_dim].
  """
  assert len(image.shape) == 5

  image_dims = image.shape[-3:]
  batch_dims = image.shape[:2]

  # Reshape to [BT x W x H x C].
  image = jnp.reshape(image, (-1,) + image.shape[-3:])
  # Perform any-image specific processing.
  image = image.astype(jnp.float32) / 255.0

  patch_height, patch_width = patch_size[0], patch_size[1]
  # If patch_size is (14, 14) for example, P = 84 / 14 = 6
  image_emb = hk.Conv2D(
      output_channels=output_dim,
      kernel_shape=(patch_height, patch_width),
      stride=(patch_height, patch_width),
      padding='VALID',
      name='image_emb')(image)  # image_emb is now [BT x P x P x D].

  # Reshape to [B x T x P*P x D].
  image_emb = jnp.reshape(image_emb, batch_dims + (-1, image_emb.shape[-1]))

  emb_init = hk.initializers.RandomNormal(stddev=0.02)
  pos_enc_shape = (1, 1, image_emb.shape[2], image_emb.shape[3])
  pos_enc = hk.get_parameter(
      'image_pos_enc', pos_enc_shape, init=emb_init, dtype=image_emb.dtype)
  image_emb = image_emb + pos_enc
  return image_emb


def sample_from_logits(
    rng: jnp.ndarray,
    logits: jnp.ndarray,
    deterministic: Optional[bool] = False,
    temperature: Optional[float] = 1e+0,
    top_k: Optional[int] = None,
    top_percentile: Optional[float] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Generate a categorical sample from given logits."""
  if deterministic:
    sample = jnp.argmax(logits, axis=-1)
  else:
    rng, sample_rng = jax.random.split(rng)

    if top_percentile is not None:
      percentile = jnp.percentile(logits, top_percentile, axis=-1)
      logits = jnp.where(logits > percentile[..., None], logits, -jnp.inf)
    if top_k is not None:
      logits, top_indices = jax.lax.top_k(logits, top_k)
    sample = jax.random.categorical(sample_rng, temperature * logits, axis=-1)
    if top_k is not None:
      sample_shape = sample.shape
      # Flatten top-k indices and samples for easy indexing.
      top_indices = jnp.reshape(top_indices, [-1, top_k])
      sample = sample.flatten()
      sample = top_indices[jnp.arange(len(sample)), sample]
      # Reshape samples back to original dimensions.
      sample = jnp.reshape(sample, sample_shape)
  return sample, rng


def autoregressive_generate(
    rng: jnp.ndarray,
    logits_fn: Callable[[jnp.ndarray, Mapping[str, jnp.ndarray]], jnp.ndarray],
    inputs: Mapping[str, jnp.ndarray],
    name: str,
    sequence_length: int,
    deterministic: Optional[bool] = False,
    temperature: Optional[float] = 1e+0,
    top_k: Optional[int] = None,
    top_percentile: Optional[float] = None,
    sample_fn: Union[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                     None] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Autoregressively generate an input field given a logit function."""
  val = jnp.zeros_like(inputs[name])

  if sample_fn is None:
    sample_fn = functools.partial(
        sample_from_logits,
        deterministic=deterministic,
        temperature=temperature,
        top_k=top_k,
        top_percentile=top_percentile)

  def loop_step(t, acc_rng):
    acc, rng = acc_rng
    datapoint = dict(inputs)
    datapoint[name] = acc
    logits = logits_fn(rng, datapoint)
    sample, rng = sample_fn(rng, logits[:, t])
    acc = acc.at[:, t].set(sample)
    return (acc, rng)

  val, rng = jax.lax.fori_loop(0, sequence_length, loop_step, (val, rng))
  return val, rng


def make_return(rew: jnp.ndarray):
  """Maximize scoring rewards (rew=1) while not terminating (rew=2)."""
  pos_ret = jnp.sum(rew == 1, axis=-1)
  neg_ret = jnp.sum(rew == 3, axis=-1)
  done = jnp.any(rew == 2, axis=-1)
  return (pos_ret - neg_ret) * (1 - done) - done


def encode_reward(rew: jnp.ndarray) -> jnp.ndarray:
  """Encode reward values into values expected by the model."""
  # 0: no reward   1: positive reward   2: terminal reward   3: negative reward
  rew = (rew > 0) * 1 + (rew < 0) * 3
  return rew.astype(jnp.int32)


def encode_return(ret: jnp.ndarray, ret_range: Tuple[int]) -> jnp.ndarray:
  """Encode (possibly negative) return values into discrete return tokens."""
  ret = ret.astype(jnp.int32)
  ret = jnp.clip(ret, ret_range[0], ret_range[1])
  ret = ret - ret_range[0]
  return ret


def decode_return(ret: jnp.ndarray, ret_range: Tuple[int]) -> jnp.ndarray:
  """Decode discrete return tokens into return values."""
  ret = ret.astype(jnp.int32)
  ret = ret + ret_range[0]
  return ret


# @title Transformer definition


class DenseBlock(hk.Module):
  """A 2-layer MLP which widens then narrows the input."""

  def __init__(self,
               init_scale: float,
               widening_factor: int = 4,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._init_scale = init_scale
    self._widening_factor = widening_factor

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hiddens = x.shape[-1]
    initializer = hk.initializers.VarianceScaling(self._init_scale)
    x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
    x = jax.nn.gelu(x)
    return hk.Linear(hiddens, w_init=initializer)(x)


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
  """Apply a unique LayerNorm to x with default settings."""
  return hk.LayerNorm(
      axis=-1, create_scale=True, create_offset=True, name=name)(
          x)


class CausalSelfAttention(hk.MultiHeadAttention):
  """Self attention with a causal mask applied."""

  def __call__(
      self,
      query: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      value: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
      custom_causal_mask: Optional[jnp.ndarray] = None,
      prefix_length: Optional[int] = 0,
  ) -> jnp.ndarray:
    key = key if key is not None else query
    value = value if value is not None else query

    if query.ndim != 3:
      raise ValueError('Expect queries of shape [B, T, D].')

    seq_len = query.shape[1]
    # If custom_causal_mask is None, the default causality assumption is
    # sequential (a lower triangular causal mask).
    causal_mask = custom_causal_mask
    if causal_mask is None:
      causal_mask = np.tril(np.ones((seq_len, seq_len)))
    causal_mask = causal_mask[None, None, :, :]

    # Similar to T5, tokens up to prefix_length can all attend to each other.
    causal_mask[:, :, :, :prefix_length] = 1
    mask = mask * causal_mask if mask is not None else causal_mask

    return super().__call__(query, key, value, mask)


class Transformer(hk.Module):
  """A transformer stack."""

  def __init__(self,
               num_heads: int,
               num_layers: int,
               dropout_rate: float,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate

  def __call__(self,
               h: jnp.ndarray,
               mask: Optional[jnp.ndarray],
               is_training: bool,
               custom_causal_mask: Optional[jnp.ndarray] = None,
               prefix_length: Optional[int] = 0) -> jnp.ndarray:
    """Connects the transformer.

    Args:
      h: Inputs, [B, T, D].
      mask: Padding mask, [B, T].
      is_training: Whether we're training or not.
      custom_causal_mask: Customized causal mask, [T, T].
      prefix_length: Number of prefix tokens that can all attend to each other.

    Returns:
      Array of shape [B, T, D].
    """

    init_scale = 2. / self._num_layers
    dropout_rate = self._dropout_rate if is_training else 0.
    if mask is not None:
      # Make sure we're not passing any information about masked h.
      h = h * mask[:, :, None]
      mask = mask[:, None, None, :]

    # Note: names chosen to approximately match those used in the GPT-2 code;
    # see https://github.com/openai/gpt-2/blob/master/src/model.py.
    for i in range(self._num_layers):
      h_norm = layer_norm(h, name=f'h{i}_ln_1')
      h_attn = CausalSelfAttention(
          num_heads=self._num_heads,
          key_size=64,
          w_init_scale=init_scale,
          name=f'h{i}_attn')(
              h_norm,
              mask=mask,
              custom_causal_mask=custom_causal_mask,
              prefix_length=prefix_length)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn
      h_norm = layer_norm(h, name=f'h{i}_ln_2')
      h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense
    h = layer_norm(h, name='ln_f')

    return h
  
  # @title Model definition


class DecisionTransformer(hk.Module):
  """Decision transformer module."""

  def __init__(self,
               num_actions: int,
               num_rewards: int,
               return_range: Tuple[int],
               d_model: int,
               num_layers: int,
               dropout_rate: float,
               predict_reward: bool,
               single_return_token: bool,
               conv_dim: int,
               name: Optional[Text] = None):
    super().__init__(name=name)

    # Expected by the transformer model.
    if d_model % 64 != 0:
      raise ValueError(f'Model size {d_model} must be divisible by 64')

    self.num_actions = num_actions
    self.num_rewards = num_rewards
    self.num_returns = return_range[1] - return_range[0]
    self.return_range = return_range
    self.d_model = d_model
    self.predict_reward = predict_reward
    self.conv_dim = conv_dim
    self.single_return_token = single_return_token
    self.spatial_tokens = True

    self.transformer = Transformer(
        name='sequence',
        num_heads=self.d_model // 64,
        num_layers=num_layers,
        dropout_rate=dropout_rate)

  def _embed_inputs(
      self, obs: jnp.array, ret: jnp.array, act: jnp.array, rew: jnp.array,
      is_training: bool) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    # Embed only prefix_frames first observations.
    # obs are [B x T x W x H x C].
    obs_emb = image_embedding(
        obs,
        self.d_model,
        is_training=is_training,
        output_conv_channels=self.conv_dim)
    # Embed returns and actions
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    # Encode returns.
    ret = encode_return(ret, self.return_range)
    rew = encode_reward(rew)
    ret_emb = hk.Embed(self.num_returns, self.d_model, w_init=embed_init)
    ret_emb = ret_emb(ret)
    act_emb = hk.Embed(self.num_actions, self.d_model, w_init=embed_init)
    act_emb = act_emb(act)
    if self.predict_reward:
      rew_emb = hk.Embed(self.num_rewards, self.d_model, w_init=embed_init)
      rew_emb = rew_emb(rew)
    else:
      rew_emb = None
    return obs_emb, ret_emb, act_emb, rew_emb

  def __call__(self, inputs: Mapping[str, jnp.array],
               is_training: bool) -> Mapping[str, jnp.array]:
    """Process sequence."""
    num_batch = inputs['actions'].shape[0]
    num_steps = inputs['actions'].shape[1]
    # Embed inputs.
    obs_emb, ret_emb, act_emb, rew_emb = self._embed_inputs(
        inputs['observations'], inputs['returns-to-go'], inputs['actions'],
        inputs['rewards'], is_training)

    if self.spatial_tokens:
      # obs is [B x T x W x D]
      num_obs_tokens = obs_emb.shape[2]
      obs_emb = jnp.reshape(obs_emb, obs_emb.shape[:2] + (-1,))
      # obs is [B x T x W*D]
    else:
      num_obs_tokens = 1
    # Collect sequence.
    # Embeddings are [B x T x D].
    if self.predict_reward:
      token_emb = jnp.concatenate([obs_emb, ret_emb, act_emb, rew_emb], axis=-1)
      tokens_per_step = num_obs_tokens + 3
      # sequence is [obs ret act rew ... obs ret act rew]
    else:
      token_emb = jnp.concatenate([obs_emb, ret_emb, act_emb], axis=-1)
      tokens_per_step = num_obs_tokens + 2
      # sequence is [obs ret act ... obs ret act]
    token_emb = jnp.reshape(
        token_emb, [num_batch, tokens_per_step * num_steps, self.d_model])
    # Create position embeddings.
    token_emb = add_position_embedding(token_emb)
    # Run the transformer over the inputs.
    # Token dropout.
    batch_size = token_emb.shape[0]
    obs_mask = jnp.ones([batch_size, num_steps, num_obs_tokens])
    ret_mask = jnp.ones([batch_size, num_steps, 1])
    act_mask = jnp.ones([batch_size, num_steps, 1])
    rew_mask = jnp.ones([batch_size, num_steps, 1])
    if self.single_return_token:
      # Mask out all return tokens expect the first one.
      ret_mask = ret_mask.at[:, 1:].set(0)
    if self.predict_reward:
      mask = [obs_mask, ret_mask, act_mask, rew_mask]
    else:
      mask = [obs_mask, ret_mask, act_mask]
    mask = jnp.concatenate(mask, axis=-1)
    mask = jnp.reshape(mask, [batch_size, tokens_per_step*num_steps])

    custom_causal_mask = None
    if self.spatial_tokens:
      # Temporal transformer by default assumes sequential causal relation.
      # This makes the transformer causal mask a lower triangular matrix.
      #     P1 P2 R  a  P1 P2 ... (Ps: image patches)
      # P1  1  0* 0  0  0  0
      # P2  1  1  0  0  0  0
      # R   1  1  1  0  0  0
      # a   1  1  1  1  0  0
      # P1  1  1  1  1  1  0*
      # P2  1  1  1  1  1  1
      # ... (0*s should be replaced with 1s in the ideal case)
      # But, when we have multiple tokens for an image (e.g. patch tokens, conv
      # feature map tokens, etc) as inputs to transformer, this assumption does
      # not hold, because there is no sequential dependencies between tokens.
      # Therefore, the ideal causal mask should not mask out tokens that belong
      # to the same images from each others.

      seq_len = token_emb.shape[1]
      sequential_causal_mask = np.tril(np.ones((seq_len, seq_len)))
      num_timesteps = seq_len // tokens_per_step
      num_non_obs_tokens = tokens_per_step - num_obs_tokens
      diag = [
          np.ones((num_obs_tokens, num_obs_tokens)) if i % 2 == 0 else np.zeros(
              (num_non_obs_tokens, num_non_obs_tokens))
          for i in range(num_timesteps * 2)
      ]
      block_diag = scipy.linalg.block_diag(*diag)
      custom_causal_mask = np.logical_or(sequential_causal_mask, block_diag)
      custom_causal_mask = custom_causal_mask.astype(np.float64)

    output_emb = self.transformer(token_emb, mask, is_training,
                                  custom_causal_mask)
    # Output_embeddings are [B x 3T x D].
    # Next token predictions (tokens one before their actual place).
    ret_pred = output_emb[:, (num_obs_tokens-1)::tokens_per_step, :]
    act_pred = output_emb[:, (num_obs_tokens-0)::tokens_per_step, :]
    embeds = jnp.concatenate([ret_pred, act_pred], -1)
    # Project to appropriate dimensionality.
    ret_pred = hk.Linear(self.num_returns, name='ret_linear')(ret_pred)
    act_pred = hk.Linear(self.num_actions, name='act_linear')(act_pred)
    # Return logits as well as pre-logits embedding.
    result_dict = {
        'embeds': embeds,
        'action_logits': act_pred,
        'return_logits': ret_pred,
    }
    if self.predict_reward:
      rew_pred = output_emb[:, (num_obs_tokens+1)::tokens_per_step, :]
      rew_pred = hk.Linear(self.num_rewards, name='rew_linear')(rew_pred)
      result_dict['reward_logits'] = rew_pred
    # Return evaluation metrics.
    result_dict['loss'] = self.sequence_loss(inputs, result_dict)
    result_dict['accuracy'] = self.sequence_accuracy(inputs, result_dict)
    return result_dict

  def _objective_pairs(self, inputs: Mapping[str, jnp.ndarray],
                       model_outputs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Get logit-target pairs for the model objective terms."""
    act_target = inputs['actions']
    ret_target = encode_return(inputs['returns-to-go'], self.return_range)
    act_logits = model_outputs['action_logits']
    ret_logits = model_outputs['return_logits']
    if self.single_return_token:
      ret_target = ret_target[:, :1]
      ret_logits = ret_logits[:, :1, :]
    obj_pairs = [(act_logits, act_target), (ret_logits, ret_target)]
    if self.predict_reward:
      rew_target = encode_reward(inputs['rewards'])
      rew_logits = model_outputs['reward_logits']
      obj_pairs.append((rew_logits, rew_target))
    return obj_pairs

  def sequence_loss(self, inputs: Mapping[str, jnp.ndarray],
                    model_outputs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute the loss on data wrt model outputs."""
    obj_pairs = self._objective_pairs(inputs, model_outputs)
    obj = [cross_entropy(logits, target) for logits, target in obj_pairs]
    return sum(obj) / len(obj)

  def sequence_accuracy(
      self, inputs: Mapping[str, jnp.ndarray],
      model_outputs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute the accuracy on data wrt model outputs."""
    obj_pairs = self._objective_pairs(inputs, model_outputs)
    obj = [accuracy(logits, target) for logits, target in obj_pairs]
    return sum(obj) / len(obj)

  @staticmethod
  def optimal_action(rng: jnp.ndarray,
                     inputs: jnp.ndarray,
                     logits_fn,
                     return_range: Tuple[int],
                     single_return_token: bool = False,
                     opt_weight: Optional[float] = 0.0,
                     num_samples: Optional[int] = 128,
                     action_temperature: Optional[float] = 1.0,
                     return_temperature: Optional[float] = 1.0,
                     action_top_percentile: Optional[float] = None,
                     return_top_percentile: Optional[float] = None):
    """Calculate optimal action for the given sequence model."""
    obs, act, rew = inputs['observations'], inputs['actions'], inputs['rewards']
    assert len(obs.shape) == 5
    assert len(act.shape) == 2
    inputs = {
        'observations': obs,
        'actions': act,
        'rewards': rew,
        'returns-to-go': jnp.zeros_like(act)
    }
    sequence_length = obs.shape[1]
    # Use samples from the last timestep.
    timestep = -1
    # A biased sampling function that prefers sampling larger returns.
    def ret_sample_fn(rng, logits):
      assert len(logits.shape) == 2
      # Add optimality bias.
      if opt_weight > 0.0:
        # Calculate log of P(optimality=1|return) := exp(return) / Z.
        logits_opt = jnp.linspace(0.0, 1.0, logits.shape[1])
        logits_opt = jnp.repeat(logits_opt[None, :], logits.shape[0], axis=0)
        # Sample from log[P(optimality=1|return)*P(return)].
        logits = logits + opt_weight * logits_opt
      logits = jnp.repeat(logits[None, ...], num_samples, axis=0)
      ret_sample, rng = sample_from_logits(
          rng,
          logits,
          temperature=return_temperature,
          top_percentile=return_top_percentile)
      # Pick the highest return sample.
      ret_sample = jnp.max(ret_sample, axis=0)
      # Convert return tokens into return values.
      ret_sample = decode_return(ret_sample, return_range)
      return ret_sample, rng

    # Set returns-to-go with an (optimistic) autoregressive sample.
    if single_return_token:
      # Since only first return is used by the model, only sample that (faster).
      ret_logits = logits_fn(rng, inputs)['return_logits'][:, 0, :]
      ret_sample, rng = ret_sample_fn(rng, ret_logits)
      inputs['returns-to-go'] = inputs['returns-to-go'].at[:, 0].set(ret_sample)
    else:
      # Auto-regressively regenerate all return tokens in a sequence.
      ret_logits_fn = lambda rng, input: logits_fn(rng, input)['return_logits']
      ret_sample, rng = autoregressive_generate(
          rng,
          ret_logits_fn,
          inputs,
          'returns-to-go',
          sequence_length,
          sample_fn=ret_sample_fn)
      inputs['returns-to-go'] = ret_sample

    # Generate a sample from action logits.
    act_logits = logits_fn(rng, inputs)['action_logits'][:, timestep, :]
    act_sample, rng = sample_from_logits(
        rng,
        act_logits,
        temperature=action_temperature,
        top_percentile=action_top_percentile)
    return act_sample, rng
  
  # @title Atari environment definition

GAME_NAMES = [
    'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
    'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing',
    'Breakout', 'Carnival', 'Centipede', 'ChopperCommand', 'CrazyClimber',
    'DemonAttack', 'DoubleDunk', 'ElevatorAction', 'Enduro', 'FishingDerby',
    'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey',
    'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
    'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall',
    'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner',
    'Robotank', 'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner',
    'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture', 'VideoPinball',
    'WizardOfWor', 'YarsRevenge', 'Zaxxon'
]
ATARI_OBSERVATION_SHAPE = (84, 84, 1)
ATARI_NUM_ACTIONS = 18  # Maximum number of actions in the full dataset.
# rew=0: no reward, rew=1: score a point, rew=2: end game rew=3: lose a point
ATARI_NUM_REWARDS = 4
ATARI_RETURN_RANGE = [
    -20, 100
]  # A reasonable range of returns identified in the dataset

_FULL_ACTION_SET = [
    'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
    'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
    'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
]

_LIMITED_ACTION_SET = {
    'AirRaid': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Alien': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Amidar': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE',
        'LEFTFIRE', 'DOWNFIRE'
    ],
    'Assault': ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Asterix': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT',
        'DOWNLEFT'
    ],
    'Asteroids': [
        'NOOP',
        'FIRE',
        'UP',
        'RIGHT',
        'LEFT',
        'DOWN',
        'UPRIGHT',
        'UPLEFT',
        'UPFIRE',
        'RIGHTFIRE',
        'LEFTFIRE',
        'DOWNFIRE',
        'UPRIGHTFIRE',
        'UPLEFTFIRE',
    ],
    'Atlantis': ['NOOP', 'FIRE', 'RIGHTFIRE', 'LEFTFIRE'],
    'BankHeist': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'BattleZone': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'BeamRider': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'UPRIGHT', 'UPLEFT', 'RIGHTFIRE',
        'LEFTFIRE'
    ],
    'Berzerk': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Bowling': ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE'],
    'Boxing': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Breakout': ['NOOP', 'FIRE', 'RIGHT', 'LEFT'],
    'Carnival': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Centipede': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'ChopperCommand': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'CrazyClimber': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT',
        'DOWNLEFT'
    ],
    'DemonAttack': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'DoubleDunk': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'ElevatorAction': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Enduro': [
        'NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN', 'DOWNRIGHT', 'DOWNLEFT',
        'RIGHTFIRE', 'LEFTFIRE'
    ],
    'FishingDerby': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Freeway': ['NOOP', 'UP', 'DOWN'],
    'Frostbite': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Gopher': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE'
    ],
    'Gravitar': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Hero': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'IceHockey': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Jamesbond': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'JourneyEscape': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT',
        'DOWNLEFT', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE',
        'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Kangaroo': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Krull': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'KungFuMaster': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'DOWNRIGHT', 'DOWNLEFT',
        'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE',
        'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'MontezumaRevenge': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'MsPacman': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT',
        'DOWNLEFT'
    ],
    'NameThisGame': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Phoenix': [
        'NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN', 'RIGHTFIRE', 'LEFTFIRE',
        'DOWNFIRE'
    ],
    'Pitfall': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Pong': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Pooyan': ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE'],
    'PrivateEye': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Qbert': ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN'],
    'Riverraid': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'RoadRunner': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Robotank': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Seaquest': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Skiing': ['NOOP', 'RIGHT', 'LEFT'],
    'Solaris': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'SpaceInvaders': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'StarGunner': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Tennis': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'TimePilot': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE',
        'LEFTFIRE', 'DOWNFIRE'
    ],
    'Tutankham': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE'
    ],
    'UpNDown': ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE'],
    'Venture': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'VideoPinball': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE',
        'LEFTFIRE'
    ],
    'WizardOfWor': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE',
        'LEFTFIRE', 'DOWNFIRE'
    ],
    'YarsRevenge': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Zaxxon': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
}

# An array that Converts an action from a game-specific to full action set.
LIMITED_ACTION_TO_FULL_ACTION = {
    game_name: np.array(
        [_FULL_ACTION_SET.index(i) for i in _LIMITED_ACTION_SET[game_name]])
    for game_name in GAME_NAMES
}

# An array that Converts an action from a full action set to a game-specific
# action set (Setting 0=NOOP if no game-specific action exists).
FULL_ACTION_TO_LIMITED_ACTION = {
    game_name: np.array([(_LIMITED_ACTION_SET[game_name].index(i)
                          if i in _LIMITED_ACTION_SET[game_name] else 0)
                         for i in _FULL_ACTION_SET]) for game_name in GAME_NAMES
}


def _process_observation(obs):
  """Process observation."""
  # Apply jpeg auto-encoding to better match observations in the dataset.
  return tf.io.decode_jpeg(tf.io.encode_jpeg(obs)).numpy()


class AtariEnvWrapper():
  """Environment wrapper with a unified API."""

  def __init__(self, game_name: str, full_action_set: Optional[bool] = True):
    # Disable randomized sticky actions to reduce variance in evaluation.
    self._env = atari_lib.create_atari_environment(
        game_name, sticky_actions=False)
    self.game_name = game_name
    self.full_action_set = full_action_set

  @property
  def observation_space(self) -> gym.Space:
    return self._env.observation_space

  @property
  def action_space(self) -> gym.Space:
    if self.full_action_set:
      return gym.spaces.Discrete(len(_FULL_ACTION_SET))
    return self._env.action_space

  def reset(self) -> np.ndarray:
    """Reset environment and return observation."""
    return _process_observation(self._env.reset())

  def step(self, action: int) -> Tuple[np.ndarray, float, bool, Any]:
    """Step environment and return observation, reward, done, info."""
    if self.full_action_set:
      # atari_py library expects limited action set, so convert to limited.
      action = FULL_ACTION_TO_LIMITED_ACTION[self.game_name][action]
    obs, rew, done, info = self._env.step(action)
    obs = _process_observation(obs)
    return obs, rew, done, info
  
  # @title Build model function

def model_fn(datapoint, is_training=False):
  model = DecisionTransformer(num_actions = ATARI_NUM_ACTIONS,
               num_rewards = ATARI_NUM_REWARDS,
               return_range = ATARI_RETURN_RANGE,
               d_model = 1280,
               num_layers = 10,
               dropout_rate = 0.1,
               predict_reward = True,
               single_return_token = True,
               conv_dim=256)
  return model(datapoint, is_training)

model_fn = hk.transform_with_state(model_fn)

@jax.jit
def optimal_action(rng, inputs):
  logits_fn = lambda rng, inputs: model_fn.apply(
        model_params, model_state, rng, inputs, is_training=False)[0]

  return functools.partial(
            DecisionTransformer.optimal_action,
            rng=rng,
            inputs=inputs,
            logits_fn=logits_fn,
            return_range = ATARI_RETURN_RANGE,
            single_return_token = True,
            opt_weight = 0,
            num_samples = 128,
            action_temperature = 1.0,
            return_temperature = 0.75,
            action_top_percentile = 50,
            return_top_percentile = None)()

# @title Test model function

rng = jax.random.PRNGKey(0)

batch_size = 2
window_size = 4
dummy_datapoint = {'observations': np.zeros((batch_size, window_size,) + ATARI_OBSERVATION_SHAPE),
      'actions': np.zeros([batch_size, window_size], dtype=np.int32),
      'rewards': np.zeros([batch_size, window_size], dtype=np.int32),
      'returns-to-go': np.zeros([batch_size, window_size], dtype=np.int32)}

init_params, init_state = model_fn.init(rng, dummy_datapoint)

result, rng = model_fn.apply(init_params, init_state, rng, dummy_datapoint, is_training=False)
print('Result contains: ', result.keys())

# @title Create environment wrappers

class WrappedGymEnv:

  def __getattr__(self, name):
    """Wrappers forward non-overridden method calls to their wrapped env."""
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

class SequenceEnvironmentWrapper(WrappedGymEnv):
  """Environment wrapper for supporting sequential model inference.
  """

  def __init__(self,
               env,
               num_stack_frames: int = 1):
    self._env = env
    self.num_stack_frames = num_stack_frames
    if self.is_goal_conditioned:
      # If env is goal-conditioned, we want to track goal history.
      self.goal_stack = collections.deque([], maxlen=self.num_stack_frames)
    self.obs_stack = collections.deque([], maxlen=self.num_stack_frames)
    self.act_stack = collections.deque([], maxlen=self.num_stack_frames)
    self.rew_stack = collections.deque([], maxlen=self.num_stack_frames)
    self.done_stack = collections.deque([], maxlen=self.num_stack_frames)
    self.info_stack = collections.deque([], maxlen=self.num_stack_frames)

  @property
  def observation_space(self):
    """Constructs observation space."""
    parent_obs_space = self._env.observation_space
    act_space = self.action_space
    episode_history = {
        'observations': gym.spaces.Box(
            np.stack([parent_obs_space.low] * self.num_stack_frames, axis=0),
            np.stack([parent_obs_space.high] * self.num_stack_frames, axis=0),
            dtype=parent_obs_space.dtype),
        'actions': gym.spaces.Box(
            0, act_space.n, [self.num_stack_frames], dtype=act_space.dtype),
        'rewards': gym.spaces.Box(-np.inf, np.inf, [self.num_stack_frames])
    }
    if self.is_goal_conditioned:
      goal_shape = np.shape(self._env.goal)  # pytype: disable=attribute-error
      episode_history['returns-to-go'] = gym.spaces.Box(
          -np.inf, np.inf, [self.num_stack_frames] + goal_shape)
    return gym.spaces.Dict(**episode_history)

  @property
  def is_goal_conditioned(self):
    return False

  def pad_current_episode(self, obs, n):
    # Prepad current episode with n steps.
    for _ in range(n):
      if self.is_goal_conditioned:
        self.goal_stack.append(self._env.goal)  # pytype: disable=attribute-error
      self.obs_stack.append(np.zeros_like(obs))
      self.act_stack.append(0)
      self.rew_stack.append(0)
      self.done_stack.append(1)
      self.info_stack.append(None)

  def _get_observation(self):
    """Return current episode's N-stacked observation.

    For N=3, the first observation of the episode (reset) looks like:

    *= hasn't happened yet.

    GOAL  OBS  ACT  REW  DONE
    =========================
    g0    0    0.   0.   True
    g0    0    0.   0.   True
    g0    x0   0.   0.   False

    After the first step(a0) taken, yielding x1, r0, done0, info0, the next
    observation looks like:

    GOAL  OBS  ACT  REW  DONE
    =========================
    g0    0    0.   0.   True
    g0    x0   0.   0.   False
    g1    x1   a0   r0   d0

    A more chronologically intuitive way to re-order the column data would be:

    PREV_ACT  PREV_REW  PREV_DONE CURR_GOAL CURR_OBS
    ================================================
    0.        0.        True      g0        0
    0.        0.        False*    g0        x0
    a0        r0        info0     g1        x1

    Returns:
      episode_history: np.ndarray of observation.
    """
    episode_history = {
        'observations': np.stack(self.obs_stack, axis=0),
        'actions': np.stack(self.act_stack, axis=0),
        'rewards': np.stack(self.rew_stack, axis=0),
    }
    if self.is_goal_conditioned:
      episode_history['returns-to-go'] = np.stack(self.goal_stack, axis=0)
    return episode_history

  def reset(self):
    """Resets env and returns new observation."""
    obs = self._env.reset()
    # Create a N-1 "done" past frames.
    self.pad_current_episode(obs, self.num_stack_frames-1)
    # Create current frame (but with placeholder actions and rewards).
    if self.is_goal_conditioned:
      self.goal_stack.append(self._env.goal)
    self.obs_stack.append(obs)
    self.act_stack.append(0)
    self.rew_stack.append(0)
    self.done_stack.append(0)
    self.info_stack.append(None)
    return self._get_observation()

  def step(self, action: np.ndarray):
    """Replaces env observation with fixed length observation history."""
    # Update applied action to the previous timestep.
    self.act_stack[-1] = action
    obs, rew, done, info = self._env.step(action)
    self.rew_stack[-1] = rew
    # Update frame stack.
    self.obs_stack.append(obs)
    self.act_stack.append(0)  # Append unknown action to current timestep.
    self.rew_stack.append(0)
    self.info_stack.append(info)
    if self.is_goal_conditioned:
      self.goal_stack.append(self._env.goal)
    if done:
      if self.is_goal_conditioned:
        # rewrite the observations to reflect hindsight RtG conditioning.
        self.replace_goals_with_hindsight()
    return self._get_observation(), rew, done, info

  def replace_goals_with_hindsight(self):
    # We perform this after rew_stack has been updated.
    assert self.is_goal_conditioned
    window_return = sum(list(self.rew_stack))
    for r in self.rew_stack:
      self.goal_stack.append(window_return)
      window_return -= r

def build_env_fn(game_name):
  """Returns env constructor fn."""

  def env_fn():
    env = AtariEnvWrapper(game_name)
    env = SequenceEnvironmentWrapper(env, 4)
    return env

  return env_fn

# @title Environment rollout


# You can add your own logic and any other collection code here.
def _batch_rollout(rng, envs, policy_fn, num_steps=2500, log_interval=None):
  """Roll out a batch of environments under a given policy function."""
  # observations are dictionaries. Merge into single dictionary with batched
  # observations.
  obs_list = [env.reset() for env in envs]
  num_batch = len(envs)
  obs = tree_util.tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
  ret = np.zeros([num_batch, 8])
  done = np.zeros(num_batch, dtype=np.int32)
  rew_sum = np.zeros(num_batch, dtype=np.float32)
  frames = []
  for t in range(num_steps):
    # Collect observations
    frames.append(
        np.concatenate([o['observations'][-1, ...] for o in obs_list], axis=1))
    done_prev = done

    actions, rng = policy_fn(rng, obs)

    # Collect step results and stack as a batch.
    step_results = [env.step(act) for env, act in zip(envs, actions)]
    obs_list = [result[0] for result in step_results]
    obs = tree_util.tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
    rew = np.stack([result[1] for result in step_results])
    done = np.stack([result[2] for result in step_results])
    # Advance state.
    done = np.logical_or(done, done_prev).astype(np.int32)
    rew = rew * (1 - done)
    rew_sum += rew
    if log_interval and t % log_interval == 0:
      print('step: %d done: %s reward: %s' % (t, done, rew_sum))
    # Don't continue if all environments are done.
    if np.all(done):
      break
  return rew_sum, frames, rng


# Select the first game from evaluation config. Feel free to change.
game_name = 'Breakout'  # @param
num_envs = 16  # @param
env_fn = build_env_fn(game_name)
# Create a batch of environments to evaluate.
env_batch = [env_fn() for i in range(num_envs)]

rng = jax.random.PRNGKey(0)
# NOTE: the evaluation num_steps is shorter than what is used for paper experiments for speed.
rew_sum, frames, rng = _batch_rollout(
    rng, env_batch, optimal_action, num_steps=5000, log_interval=100)

print('scores:', rew_sum, 'average score:', np.mean(rew_sum))

print(f'total score: mean: {np.mean(rew_sum)} std: {np.std(rew_sum)} max: {np.max(rew_sum)}')

# @title Plot scores

plt.plot(rew_sum, 'o')
plt.title(f'Game scores for {game_name}')
plt.xlabel('trial index')
plt.ylabel('score')























if __name__ == "__main__":
    logger.info("Main() - Start")

    # Based on https://stackoverflow.com/questions/67504079/how-to-check-if-an-nvidia-gpu-is-available-on-my-system
    try:
        subprocess.check_output('nvidia-smi')
        print("a GPU is connected.")
    except Exception:
        # TPU or CPU
        if "COLAB_TPU_ADDR" in os.environ and os.environ["COLAB_TPU_ADDR"]:
            import jax.tools.colab_tpu

            jax.tools.colab_tpu.setup_tpu()
            print("A TPU is connected.")
        else:
            print("Only CPU accelerator is connected.")

    logger.info({"devices": jax.local_devices()})

    env = jumanji.make(config["env_name"])
    env = RwareMultiAgentWithGlobalStateWrapper(env)
    if config["add_agent_id"]:
        env = AgentIDWrapper(env=env, has_global_state=True)
    env = AutoResetWrapper(env)
    env = LogWrapper(env)
    eval_env = jumanji.make(config["env_name"])
    eval_env = RwareMultiAgentWithGlobalStateWrapper(eval_env)
    if config["add_agent_id"]:
        eval_env = AgentIDWrapper(env=eval_env, has_global_state=True)

    # PRNG keys.
    logger.info("   PRNG keys")
    rng, rng_e, rng_p = jax.random.split(jax.random.PRNGKey(config["seed"]), num=3)

    # Setup learner.
    logger.info("   Learner")
    learn, actor_network, learner_state = learner_setup(env, (rng, rng_p), config)

    # Setup evaluator.
    logger.info("   Evaluator")
    evaluator, absolute_metric_evaluator, (trained_params, eval_rngs) = evaluator_setup(
            eval_env=eval_env,
            rng_e=rng_e,
            network=actor_network,
            params=learner_state.params.actor_params,
            config=config,
        )
     
    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config["num_updates_per_eval"] = config["num_updates"] // config["num_evaluation"]
    timesteps_per_training = (
        n_devices
        * config["num_updates_per_eval"]
        * config["rollout_length"]
        * config["update_batch_size"]
        * config["num_envs"]
    )
    logger.info(f"  timesteps_per_training:{timesteps_per_training}")

    # Run experiment for a total number of evaluations.
    ep_returns=[]

    # Compiling the learner and evaluator
    print("Compiling the learner and evaluator...")
    with TimeIt(tag="COMPILATION", environment_steps=timesteps_per_training):
        # Compile the learner function to optimize performance during training.
        learn(learner_state)

    # Compile the evaluator function
    print("Compile the evaluator function...")
    _ = evaluator(trained_params, eval_rngs)

    start_time=time.time()

    # Start date and time
    now = datetime.now() 
    start = time.time()
    date_time_start = now.strftime("%m/%d/%Y, %H:%M:%S")
    print("Start date and time:",date_time_start)


    for i in tqdm(range(config["num_evaluation"])):
        # Train.
        print("1) Train")
        with TimeIt("EXECUTION",environment_steps=timesteps_per_training,):
            learner_output = learn(learner_state)
            jax.block_until_ready(learner_output)


        # Prepare for evaluation.
        print("2) Prepare for eval")
        trained_params = jax.tree_util.tree_map(
                lambda x: x[:, 0, ...],
                learner_output.learner_state.params.actor_params,  # Select only actor params
        )
        rng_e, *eval_rngs = jax.random.split(rng_e, n_devices + 1)
        eval_rngs = jnp.stack(eval_rngs)
        eval_rngs = eval_rngs.reshape(n_devices, -1)

        # Evaluate.
        print("3) Evaluate")
        evaluator_output = evaluator(trained_params, eval_rngs)
        jax.block_until_ready(evaluator_output)
        ep_returns=plot_performance(evaluator_output, ep_returns, start_time)

        # Update runner state to continue training.
        print("4) Update learner_state")
        learner_state = learner_output.learner_state

    # Write learner state to file


    # Return trained params to be used for rendering or testing.
    print("Return trained params to be used for rendering or testing...")
    trained_params= jax.tree_util.tree_map(
        lambda x: x[0, 0, ...], learner_output.learner_state.params.actor_params
    )
    print(f"{Fore.CYAN}{Style.BRIGHT}MAPPO experiment completed{Style.RESET_ALL}")

    # Start date and time
    date_time_end = now.strftime("%m/%d/%Y, %H:%M:%S")
    print("End date and time:",date_time_end)    
    
    end = time.time()
    elapsed_time = end - start
    print("Elapsed time seconds", elapsed_time) 

    render_one_episode(config, trained_params)
        

    # import IPython
    # from IPython.display import Image
    # Image(filename='/content/rware.gif',embed=True)
    logger.info("Main() End")
     


     

