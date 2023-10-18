
# @title Model definition
from typing import Any, Callable, Mapping, Optional, Text, Tuple, Union
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from jax import tree_util
import scipy.linalg
import functools
import os
import pickle
import tensorflow.compat.v2 as tf

import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")

from atari.madt_atari_env import ATARI_NUM_ACTIONS, ATARI_NUM_REWARDS, ATARI_RETURN_RANGE
from madt_transformer import Transformer
from madt_utilities import image_embedding, encode_return, encode_reward, add_position_embedding, cross_entropy, accuracy, sample_from_logits, decode_return

# ---------------------------------------------
# From Jumanji
# model_state = "model_state"
# # print(f"model_checkpoint:{model_checkpoint}")
# path = os.getcwd()
# print("PWD:", path)

# with open(model_state, "rb") as f:
#     model_state = pickle.load(f)

# model_params = first_from_device(model_state.params_state.params)
# ---------------------------------------------

# ---------------------------------------------
# @title Load model checkpoint
# See 
# https://offline-rl.github.io/
# Follow steps for gsutil installation, then
#       gsutil -m cp -R gs://atari-replay-datasets/dqn ./
#       gsutil -m cp -R gs://rl-infra-public/multi_game_dt/checkpoint_38274228.pkl ./
# file_path = 'gs://rl-infra-public/multi_game_dt/checkpoint_38274228.pkl'
file_path = './checkpoint_38274228.pkl'
print('loading checkpoint from:', file_path)
with tf.io.gfile.GFile(file_path, 'rb') as f:
  model_params, model_state = pickle.load(f)

model_param_count = sum(x.size for x in jax.tree_util.tree_leaves(model_params))
print('Number of model parameters: %.2e' % model_param_count)
# ---------------------------------------------

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

    logger.debug("__init__()")

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

    logger.debug("_embed_inputs()")
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
    logger.debug("__call__()")
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
    logger.debug("_objective_pairs()")
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
    logger.debug("sequence_loss()")
    """Compute the loss on data wrt model outputs."""
    obj_pairs = self._objective_pairs(inputs, model_outputs)
    obj = [cross_entropy(logits, target) for logits, target in obj_pairs]
    return sum(obj) / len(obj)

  def sequence_accuracy(
      self, inputs: Mapping[str, jnp.ndarray],
      model_outputs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    logger.debug("sequence_accuracy()")
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

    logger.info("optimal_action()")
    logger.info(f"len(obs.shape): {len(obs.shape)}")
    logger.info(f"len(act.shape): {len(act.shape)}")

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
     


# @title Build model function

from atari.madt_atari_env import ATARI_OBSERVATION_SHAPE

def model_fn(datapoint, is_training=False):
  logger.debug("model_fn()")
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
  logger.debug("optimal_action()")
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
logger.debug("Test model function... ")
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
     

