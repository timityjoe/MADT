

# @title Transformer definition
from typing import Any, Callable, Mapping, Optional, Text, Tuple, Union
import haiku as hk
import jax
import jax.numpy as jnp


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
     


