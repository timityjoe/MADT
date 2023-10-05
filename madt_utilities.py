

# @title Utilities
from typing import Any, Callable, Mapping, Optional, Text, Tuple, Union
import jax
import jax.numpy as jnp
import haiku as hk


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
     

