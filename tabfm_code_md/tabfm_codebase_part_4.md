# TabFM Codebase Part 4

## File: tabfm/src/jax/memory_efficient_attention.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of memory-efficient attention.

Original version published here: https://arxiv.org/abs/2112.05682
Also known as Flash Attention: https://arxiv.org/abs/2205.14135
"""


from collections.abc import Callable
import functools
from typing import NamedTuple, Optional

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
import jaxtyping as jt; import typeguard; import numpy as np; jt.typed = jt.jaxtyped(typechecker=typeguard.typechecked)

Array = jax.Array | np.ndarray


@jt.typed
def _causal_bias(
    q_len: int,
    k_len: int,
    offset: Optional[int] = None,
    mask_to_bias_factor: float = 1e6,
) -> jt.Float[jax.Array | np.ndarray, 'T_q T_k']:
  """Computes a causal bias matrix.

  Args:
    q_len: Length of queries.
    k_len: Length of keys.
    offset: Optional offset for causal masking.
    mask_to_bias_factor: Value used for masked positions.

  Returns:
    Causal bias matrix.
  """

  q_idxs = lax.broadcasted_iota(dtype=jnp.int32, shape=(q_len, 1), dimension=0)
  k_idxs = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, k_len), dimension=1)
  if offset is not None:
    q_idxs += offset
  inverted_mask = q_idxs < k_idxs  # broadcasts to shape (q_len, k_len)
  return inverted_mask * (-1 * mask_to_bias_factor)


@jt.typed
def _local_causal_bias(
    q_len: int,
    k_len: int,
    query_offset: int,
    key_offset: int,
) -> jt.Float[jax.Array | np.ndarray, 'T_q T_k']:
  """Computes a local causal bias matrix.

  Args:
    q_len: Length of query chunk.
    k_len: Length of key chunk.
    query_offset: Offset of the query chunk.
    key_offset: Offset of the key chunk.

  Returns:
    Local causal bias matrix.
  """

  offset = query_offset - key_offset
  return _causal_bias(q_len, k_len, offset=offset)


class _AttentionSummary(NamedTuple):
  """The summary of the attention over a segment of keys and values.

  Attributes:
    numerator: Sum of values weighted by scores.
    denominator: Sum of exponentiated scores.
    max_so_far: Maximum score encountered.
  """

  numerator: jt.Float[jax.Array | np.ndarray, 'B T N G E']
  denominator: jt.Float[jax.Array | np.ndarray, 'B T N G']
  max_so_far: jt.Float[jax.Array | np.ndarray, 'B T N G']


@jt.typed
def _summarize_chunk(
    query: jt.Float[jax.Array | np.ndarray, 'B T N G E_qk'],
    key: jt.Float[jax.Array | np.ndarray, 'B T_kv N E_qk'],
    value: jt.Float[jax.Array | np.ndarray, 'B T_kv N E_v'],
    current_summary: _AttentionSummary,
    bias: Optional[jt.Float[jax.Array | np.ndarray, '...']],
    precision: Optional[lax.Precision] = None,
) -> _AttentionSummary:
  """Attention for a segment of queries, keys, and values.

  Args:
    query: An array of shape `[batch, q_length, heads, queries_per_head,
      qk_depth_per_head]`.
    key: An array of shape `[batch, kv_length, heads, qk_depth_per_head]`.
    value: An array of shape `[batch, kv_length, heads, v_depth_per_head]`.
    current_summary: The partially summarized queries so far, before adding the
      summarization of this kv chunk.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, heads, queries_per_head, q_length, kv_length]` This can be
      used for incorporating causal masks, padding masks, proximity bias, etc.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    The summary for this segment, consisting of sum of the sum of the
    values weighted by their exponentiated attention scores, the exponentiated
    attention scores, and the maximum score of this segment.
  """
  batch, q_len, q_heads, queries_per_head, q_feat = query.shape
  del q_feat
  _, kv_len, _, v_feat = value.shape
  ref_dtype = value.dtype  # Align dtype to `carry` in jax.lax.cond.

  (numerator, denominator, max_so_far) = current_summary

  attn_weights = jnp.einsum(
      'bqhnd,bkhd->bqhnk', query, key, precision=precision
  )

  if bias is not None:
    bias = jnp.moveaxis(bias, -2, 1)  # move sequence length outside
    attn_weights += bias

  previous_max = max_so_far
  assert previous_max.shape == (batch, q_len, q_heads, queries_per_head)
  chunk_maxima = jnp.max(attn_weights, axis=-1)
  assert chunk_maxima.shape == (batch, q_len, q_heads, queries_per_head)
  max_so_far = jnp.maximum(max_so_far, chunk_maxima)
  max_so_far = jax.lax.stop_gradient(max_so_far)
  correction = jnp.exp(previous_max - max_so_far)
  assert correction.shape == (batch, q_len, q_heads, queries_per_head)

  corrected_weights = jnp.exp(
      attn_weights - max_so_far[:, :, :, :, jnp.newaxis]
  )

  assert corrected_weights.shape == (
      batch,
      q_len,
      q_heads,
      queries_per_head,
      kv_len,
  )

  expected_numerator_shape = (batch, q_len, q_heads, queries_per_head, v_feat)
  assert numerator.shape == expected_numerator_shape, (
      f'numerator.shape is {numerator.shape}, but expected'
      f' {expected_numerator_shape}.'
  )

  numerator = numerator * correction[:, :, :, :, jnp.newaxis]
  numerator = numerator + jnp.einsum(
      'bqhnv,bvhf->bqhnf', corrected_weights, value, precision=precision
  )

  assert denominator.shape == (batch, q_len, q_heads, queries_per_head)
  denominator = denominator * correction
  denominator = denominator + corrected_weights.sum(axis=-1)

  return _AttentionSummary(
      numerator.astype(ref_dtype),
      denominator.astype(ref_dtype),
      max_so_far.astype(ref_dtype),
  )


@jt.typed
def _memory_efficient_attention(
    query: jt.Float[jax.Array | np.ndarray, 'B T N G E_qk'],
    key: jt.Float[jax.Array | np.ndarray, 'B T_kv N E_qk'],
    value: jt.Float[jax.Array | np.ndarray, 'B T_kv N E_v'],
    bias_fn: Callable[[int, int], jt.Float[jax.Array | np.ndarray, '...']],
    query_chunk_size: int,
    key_chunk_size: int,
    precision: Optional[lax.Precision] = None,
    dtype: jnp.dtype = jnp.float32,
    use_extra_logit: bool = False,
    causal_mask: bool = False,
) -> jt.Float[jax.Array | np.ndarray, 'B T N G E_v']:
  """Computes dot-product multiquery-attention given query, key, and value."""
  batch, num_q, heads, queries_per_head, q_feat = query.shape
  batch, num_kv, heads, k_features = key.shape
  batch, num_kv, heads, v_features = value.shape

  num_q_chunks = num_q // query_chunk_size
  num_kv_chunks = num_kv // key_chunk_size

  query = query.reshape(
      (batch, num_q_chunks, query_chunk_size, heads, queries_per_head, q_feat)
  )
  key = key.reshape((batch, num_kv_chunks, key_chunk_size, heads, k_features))
  value = value.reshape(
      (batch, num_kv_chunks, key_chunk_size, heads, v_features)
  )
  # We move the chunk_idx axis to the front to iterate over it with lax.map.
  query = jnp.moveaxis(query, 1, 0)
  key = jnp.moveaxis(key, 1, 0)
  value = jnp.moveaxis(value, 1, 0)

  # The zero_chunk is the output of _summarize_chunk when the inputs are zeros.
  # We define the zero_chunk outside of the loops to prevent the compiler from
  # re-creating these arrays in every loop iteration.
  zero_chunk = _AttentionSummary(
      # numerator
      jnp.zeros(
          (batch, query_chunk_size, heads, queries_per_head, v_features),
          dtype=dtype,
      ),
      # denominator
      jnp.zeros(
          (batch, query_chunk_size, heads, queries_per_head), dtype=dtype
      ),
      # max_so_far
      (-jnp.inf)
      * jnp.ones(
          (batch, query_chunk_size, heads, queries_per_head), dtype=dtype
      ),
  )

  def _query_chunk_attention(args):
    query_chunk, query_chunk_idx = args

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def conditional_summarize_fn(carry, args):
      key_chunk, value_chunk, key_chunk_idx = args

      skip_block = jnp.array(False)
      if causal_mask:
        skip_block = query_chunk_idx < key_chunk_idx

      def cond_fn(query, key, value, carry, key_chunk_idx):
        with jax.named_scope('compute_bias'):
          chunk_bias = bias_fn(query_chunk_idx, key_chunk_idx)
        return (
            _summarize_chunk(
                query, key, value, carry, chunk_bias, precision=precision
            ),
            None,
        )

      return jax.lax.cond(
          skip_block,
          lambda a, b, c, carry, d: (carry, None),
          cond_fn,
          query_chunk,
          key_chunk,
          value_chunk,
          carry,
          key_chunk_idx,
      )

    (numerator, denominator, max_so_far), _ = jax.lax.scan(
        conditional_summarize_fn,
        zero_chunk,
        xs=(key, value, jnp.arange(0, num_kv_chunks)),
    )

    if use_extra_logit:
      denominator += jnp.exp(-max_so_far)

    return numerator / denominator[:, :, :, :, jnp.newaxis]

  res = lax.map(_query_chunk_attention, xs=(query, jnp.arange(0, num_q_chunks)))

  expected_res_shape = (
      num_q_chunks,
      batch,
      query_chunk_size,
      heads,
      queries_per_head,
      v_features,
  )
  assert (
      res.shape == expected_res_shape
  ), f'res.shape is {res.shape}, but expected {expected_res_shape}.'
  res = jnp.moveaxis(res, 0, 1)
  return res.reshape(batch, num_q, heads, queries_per_head, value.shape[-1])


@jt.typed
def dot_product_attention_queries_per_head(
    query: jt.Float[jax.Array | np.ndarray, '... T N G E_qk'],
    key: jt.Float[jax.Array | np.ndarray, '... T_kv N E_qk'],
    value: jt.Float[jax.Array | np.ndarray, '... T_kv N E_v'],
    bias: Optional[jt.Float[jax.Array | np.ndarray, '...']] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = False,
    dropout_rng: Optional[jax.Array | np.ndarray] = None,
    dropout_rate: float = 0.0,
    enable_dropout: bool = False,
    dtype: jnp.dtype = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = False,
    causal_mask: bool = False,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 2048,
) -> jt.Float[jax.Array | np.ndarray, '... T N G E_v']:
  """Computes dot-product attention given query, key, and value.

  Args:
    query: Query tensor.
    key: Key tensor.
    value: Value tensor.
    bias: Optional bias tensor.
    broadcast_dropout: Whether to use broadcasted dropout.
    rescale_logits: Whether to rescale logits.
    dropout_rng: RNG for dropout.
    dropout_rate: Dropout rate.
    enable_dropout: Whether to enable dropout.
    dtype: Computation dtype.
    precision: Numerical precision.
    use_extra_logit: Whether to include virtual extra logit.
    float32_logits: Whether to compute logits in float32.
    causal_mask: Whether to apply causal mask.
    query_chunk_size: control the size of the query chunks.
    key_chunk_size: control the size of the key chunks.

  Returns:
    Attention output.
  """
  assert (
      key.ndim == value.ndim
  ), f'k, v must have same rank. key: {key.shape}, value: {value.shape}'
  assert (
      query.shape[:-4] == key.shape[:-3] == value.shape[:-3]
  ), f'q, k, v batch dim must match. query: {query.shape}'

  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Ensure that we have exactly one batch dimension
  orig_batch_dims = query.shape[:-4]
  query = query.reshape(-1, *query.shape[-4:])
  key = key.reshape(-1, *key.shape[-3:])
  value = value.reshape(-1, *value.shape[-3:])
  if bias is not None:
    bias = bias.reshape(-1, *bias.shape[-4:])

  batch_size, query_length, heads, queries_per_head, _ = query.shape
  _, key_length, _, _ = key.shape

  # TODO: Consider automatic padding to remove this constraint.
  # if query_length % query_chunk_size != 0 and query_length > query_chunk_size:
  #   raise ValueError(
  #       'Sequence length of the query vector %d needs to be less '
  #       'than, or a multiple of the query_chunk_size %d.'
  #       % (query_length, query_chunk_size)
  #   )
  # if key_length % key_chunk_size != 0 and key_length > key_chunk_size:
  #   raise ValueError(
  #       'Sequence length of the key/value vector %d needs to be less '
  #       'than, or a multiple of the key_chunk_size %d.'
  #       % (key_length, key_chunk_size)
  #   )

  # query_chunk_size = jnp.minimum(query_chunk_size, query_length)
  # key_chunk_size = jnp.minimum(key_chunk_size, key_length)

  if bias is not None:
    broadcastable_to = (
        batch_size,
        heads,
        queries_per_head,
        query_length,
        key_length,
    )
    # Check that bias is broadcastable as expected:
    for bias_dim, broadcast_dim in zip(bias.shape, broadcastable_to):
      if bias_dim not in [1, broadcast_dim]:
        raise ValueError(
            f'Expected bias dimensions {bias.shape} to be broadcastable to'
            f' {broadcastable_to}.'
        )
    # bias = jnp.broadcast_to(bias, broadcastable_to)

  if enable_dropout and dropout_rate > 0.0:
    # Precompute dropout
    drop_shape = [batch_size, heads, queries_per_head, query_length, key_length]
    if broadcast_dropout:
      # We mimick the semantics of T5 and broadcast along the "length" dim.
      drop_shape[-2] = 1  # query_length dim
    precomputed_dropout = random.bernoulli(
        dropout_rng, dropout_rate, drop_shape
    )
    precomputed_dropout = jnp.broadcast_to(
        precomputed_dropout,
        (batch_size, heads, queries_per_head, query_length, key_length),
    )
  def bias_fn(
      query_chunk_idx: int,
      key_chunk_idx: int,
  ) -> Array:
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size

    local_bias = jnp.zeros((1, 1, 1, 1, 1), dtype=dtype)
    if bias is not None:
      # If bias is not broadcasted yet, dynamic slice would fail with full slice
      # size. In this case we keep the bias unbroadcasted.
      slice_q_len = min(bias.shape[-2], query_chunk_size)
      slice_k_len = min(bias.shape[-1], key_chunk_size)

      local_bias = lax.dynamic_slice(
          bias,
          # query_offset and key_offset might be > 1 but bias dims might
          # not yet be broadcasted. We rely on the protection against
          # out-of-bounds array accesses built into dynamic_slice.
          start_indices=(0, 0, 0, query_offset, key_offset),
          slice_sizes=(*bias.shape[:3], slice_q_len, slice_k_len),
      )
    if causal_mask:
      causal = _local_causal_bias(
          query_chunk_size, key_chunk_size, query_offset, key_offset
      )
      # add batch, head, and queries_per_head dims
      local_bias += causal.reshape(1, 1, 1, *causal.shape)
    # We implement dropout as part of the bias, which is additive to the
    # attention scores. In some other implementations it is treated as a
    # multiplicative factor applied to the probabilities after softmax.
    if enable_dropout and dropout_rate > 0.0:
      with jax.named_scope('dropout'):
        # If dropout is not broadcasted yet, we need the collapsed dims.
        slice_q_len = query_chunk_size
        slice_k_len = key_chunk_size
        dropout_slice = lax.dynamic_slice(
            precomputed_dropout,
            # query_offset and key_offset might be > 1 but dropout dims might
            # not yet be broadcasted. We rely on the protection against
            # out-of-bounds array accesses built into dynamic_slice.
            start_indices=(0, 0, 0, query_offset, key_offset),
            slice_sizes=(
                *precomputed_dropout.shape[:3],
                slice_q_len,
                slice_k_len,
            ),
        )
        local_bias -= dropout_slice * 1e6
    return local_bias

  # NOTE: T5 does not explicitly rescale the attention logits by
  #       1/sqrt(depth_kq)!  This is folded into the initializers of the
  #       linear transformations, which is equivalent under Adafactor.
  if rescale_logits:
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  result = _memory_efficient_attention(
      query,
      key,
      value,
      bias_fn,
      query_chunk_size=query_chunk_size,
      key_chunk_size=key_chunk_size,
      precision=precision,
      dtype=dtype,
      use_extra_logit=use_extra_logit,
      causal_mask=causal_mask,
  )
  result = result.reshape(*orig_batch_dims, *result.shape[1:])
  return result


@jt.typed
def dot_product_attention_multiquery(
    query: jt.Float[jax.Array | np.ndarray, '... T N E_qk'],
    key: jt.Float[jax.Array | np.ndarray, '... T_kv E_qk'],
    value: jt.Float[jax.Array | np.ndarray, '... T_kv E_v'],
    bias: Optional[jt.Float[jax.Array | np.ndarray, '...']] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = False,
    dropout_rng: Optional[jax.Array | np.ndarray] = None,
    dropout_rate: float = 0.0,
    enable_dropout: bool = True,
    dtype: jnp.dtype = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = False,
    causal_mask: bool = False,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 2048,
) -> jt.Float[jax.Array | np.ndarray, '... T N E_v']:
  """Computes dot-product multiquery-attention given query, key, and value.

  Args:
    query: Query tensor.
    key: Key tensor.
    value: Value tensor.
    bias: Optional bias tensor.
    broadcast_dropout: Whether to use broadcasted dropout.
    rescale_logits: Whether to rescale logits.
    dropout_rng: RNG for dropout.
    dropout_rate: Dropout rate.
    enable_dropout: Whether to enable dropout.
    dtype: Computation dtype.
    precision: Numerical precision.
    use_extra_logit: Whether to include virtual extra logit.
    float32_logits: Whether to compute logits in float32.
    causal_mask: Whether to apply causal mask.
    query_chunk_size: control the size of the query chunks.
    key_chunk_size: control the size of the key chunks.

  Returns:
    Attention output.
  """
  # num_heads, treat current num_heads as queries_per_head
  query = jnp.expand_dims(query, axis=-3)
  # add num_heads dim
  key = jnp.expand_dims(key, axis=-2)
  # add num_heads dim
  value = jnp.expand_dims(value, axis=-2)
  if bias is not None:
    # add num_heads, treat current num_heads dim as queries_per_head
    bias = jnp.expand_dims(bias, axis=-4)
  result = dot_product_attention_queries_per_head(
      query,
      key,
      value,
      bias,
      broadcast_dropout=broadcast_dropout,
      rescale_logits=rescale_logits,
      dropout_rng=dropout_rng,
      dropout_rate=dropout_rate,
      enable_dropout=enable_dropout,
      dtype=dtype,
      precision=precision,
      use_extra_logit=use_extra_logit,
      float32_logits=float32_logits,
      causal_mask=causal_mask,
      query_chunk_size=query_chunk_size,
      key_chunk_size=key_chunk_size,
  )
  return jnp.squeeze(result, axis=-3)  # remove head dim


@jt.typed
def dot_product_attention_multihead(
    query: jt.Float[jax.Array | np.ndarray, '... T N E_qk'],
    key: jt.Float[jax.Array | np.ndarray, '... T_kv N E_qk'],
    value: jt.Float[jax.Array | np.ndarray, '... T_kv N E_v'],
    bias: Optional[jt.Float[jax.Array | np.ndarray, '...']] = None,
    broadcast_dropout: bool = True,
    rescale_logits: bool = False,
    dropout_rng: Optional[jax.Array | np.ndarray] = None,
    dropout_rate: float = 0.0,
    enable_dropout: bool = True,
    dtype: jnp.dtype = jnp.float32,
    precision: Optional[lax.Precision] = None,
    use_extra_logit: bool = False,
    float32_logits: bool = False,
    causal_mask: bool = False,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 2048,
) -> jt.Float[jax.Array | np.ndarray, '... T N E_v']:
  """Computes dot-product multi-head attention given query, key, and value.

  Args:
    query: Query tensor.
    key: Key tensor.
    value: Value tensor.
    bias: Optional bias tensor.
    broadcast_dropout: Whether to use broadcasted dropout.
    rescale_logits: Whether to rescale logits.
    dropout_rng: RNG for dropout.
    dropout_rate: Dropout rate.
    enable_dropout: Whether to enable dropout.
    dtype: Computation dtype.
    precision: Numerical precision.
    use_extra_logit: Whether to include virtual extra logit.
    float32_logits: Whether to compute logits in float32.
    causal_mask: Whether to apply causal mask.
    query_chunk_size: control the size of the query chunks.
    key_chunk_size: control the size of the key chunks.

  Returns:
    Attention output.
  """
  # queries_per_head
  query = jnp.expand_dims(query, axis=-2)
  if bias is not None:
    # add num_heads, treat current num_heads dim as queries_per_head
    bias = jnp.expand_dims(bias, axis=-3)
  result = dot_product_attention_queries_per_head(
      query,
      key,
      value,
      bias,
      broadcast_dropout=broadcast_dropout,
      rescale_logits=rescale_logits,
      dropout_rng=dropout_rng,
      dropout_rate=dropout_rate,
      enable_dropout=enable_dropout,
      dtype=dtype,
      precision=precision,
      use_extra_logit=use_extra_logit,
      float32_logits=float32_logits,
      causal_mask=causal_mask,
      query_chunk_size=query_chunk_size,
      key_chunk_size=key_chunk_size,
  )
  return jnp.squeeze(result, axis=-2)  # remove queries_per_head dim

```

## File: tabfm/src/jax/memory_efficient_attention_test.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for query-varying mask/bias handling in FLASH attention."""

import jax
import jax.numpy as jnp
import numpy as np
from tabfm.src.jax import memory_efficient_attention

from absl.testing import absltest

_BATCH = 2
_HEADS = 2
_DIM = 8


def _reference_attention(query, key, value, bias):
  """Exact attention: the model's JAX backend."""
  return jax.nn.dot_product_attention(
      query=query, key=key, value=value, bias=bias, scale=1.0
  )


def _flash_attention(query, key, value, bias, chunk_size):
  """FLASH attention, called the way MultiheadAttention calls it."""
  return memory_efficient_attention.dot_product_attention_multihead(
      query=query,
      key=key,
      value=value,
      bias=bias,
      dtype=query.dtype,
      query_chunk_size=chunk_size,
      key_chunk_size=chunk_size,
  )


def _mask_to_bias(mask):
  """Keep-mask to additive bias, as MultiheadAttention builds it."""
  return jnp.where(mask, 0.0, -1e30)


class MemoryEfficientAttentionTest(absltest.TestCase):

  def _random_qkv(self, q_len, kv_len):
    rng = np.random.default_rng(0)
    query = jnp.asarray(
        rng.standard_normal((_BATCH, q_len, _HEADS, _DIM)), jnp.float32
    )
    key = jnp.asarray(
        rng.standard_normal((_BATCH, kv_len, _HEADS, _DIM)), jnp.float32
    )
    value = jnp.asarray(
        rng.standard_normal((_BATCH, kv_len, _HEADS, _DIM)), jnp.float32
    )
    return query, key, value

  def _assert_flash_matches_reference(self, q_len, kv_len, bias, chunk_size):
    query, key, value = self._random_qkv(q_len, kv_len)
    actual = _flash_attention(query, key, value, bias, chunk_size)
    expected = _reference_attention(query, key, value, bias)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-5
    )

  def test_query_varying_mask_multiple_chunks(self):
    # A mask that varies per query row must not collapse to the chunk's first row.
    q_len = kv_len = 256
    query_idx = jnp.arange(q_len)[:, None]
    key_idx = jnp.arange(kv_len)[None, :]
    mask = jnp.broadcast_to(
        (key_idx <= query_idx + 50)[None, None], (_BATCH, 1, q_len, kv_len)
    )
    self._assert_flash_matches_reference(
        q_len, kv_len, _mask_to_bias(mask), chunk_size=128
    )

  def test_query_varying_mask_single_chunk(self):
    # Same guarantee when the whole sequence is a single chunk (chunk_size == q_len).
    q_len = kv_len = 64
    query_idx = jnp.arange(q_len)[:, None]
    key_idx = jnp.arange(kv_len)[None, :]
    mask = jnp.broadcast_to(
        (key_idx <= query_idx)[None, None], (_BATCH, 1, q_len, kv_len)
    )
    self._assert_flash_matches_reference(
        q_len, kv_len, _mask_to_bias(mask), chunk_size=q_len
    )

  def test_broadcast_key_mask(self):
    # The [B, 1, 1, S] key-padding mask the shipped model builds must stay exact.
    q_len = kv_len = 256
    key_idx = jnp.arange(kv_len)[None, :]
    mask = jnp.broadcast_to(
        (key_idx < 100)[None, None], (_BATCH, 1, 1, kv_len)
    )
    self._assert_flash_matches_reference(
        q_len, kv_len, _mask_to_bias(mask), chunk_size=128
    )

  def test_kv_broadcastable_bias(self):
    # A bias with kv dim 1 (broadcast over keys) must not crash dynamic_slice.
    q_len = kv_len = 256
    rng = np.random.default_rng(1)
    bias = jnp.asarray(
        rng.standard_normal((_BATCH, 1, q_len, 1)), jnp.float32
    )
    self._assert_flash_matches_reference(q_len, kv_len, bias, chunk_size=128)


if __name__ == "__main__":
  absltest.main()

```

## File: tabfm/src/jax/model.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tabular Foundation Model (TabFM) architecture implementation.

We use the following notation for tensor axes dimensions:

B: Batch size
T: Number of rows (sequence length)
H: Number of columns (features)
E: Embedding size
I: Number of induced tokens
C: Number of CLS tokens
K: Number of classes
D: Head dimension
N: Number of attention heads
G: Feature group size
F: Number of fourier frequencies
M: Number of ensemble estimators
L: Generic linear/feature dimension
Y: Number of layers (a.k.a blocks)
"""

from absl import logging
import chex
import math
from math import pi
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import enum
import typing

import einops
rearrange = einops.rearrange
repeat = einops.repeat
from flax import nnx
from flax.nnx import Module
import jax
import jax.numpy as jnp
import jaxtyping as jt; import typeguard; import numpy as np; jt.typed = jt.jaxtyped(typechecker=typeguard.typechecked)

Array = jax.Array | np.ndarray
DType = Any
PRNGKey = jax.Array
Float = jt.Float

from . import memory_efficient_attention



class AttentionImplementation(str, enum.Enum):
  JAX = 'jax'
  # vmaps jax.nn.dot_product_attention over the head dimension to save memory
  JAX_VMAP_ON_HEAD_DIM = 'jax_vmap_on_head_dim'
  FLASH = 'flash'
  NONE = 'none'


# Rotary Embedding


@jt.typed
def exists(val: Any) -> bool:
  """Checks if a value is not None."""
  return val is not None


@jt.typed
def default(val: Any, d: Any) -> Any:
  """Returns 'val' if it exists, otherwise returns 'd'."""
  return val if exists(val) else d


@jt.typed
def rotate_half(x: jt.Float[jax.Array | np.ndarray, '*B D_r']) -> jt.Float[jax.Array | np.ndarray, '*B D_r']:
  """Rotates half of the input tensor's dimensions for rotary positional embeddings.

  This operation splits the last dimension into two halves and swaps them,
  negating one half.

  Args:
    x: Input tensor to rotate.

  Returns:
    Rotated tensor.
  """
  # Reshape the input to separate the rotation dimension
  x = rearrange(x, '... (d r) -> ... d r', r=2)
  # Split the last dimension into two parts (x1 and x2 correspond to the original x[..., :d] and x[..., d:])
  x1, x2 = x[..., 0], x[..., 1]
  # Stack them back with x2 negated and in the first position
  x = jnp.stack((-x2, x1), axis=-1)
  # Reshape back to the original format
  return rearrange(x, '... d r -> ... (d r)')


@jt.typed
def apply_rotary_emb(
    freqs: jax.Array | np.ndarray,
    t: jax.Array | np.ndarray,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
) -> jax.Array | np.ndarray:
  """Applies rotary positional embeddings to a tensor 't'.

  Args:
    freqs: Frequencies tensor for the rotation.
    t: Input tensor to apply rotary embedding to, typically query or key.
    start_index: The starting index in the embedding dimension to apply the
      rotation. Useful if only a part of the embedding needs rotation.
    scale: Scaling factor to apply after rotation (used in XPOS).
    seq_dim: The dimension corresponding to the sequence length in the input
      tensor 't'.

  Returns:
    Tensor with rotary embeddings applied.
  """
  dtype = t.dtype

  # Adjust frequencies if the input tensor has 3 dimensions (e.g., (batch, seq_len, dim))
  # and we need to match the sequence length.
  if t.ndim == 3:
    # Get the current sequence length from the specified sequence dimension
    current_seq_len = t.shape[seq_dim]
    # Slice freqs to match the current sequence length. Assumes freqs are pre-computed
    # for a maximum possible sequence length and we take the tail.
    freqs = freqs[freqs.shape[0] - current_seq_len :]

  rot_dim = freqs.shape[-1]  # Dimension of the part to be rotated
  end_index = start_index + rot_dim

  assert rot_dim <= t.shape[-1], (
      f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in'
      f' all the positions {rot_dim}'
  )

  # Split 't' into three parts:
  # 1. 't_left': The part of the tensor before 'start_index' (not rotated).
  # 2. 't_middle': The part of the tensor from 'start_index' to 'end_index' (will be rotated).
  # 3. 't_right': The part of the tensor after 'end_index' (not rotated).
  t_left = t[..., :start_index]
  t_middle = t[..., start_index:end_index]
  t_right = t[..., end_index:]

  # Apply the rotary embedding transformation:
  # (t_middle * cos(freqs) * scale) + (rotate_half(t_middle) * sin(freqs) * scale)
  t_transformed = (t_middle * jnp.cos(freqs) * scale) + (
      rotate_half(t_middle) * jnp.sin(freqs) * scale
  )

  # Concatenate the three parts back together to form the output tensor
  out = jnp.concatenate((t_left, t_transformed, t_right), axis=-1)

  # Ensure the output tensor has the original dtype
  return out.astype(dtype)


class PerDimScale(nnx.Module):
  """Per-dimension scaling."""

  __data__ = ('per_dim_scale',)

  @jt.typed
  def __init__(self, num_dims: int, *, rngs: Any):
    del rngs
    self.num_dims = num_dims
    self.per_dim_scale = nnx.Param(jnp.zeros(shape=(num_dims,)))

  @jt.typed
  def __call__(self, x: jt.Float[jax.Array | np.ndarray, '*B L']) -> jt.Float[jax.Array | np.ndarray, '*B L']:
    """Applies per-dimension scaling.

    Args:
      x: Input tensor.

    Returns:
      Scaled tensor.
    """
    return x * (
        1.442695041
        / jnp.sqrt(self.num_dims)
        * jax.nn.softplus(self.per_dim_scale)
    ).astype(x.dtype)


class RotaryEmbedding(nnx.Module):
  """RotaryEmbedding is a module that implements rotary positional embeddings for

  use in transformer models. Rotary embeddings encode positional information
  in a way that allows continuous rotation of embeddings, enhancing the model's
  ability to capture long-range dependencies and positional relations.
  """

  def __init__(
      self,
      dim: int,
      custom_freqs: Optional[jax.Array | np.ndarray] = None,
      freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
      theta: float = 10000,
      max_freq: int = 10,
      num_freqs: int = 1,
      interpolate_factor: float = 1.0,
      theta_rescale_factor: float = 1.0,
      seq_before_head_dim: bool = True,
      *,
      rngs: Any,  # Required for Flax nnx modules, even if not directly used for random init here.
      dtype: DType = jnp.bfloat16,
  ):
    """Initializes the RotaryEmbedding module.

    Parameters
    ----------
    dim (int): The dimension of the embeddings to apply RoPE to.
    custom_freqs (jax.Array | np.ndarray | None): Custom frequency tensor. If None,
                                    frequencies are generated based on
                                    `freqs_for`.
    freqs_for (Literal["lang", "pixel", "constant"]): Specifies the type of
    frequencies.
                                                    'lang' (language), 'pixel'
                                                    (image), or 'constant'.
    theta (float): Base scaling factor for language frequencies.
    max_freq (int): Maximum frequency for pixel-based embeddings.
    num_freqs (int): Number of frequencies for 'constant' type.
    interpolate_factor (float): Factor to interpolate sequence length for longer
    sequences.
    theta_rescale_factor (float): Rescaling factor for theta for longer
    sequences (NTK-aware scaling).
    seq_before_head_dim (bool): If True, the sequence dimension is before the
    head dimension (e.g., (B, S, H, D)).
                                Otherwise, it's typically (B, H, S, D).
    rngs (nnx.Rngs): Random number generators for NNX.
    dtype (DType): Data type for calculations.
    """
    self.dim = dim
    self.freqs_for = freqs_for
    self.interpolate_factor = interpolate_factor
    self.dtype = dtype

    assert dim >=2, f'dim must be at least 2. Got {dim}'
    # Apply theta rescaling based on NTK-aware scaling for longer sequence lengths
    theta *= theta_rescale_factor ** (dim / (dim - 2))

    # Initialize frequencies based on 'freqs_for'
    if exists(custom_freqs):
      freqs_init = custom_freqs
    elif freqs_for == 'lang':
      # Language-based frequencies (standard RoPE formulation)
      freqs_init = 1.0 / (
          theta
          ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim)
      )
    elif freqs_for == 'pixel':
      # Pixel-based frequencies (often used for images)
      freqs_init = jnp.linspace(1.0, max_freq / 2, dim // 2) * pi
    elif freqs_for == 'constant':
      # Constant frequencies
      freqs_init = jnp.ones(num_freqs).astype(dtype)
    else:
      raise ValueError(f'Unknown freqs_for type: {freqs_for}')

    # Define 'freqs' as an nnx.Param if learnable, otherwise as a regular nnx.Variable
    assert freqs_init is not None  # For pytype.
    self.freqs = nnx.Variable(freqs_init.astype(self.dtype))

    # Determine the default sequence dimension based on `seq_before_head_dim`
    self.default_seq_dim = -3 if seq_before_head_dim else -2

  @jt.typed
  def get_seq_pos(
      self, seq_len: Any, dtype: Any, offset: int = 0
  ) -> jt.Float[jax.Array | np.ndarray, 'T']:
    """Computes the sequence positions for rotary embeddings.

    Args:
      seq_len: Length of the sequence.
      dtype: Data type for the output tensor.
      offset: Offset for the sequence positions.

    Returns:
      Sequence positions tensor.
    """
    return (jnp.arange(seq_len, dtype=dtype) + offset) / self.interpolate_factor

  @jt.typed
  def rotate_queries_or_keys(
      self,
      t: jax.Array | np.ndarray,
      seq_dim: Optional[int] = None,
      offset: int = 0,
      scale: Optional[float] = None,
  ) -> jax.Array | np.ndarray:
    """Applies rotary embeddings to a single tensor (either queries or keys).

    Args:
      t: Input tensor to apply rotary embedding to.
      seq_dim: The dimension corresponding to the sequence length.
      offset: Offset for the sequence positions.
      scale: Scaling factor to apply after rotation.

    Returns:
      Tensor with rotary embeddings applied.
    """
    seq_dim = default(seq_dim, self.default_seq_dim)

    dtype, seq_len = t.dtype, t.shape[seq_dim]

    # Get sequence positions
    seq = self.get_seq_pos(seq_len, dtype=dtype, offset=offset)

    # Call the module's __call__ method to get frequencies
    freqs = self(seq, seq_len=seq_len, offset=offset)

    # If the sequence dimension is -3 (e.g., (batch, seq, head, dim)),
    # rearrange frequencies to match the broadcastable shape.
    if seq_dim == -3:
      freqs = rearrange(freqs, 'n d -> n 1 d')

    # Apply the rotary embedding transformation
    return apply_rotary_emb(
        freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim
    )

  @jt.typed
  def __call__(
      self,
      t: jt.Float[jax.Array | np.ndarray, '*B T'],
      seq_len: Any = None,
      offset: int = 0,
  ) -> jt.Float[jax.Array | np.ndarray, '*B T D_rot']:
    """The main forward pass for generating rotary embedding frequencies.

    Args:
      t: Sequence positions tensor.
      seq_len: Length of the sequence.
      offset: Offset for the sequence positions.

    Returns:
      Frequencies tensor.
    """
    # Get the base frequencies (either learned or pre-computed)
    freqs = self.freqs

    # Calculate frequencies by multiplying positions (t) with base frequencies
    freqs = jnp.einsum('..., f -> ... f', t.astype(freqs.dtype), freqs)
    # Repeat frequencies to match the 'r=2' pattern for complex number rotation
    freqs = repeat(freqs, '... n -> ... (n r)', r=2)

    return freqs


# Attention


# ----------------------------------------------------------------------------
# 1. Helper Functions
#
# Utility functions used by the NNX modules.
# ----------------------------------------------------------------------------


@jt.typed
def get_activation(activation: Union[str, Callable]) -> Callable:
  """Get activation function class from a string name.

  Translated to return JAX activation functions.
  """
  if callable(activation):
    return activation

  activation_map = {
      'relu': nnx.relu,
      'leaky_relu': nnx.leaky_relu,
      'gelu': nnx.gelu,
      'tanh': nnx.tanh,
  }

  if activation not in activation_map:
    raise ValueError(
        f'Unknown activation: {activation}. '
        f'Supported: {list(activation_map.keys())}'
    )
  return activation_map[activation]


# ----------------------------------------------------------------------------
# 2. Translated NNX Modules
#
# flax.nnx.Module implementations.
# ----------------------------------------------------------------------------


class OneHotAndLinear(nnx.Module):
  """Combines one-hot encoding and linear projection in a single efficient

  operation. Flax/NNX implementation.

  Parameters
  ----------
  num_classes : int
      Number of distinct categories for one-hot encoding.
  embed_dim : int
      Output embedding dimension.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      num_classes: int,
      embed_dim: int,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.num_classes = num_classes
    self.embed_dim = embed_dim
    self.projection = nnx.Linear(
        self.num_classes, self.embed_dim, rngs=rngs, dtype=dtype
    )
    self.dtype = dtype

  @jt.typed
  def __call__(self, src: jt.Shaped[jax.Array | np.ndarray, '... T']) -> jt.Float[jax.Array | np.ndarray, '... T E']:
    """Transforms integer indices to dense embeddings.

    Args:
      src: Integer tensor containing category indices.

    Returns:
      Embedded representation.
    """
    # The dtype of the one-hot vector must match the kernel's dtype.
    one_hot = jax.nn.one_hot(
        src, self.num_classes, dtype=self.projection.kernel.dtype
    )
    return self.projection(one_hot)


class MLP(nnx.Module):
  """Multi-layer perceptron with configurable architecture.

  Flax/NNX implementation.

  Parameters
  ----------
  in_dim : int
      Input feature dimension.
  out_dim : Optional[int], default=None
      Output dimension. If None, uses the last hidden dimension.
  hidden_dims : List[int], default=[256, 256, 256]
      Dimensions of hidden layers.
  activation : str, default='gelu'
      Activation function: 'relu', 'gelu', 'leaky_relu', or 'tanh'.
  use_bias : bool, default=True
      Whether to include bias terms in linear layers.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      in_dim: int,
      out_dim: Optional[int] = None,
      hidden_dims: Tuple[int, ...] = (256, 256, 256),
      activation: str = 'gelu',
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.layers = nnx.List()
    act_fn = get_activation(activation)
    self.dtype = dtype

    prev_dim = in_dim
    for hidden_dim in hidden_dims:
      self.layers.append(
          nnx.Linear(
              prev_dim, hidden_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
          )
      )
      self.layers.append(act_fn)
      prev_dim = hidden_dim

    if out_dim is not None:
      self.layers.append(
          nnx.Linear(
              prev_dim, out_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
          )
      )

  @jt.typed
  def __call__(self, x: jt.Float[jax.Array | np.ndarray, '... L_in']) -> jt.Float[jax.Array | np.ndarray, '... L_out']:
    """Forward pass through the MLP.

    Args:
      x: Input tensor.

    Returns:
      Output tensor.
    """
    for layer in self.layers:
      x = layer(x)
    return x


@jt.typed
def _logn(n: int, dtype: jnp.dtype) -> jax.Array | np.ndarray:
  """Compute :math:`\\log(n)` safely, avoiding fp16 overflow for large ``n``."""
  return jnp.array(math.log(max(n, 1)), dtype=dtype)


@jt.typed
def _extract_kv_from_cache(
    cached_kv: Tuple[jt.Float[jax.Array | np.ndarray, 'B T N D'],
                     jt.Float[jax.Array | np.ndarray, 'B T N D']]
) -> Tuple[jt.Float[jax.Array | np.ndarray, 'B T N D'],
           jt.Float[jax.Array | np.ndarray, 'B T N D']]:
  """Extracts key and value tensors from the cache.

  Args:
    cached_kv: Tuple of (key, value) tensors.

  Returns:
    Tuple of (key, value) tensors.
  """
  assert isinstance(cached_kv, tuple), (
      'cached_kv must be a tuple of (k, v).',
      f'cached_kv type: {type(cached_kv)}',
  )
  return cached_kv


@jt.typed
def _encode_kv_into_cache(
    k: jt.Float[jax.Array | np.ndarray, 'B T N D'],
    v: jt.Float[jax.Array | np.ndarray, 'B T N D']
) -> Tuple[jt.Float[jax.Array | np.ndarray, 'B T N D'],
           jt.Float[jax.Array | np.ndarray, 'B T N D']]:
  """Encodes key and value tensors into the cache.

  Args:
    k: Key tensor.
    v: Value tensor.

  Returns:
    Tuple of (key, value) tensors.
  """
  return (k, v)


class MultiheadAttention(nnx.Module):
  """Enhanced multi-head attention with rotary positional embedding support.

  Flax/NNX implementation.

  The logic from the original `multi_head_attention_forward` is integrated
  directly into the `__call__` method for better encapsulation.

  Parameters
  ----------
  embed_dim : int
      Model dimension.
  num_heads : int
      Number of attention heads.
  use_bias : bool, default=True
      Whether to use bias in projection layers.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.attention_impl = attention_impl
    self.dtype = dtype


    assert (
        self.head_dim * num_heads == self.embed_dim
    ), 'embed_dim must be divisible by num_heads'

    kwargs = {}
    if zero_out_proj_init:
      kwargs['kernel_init'] = nnx.initializers.zeros
      if use_bias:
        kwargs['bias_init'] = nnx.initializers.zeros

    self.out_proj = nnx.Linear(
        embed_dim,
        embed_dim,
        use_bias=use_bias,
        rngs=rngs,
        dtype=self.dtype,
        **kwargs,
    )
    # self.out_proj.kernel[...] = jnp.zeros_like(self.out_proj.kernel[...])
    # if self.out_proj.bias is not None:
    #     self.out_proj.bias[...] = jnp.zeros_like(self.out_proj.bias[...])
    self.q_proj = nnx.Linear(
        self.embed_dim, self.embed_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
    )
    self.k_proj = nnx.Linear(
        self.embed_dim, self.embed_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
    )
    self.v_proj = nnx.Linear(
        self.embed_dim, self.embed_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
    )
    self.query_ln = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=self.dtype)
    self.key_ln = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=self.dtype)
    self.per_dim_scale = PerDimScale(num_dims=self.head_dim, rngs=rngs)

  @jt.typed
  def __call__(
      self,
      query: jt.Float[jax.Array | np.ndarray, 'B T E'],
      key: jt.Float[jax.Array | np.ndarray, 'B T_src E'] | None,
      value: jt.Float[jax.Array | np.ndarray, 'B T_src E'] | None,
      attn_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B #N #T T_src']] = None,
      rope: Optional[RotaryEmbedding] = None,
      cached_kv: Optional[
          Tuple[
              jt.Float[jax.Array | np.ndarray, 'B T_src N D'],
              jt.Float[jax.Array | np.ndarray, 'B T_src N D'],
          ]
      ] = None,
      return_kv: bool = False,
  ) -> Union[
      jt.Float[jax.Array | np.ndarray, 'B T E'],  # Output tensor.
      Tuple[
          jt.Float[jax.Array | np.ndarray, 'B T E'],  # Output tensor.
          Tuple[
              jt.Float[jax.Array | np.ndarray, 'B T_src N D'],  # New key cache
              jt.Float[jax.Array | np.ndarray, 'B T_src N D'],  # New value cache
          ],
      ],
  ]:
    """Computes multi-head attention.

    Args:
      query: Query tensor.
      key: Key tensor.
      value: Value tensor.
      attn_mask: Boolean mask for attention.
      rope: Rotary positional embedding instance.
      cached_kv: Cached key and value tensors. If provided, then key and value
        must be None since we are in decode mode and will extract k, v from
        cached_kv.
      return_kv: If True, returns the computed key and value tensors.

    Returns:
      Attention output tensor or (output, (k, v)).
    """
    batch_size, tgt_len, _ = query.shape
    dtype = query.dtype

    is_prefill = return_kv
    is_decode = cached_kv is not None
    is_train = (not return_kv) and (cached_kv is None)
    assert int(is_prefill) + int(is_decode) + int(is_train) == 1, (
        'Exactly one of is_prefill, is_decode, and is_train can be True.'
        + f' Got {is_prefill=}, {is_decode=}, {is_train=}.'
    )

    if attn_mask is not None:
      assert attn_mask.shape[0] == batch_size, (
          f'attn_mask batch size must match query batch size: {attn_mask.shape[0]} != {batch_size}'
      )
    if self.attention_impl == AttentionImplementation.NONE:
      # For None attention, kv cache is empty.
      if return_kv:
        return query, (jnp.array([]), jnp.array([]))
      else:
        return query

    # 1. Project Q
    q = self.q_proj(query)
    q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim)

    # 2. Handle K, V
    if cached_kv is not None:
      assert key is None, f'key must be None if cached_kv is not None {key=}'
      assert value is None, f'value must be None if cached_kv is not None {value=}.'
      k, v = _extract_kv_from_cache(cached_kv)
      src_len = k.shape[-3]  # (Batch, Seq, Head, Dim)
    else:
      assert key is not None, f'key must not be None if cached_kv is None.'
      assert value is not None, f'value must not be None if cached_kv is None.'
      src_len = key.shape[-2]
      k = self.k_proj(key)
      v = self.v_proj(value)
      k = k.reshape(batch_size, src_len, self.num_heads, self.head_dim)
      v = v.reshape(batch_size, src_len, self.num_heads, self.head_dim)

    # 3. Apply RoPE if provided
    if rope is not None:
      q = rope.rotate_queries_or_keys(q)
      # Only rotate K if it was just computed (not cached).
      # We assume cached_kv is already rotated.
      if cached_kv is None:
        k = rope.rotate_queries_or_keys(k)

    q = self.query_ln(q)
    # Only normalize K if it was just computed
    if cached_kv is None:
      k = self.key_ln(k)
    q = self.per_dim_scale(q)


    new_kv = _encode_kv_into_cache(k, v)
    if attn_mask is not None:
      assert attn_mask.ndim == 4, 'attn_mask must be 4D'
      assert (
          attn_mask.shape[-3] == 1
      ), 'attn_mask with ndim=4 must have shape[-3]==1 when using JAX attention'

    if self.attention_impl == AttentionImplementation.FLASH:
      attention_bias = None
      if attn_mask is not None:
        # This assumes attn_mask is not all zeros otherwise this will result
        # in attending to everything equally.
        attention_bias = jnp.where(attn_mask, 0.0, -1e30)
      attn_output = memory_efficient_attention.dot_product_attention_multihead(
          query=q,
          key=k,
          value=v,
          bias=attention_bias,
          dtype=dtype,
          query_chunk_size=128 if tgt_len >= 128 else tgt_len,
          key_chunk_size=128 if src_len >= 128 else src_len,
      )
    elif self.attention_impl == AttentionImplementation.JAX:
      attn_output = jax.nn.dot_product_attention(
          query=q,
          key=k,
          value=v,
          mask=attn_mask,
          scale=1.0,
      )
    elif self.attention_impl == AttentionImplementation.JAX_VMAP_ON_HEAD_DIM:
      if attn_mask is not None:
        attn_mask = attn_mask[:, 0, :, :]
      q_h = jnp.swapaxes(q, -2, 0)
      k_h = jnp.swapaxes(k, -2, 0)
      v_h = jnp.swapaxes(v, -2, 0)

      # Define the function to apply to each head.
      @jax.remat
      def _attention_fn(inputs):
        queries, keys, values = inputs
        return jax.nn.dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            mask=attn_mask,
            scale=1.0,
        )

      # Map the attention function over the head dimension.
      attn_output_h = jax.lax.map(_attention_fn, (q_h, k_h, v_h))

      # Move the head dimension back to its original position.
      attn_output = jnp.swapaxes(attn_output_h, 0, -2)
    else:
      raise ValueError(
          'Unsupported attention implementation: %s'
          % self.attention_impl
      )

    # 6. Reshape and final projection
    flat_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)
    out = self.out_proj(flat_output)

    if return_kv:
      return out, new_kv
    return out


class MultiheadAttentionBlock(nnx.Module):
  """Attention block supporting RoPE.

  Flax/NNX implementation of TransformerEncoderLayer.

  Parameters
  ----------
  d_model : int
      Model dimension.
  nhead : int
      Number of attention heads.
  dim_feedforward : int
     Dimension of the feedforward network.
  activation : str or unary callable, default="gelu"
      The activation function for the feedforward network.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      activation: str | Callable = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.attn = MultiheadAttention(
        d_model,
        nhead,
        rngs=rngs,
        dtype=dtype,
        attention_impl=attention_impl,
        zero_out_proj_init=zero_out_proj_init,
    )
    self.pre_attn_ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype)
    self.post_attn_ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype)
    self.pre_ff_ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype)
    self.post_ff_ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype)

    kwargs = {}
    if zero_out_proj_init:
      kwargs['kernel_init'] = nnx.initializers.zeros
      kwargs['bias_init'] = nnx.initializers.zeros

    self.activation_name = (
        activation
        if isinstance(activation, str)
        else getattr(activation, '__name__', None)
    )

    if self.activation_name == 'swiglu':
      self.linear1_gate = nnx.Linear(
          d_model,
          dim_feedforward,
          rngs=rngs,
          dtype=dtype,
          use_bias=use_bias,
          **kwargs,
      )
      self.activation = nnx.silu
    else:
      self.activation = get_activation(activation)

    self.linear1 = nnx.Linear(
        d_model,
        dim_feedforward,
        rngs=rngs,
        dtype=dtype,
        use_bias=use_bias,
        **kwargs,
    )
    self.linear2 = nnx.Linear(
        dim_feedforward,
        d_model,
        rngs=rngs,
        dtype=dtype,
        use_bias=use_bias,
        **kwargs,
    )

    self.dtype = dtype

  def _ff_block(self, x: Array) -> Array:
    x = self.pre_ff_ln(x)
    if getattr(self, 'activation_name', None) == 'swiglu':
      x_gate = self.linear1_gate(x)
      x_val = self.linear1(x)
      x = self.activation(x_gate) * x_val
    else:
      x = self.linear1(x)
      x = self.activation(x)
    x = self.linear2(x)
    x = self.post_ff_ln(x)
    return x

  @jt.typed
  def __call__(
      self,
      q: jt.Float[jax.Array | np.ndarray, 'B T E'],
      k: Optional[jt.Float[jax.Array | np.ndarray, 'B T_src E']] = None,
      v: Optional[jt.Float[jax.Array | np.ndarray, 'B T_src E']] = None,
      attn_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B #N #T T_src']] = None,
      rope: Optional[RotaryEmbedding] = None,
      *,
      cached_kv: Optional[Tuple[jt.Float[jax.Array | np.ndarray, 'B T_src N D'],
                                jt.Float[jax.Array | np.ndarray, 'B T_src N D']]] = None,
      return_kv: bool = False,
  ) -> Union[jt.Float[jax.Array | np.ndarray, 'B T E'],   # output
             Tuple[jt.Float[jax.Array | np.ndarray, 'B T E'],  # output
                   Tuple[jt.Float[jax.Array | np.ndarray, 'B T_src N D'],  # key cache
                         jt.Float[jax.Array | np.ndarray, 'B T_src N D']]]  # value cache
             ]:
    """Forward pass through the attention block.

    Args:
      q: Query tensor.
      k: Key tensor.
      v: Value tensor.
      attn_mask: Attention mask.
      rope: Rotary positional embedding.
      cached_kv: Cached key/value tensors.
      return_kv: Whether to return updated KV cache.

    Returns:
      Output tensor or (output, new_kv).
    """
    x = q
    q_n = self.pre_attn_ln(q)
    if cached_kv is not None:
      assert k is None, f'k must be None when cached_kv is not None, {k=}'
      assert v is None, f'v must be None when cached_kv is not None, {v=}'
      k_n = None
      v_n = None
    else:
      k = q if k is None else k
      v = q if v is None else v
      k_n = self.pre_attn_ln(k)
      v_n = self.pre_attn_ln(v)

    # Pre-norm: norm -> sublayer -> residual
    attn_res = self.attn(
        query=q_n,
        key=k_n,
        value=v_n,
        attn_mask=attn_mask,
        rope=rope,
        cached_kv=cached_kv,
        return_kv=return_kv,
    )
    if return_kv:
      attn_out, new_kv = attn_res
    else:
      attn_out = attn_res

    attn_out = self.post_attn_ln(attn_out)

    x = x + attn_out
    ff_out = self._ff_block(x)
    x = x + ff_out
    if return_kv:
      return x, new_kv
    return x


class InducedSelfAttentionBlock(nnx.Module):
  """Induced Self-Attention for efficient O(n) attention.

  Flax/NNX implementation.

  Parameters
  ----------
  d_model, nhead, dim_feedforward, activation:
      Parameters for the underlying MultiheadAttentionBlocks.
  num_inds : int
      Number of learnable inducing points.

  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      num_inds: int,
      activation: Union[str, Callable] = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):

    self.num_inds = num_inds
    self.dtype = dtype

    # Learnable inducing points, initialized with truncated normal
    self.ind_vectors = nnx.Param(
        jax.random.truncated_normal(
            rngs.params(), -2, 2, (num_inds, d_model)
        ).astype(self.dtype)
        * 0.02
    )

    # Two-stage attention mechanism
    block_args = {
        'd_model': d_model,
        'nhead': nhead,
        'dim_feedforward': dim_feedforward,
        'activation': activation,
        'rngs': rngs,
        'attention_impl': attention_impl,
        'zero_out_proj_init': zero_out_proj_init,
        'use_bias': use_bias,
        'dtype': dtype,
    }
    self.mab1 = MultiheadAttentionBlock(**block_args)
    self.mab2 = MultiheadAttentionBlock(**block_args)

  @jt.typed
  def _induced_attention(
      self,
      src: jt.Float[jax.Array | np.ndarray, 'B T E'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      *,
      cached_inducing_repr: Optional[jt.Float[jax.Array | np.ndarray, 'B I E']] = None,
      return_inducing_repr: bool = False,
  ) -> Union[Array, Tuple[Array, Array]]:
    """Helper to run the two-stage attention with static masking."""
    *batch_shape, seq_len, d_model = src.shape

    hidden = None
    if cached_inducing_repr is not None:
      hidden = cached_inducing_repr
    else:
      # Broadcast inducing vectors to match batch shape
      ind_vectors = jnp.broadcast_to(
          self.ind_vectors[...], (*batch_shape, self.num_inds, d_model)
      )

      mask = None
      if train_size is not None:
        # train_size is (B,)
        # True means attend
        # mask shape: (B, seq_len)
        mask = jnp.arange(seq_len)[None, :] < train_size[:, None]
        # Reshape for attention: (B, 1, 1, seq_len) to broadcast over heads and query length (num_inds)
        mask = mask[:, None, None, :]

      # Stage 1: Input projects to inducing points.
      hidden = self.mab1(
          q=ind_vectors,
          k=src,
          v=src,
          attn_mask=mask,  # Pass boolean mask
      )

    # Stage 2: Inducing points project back to input (unchanged).
    out = self.mab2(
        q=src,
        k=hidden,
        v=hidden,
    )
    if return_inducing_repr:
      return typing.cast(Array, out), typing.cast(Array, hidden)
    return typing.cast(Array, out)

  @jt.typed
  def __call__(
      self,
      src: jt.Float[jax.Array | np.ndarray, 'B T E'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      *,
      cached_inducing_repr: Optional[jt.Float[jax.Array | np.ndarray, 'B I E']] = None,
      return_inducing_repr: bool = False,
  ) -> Union[jt.Float[jax.Array | np.ndarray, 'B T E'],
             Tuple[jt.Float[jax.Array | np.ndarray, 'B T E'],   # output
                   jt.Float[jax.Array | np.ndarray, 'B I E']    # cached_inducing_repr
                  ]
            ]:
    """Apply induced self-attention.

    Args:
      src: Input tensor.
      train_size: If provided, limits inducing points' attention.
      cached_inducing_repr: Cached inducing points representation.
      return_inducing_repr: Whether to return the inducing points representation.

    Returns:
      Output tensor or (output, inducing_repr).
    """
    return self._induced_attention(
        src,
        train_size,
        cached_inducing_repr=cached_inducing_repr,
        return_inducing_repr=return_inducing_repr,
    )


# Encoders


class Encoder(Module):
  """A stack of multi-head attention blocks.

  Attributes:
      num_blocks (int): Number of blocks in the encoder.
      d_model (int): The dimension of the model.
      nhead (int): The number of attention heads.
      dim_feedforward (int): The dimension of the feed-forward part of the
        block.
      activation (str): The activation function ('relu' or 'gelu').
      use_rope (bool): If True, use rotary positional embeddings.
      rope_base (float): The base for RoPE.
  """

  @jt.typed
  def __init__(
      self,
      num_blocks: int,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      activation: str = 'gelu',
      use_rope: bool = False,
      rope_base: float = 10000,
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Optional[Any] = None,
      dtype: DType = jnp.bfloat16,
      cache_icl_input_only: bool = False,
  ):
    if d_model % nhead != 0:
      raise ValueError(
          f'd_model ({d_model}) must be divisible by nhead ({nhead})'
      )

    activations = {'relu': jax.nn.relu, 'gelu': jax.nn.gelu, 'swiglu': 'swiglu'}
    if activation not in activations:
      raise ValueError(f'Activation must be one of {list(activations.keys())}')

    act_fn = activations[activation]

    @nnx.split_rngs(splits=num_blocks)
    @nnx.vmap(axis_size=num_blocks)
    def create_block(rngs):
      return MultiheadAttentionBlock(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          activation=act_fn,
          attention_impl=attention_impl,
          zero_out_proj_init=zero_out_proj_init,
          use_bias=use_bias,
          rngs=rngs,
          dtype=dtype,
      )

    self.blocks = create_block(rngs)
    self.dtype = dtype
    self.cache_icl_input_only = cache_icl_input_only
    self.rope = (
        RotaryEmbedding(
            dim=d_model // nhead, theta=rope_base, rngs=rngs, dtype=self.dtype
        )
        if use_rope
        else None
    )

  @jt.typed
  def __call__(
      self,
      src: jt.Float[jax.Array | np.ndarray, 'B T E'],
      attn_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B #N #T T_prefill']] = None,
      *,
      cached_kv: (
          jt.Float[jax.Array | np.ndarray, 'Y B T_prefill E'] |         # For cache_icl_input_only
          Tuple[jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D'],    # Key cache
                jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D']] |  # Value cache
          None
      )=None,
      return_kv: bool = False,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T E'] |
        Tuple[
          jt.Float[jax.Array | np.ndarray, 'B T E'],            # Output
          jt.Float[jax.Array | np.ndarray, 'Y B T_prefill E'] |         # For cache_icl_input_only
          Tuple[jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D'],    # Key cache
                jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D']]    # Value cache
        ]):
    """Forward pass through the stacked encoder blocks.

    Args:
      src: Input tensor.
      attn_mask: Optional attention mask.
      cached_kv: Optional cached KV tensors for decoding.
      return_kv: If True, returns the updated KV cache.

    Returns:
      Output tensor or (output, kvs).
    """
    if cached_kv is not None and return_kv:
      raise ValueError('Cannot both use cached_kv and return_kv in Encoder.')

    if cached_kv is not None:
      if self.cache_icl_input_only:
        # cached_kv is the stacked pre-block inputs: shape (num_layers, B, T, d_model).
        @nnx.scan(
            in_axes=(0, nnx.Carry, 0),  # block, carry, layer_input
            out_axes=nnx.Carry,
        )
        @nnx.remat
        def scan_fn_cached_input(
            block: MultiheadAttentionBlock,
            carry: jax.Array | np.ndarray,
            layer_input: jax.Array | np.ndarray,
        ):
          out = block(
              q=carry,
              k=layer_input,
              v=layer_input,
              attn_mask=attn_mask,
              rope=self.rope,
              cached_kv=None,
              return_kv=False,
          )
          return out

        final_out = scan_fn_cached_input(self.blocks, src, cached_kv)
        return final_out
      else:
        # cached_kv is a stack of per-layer (K, V) caches.
        @nnx.scan(
            in_axes=(0, nnx.Carry, 0),  # block, carry, cached_kv
            out_axes=nnx.Carry,
        )
        @nnx.remat
        def scan_fn_cached(
            block: MultiheadAttentionBlock, carry: jax.Array | np.ndarray, layer_kv
        ):
          out = block(
              q=carry,
              k=None,
              v=None,
              attn_mask=attn_mask,
              rope=self.rope,
              cached_kv=layer_kv,
              return_kv=False,
          )
          return out

        final_out = scan_fn_cached(self.blocks, src, cached_kv)
        return final_out

    if return_kv:
      if self.cache_icl_input_only:
        # Return per-layer inputs to the block.
        @nnx.scan(
            in_axes=(0, nnx.Carry),  # block, carry
            out_axes=(nnx.Carry, 0),  # carry, layer_input
        )
        @nnx.remat
        def scan_fn_return_input(
            block: MultiheadAttentionBlock, carry: jax.Array | np.ndarray
        ):
          out = block(
              q=carry,
              k=carry,
              v=carry,
              attn_mask=attn_mask,
              rope=self.rope,
              cached_kv=None,
              return_kv=False,
          )
          return out, carry  # carry is the input to this block

        final_out, kvs = scan_fn_return_input(self.blocks, src)
        return final_out, kvs
      else:
        @nnx.scan(
            in_axes=(0, nnx.Carry),  # block, carry
            out_axes=(nnx.Carry, 0),  # carry, layer_kv
        )
        @nnx.remat
        def scan_fn_return_kv(block: MultiheadAttentionBlock, carry: jax.Array | np.ndarray):
          out, new_kv = block(
              q=carry,
              k=carry,
              v=carry,
              attn_mask=attn_mask,
              rope=self.rope,
              cached_kv=None,
              return_kv=True,
          )
          return out, new_kv

        final_out, kvs = scan_fn_return_kv(self.blocks, src)
        return final_out, kvs

    @nnx.scan(
        in_axes=(0, nnx.Carry),  # block, carry
        out_axes=nnx.Carry,
    )
    @nnx.remat
    def scan_fn(block: MultiheadAttentionBlock, carry: jax.Array | np.ndarray):
      out = block(
          q=carry,
          k=carry,
          v=carry,
          attn_mask=attn_mask,
          rope=self.rope,
      )
      return out

    final_out = scan_fn(self.blocks, src)
    return final_out


class SetTransformer(Module):
  """A stack of induced self-attention blocks for permutation-invariant processing.

  Attributes:
      num_blocks (int): The number of induced self-attention blocks.
      d_model (int): The dimension of the model.
      nhead (int): The number of attention heads.
      dim_feedforward (int): The dimension of the feed-forward network.
      num_inds (int): The number of inducing points.
      activation (str): The activation function ('relu' or 'gelu').
  """

  @jt.typed
  def __init__(
      self,
      num_blocks: int,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      num_inds: int = 13,
      activation: str = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Optional[Any] = None,
      dtype: DType = jnp.bfloat16,
  ):
    self.dtype = dtype
    if d_model % nhead != 0:
      raise ValueError(
          f'd_model ({d_model}) must be divisible by nhead ({nhead})'
      )

    activations = {'relu': jax.nn.relu, 'gelu': jax.nn.gelu, 'swiglu': 'swiglu'}
    if activation not in activations:
      raise ValueError(f'Activation must be one of {list(activations.keys())}')

    act_fn = activations[activation]
    @nnx.split_rngs(splits=num_blocks)
    @nnx.vmap(axis_size=num_blocks)
    def create_block(rngs):
      return InducedSelfAttentionBlock(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          num_inds=num_inds,
          activation=act_fn,
          attention_impl=attention_impl,
          zero_out_proj_init=zero_out_proj_init,
          rngs=rngs,
          use_bias=use_bias,
          dtype=dtype,
      )

    self.blocks = create_block(rngs)

  @jt.typed
  def __call__(
      self,
      src: jt.Float[jax.Array | np.ndarray, 'B T E'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      *,
      cached_inducing_repr: Optional[jt.Float[jax.Array | np.ndarray, 'Y B I E']] = None,
      return_inducing_repr: bool = False,
  ) -> Union[jt.Float[jax.Array | np.ndarray, 'B T E'],    # Output
             # For cached_inducing_repr:
             Tuple[
                 # Output
                 jt.Float[jax.Array | np.ndarray, 'B T E'],
                 # Cache
                 jt.Float[jax.Array | np.ndarray, 'Y B I E']]]:
    """Applies the Set Transformer to the input.

    Args:
      src: Input tensor.
      train_size: Optional training size for masking.
      cached_inducing_repr: Optional cached inducing points representation.
      return_inducing_repr: If True, returns the inducing points representation.

    Returns:
      Output tensor or (output, inducing_repr).
    """
    if cached_inducing_repr is not None and return_inducing_repr:
      raise ValueError(
          'Cannot both use cached_inducing_repr and return_inducing_repr.'
      )

    if cached_inducing_repr is not None:

      @nnx.scan(
          in_axes=(0, nnx.Carry, 0),
          out_axes=nnx.Carry,
      )
      @nnx.remat
      def scan_fn_cached(
          block: InducedSelfAttentionBlock, carry: jax.Array | np.ndarray, layer_repr
      ):
        out = block(
            carry,
            train_size=train_size,
            cached_inducing_repr=layer_repr,
        )
        return out

      final_out = scan_fn_cached(self.blocks, src, cached_inducing_repr)
      return final_out

    elif return_inducing_repr:

      @nnx.scan(
          in_axes=(0, nnx.Carry),
          out_axes=(nnx.Carry, 0),
      )
      @nnx.remat
      def scan_fn_return(block: InducedSelfAttentionBlock, carry: jax.Array | np.ndarray):
        out, repr = block(
            carry,
            train_size=train_size,
            return_inducing_repr=True,
        )
        return out, repr

      final_out, stacked_repr = scan_fn_return(self.blocks, src)
      return final_out, stacked_repr

    else:

      @nnx.scan(
          in_axes=(0, nnx.Carry),  # block, carry, rngs
          out_axes=nnx.Carry,
      )
      @nnx.remat
      def scan_fn(block: InducedSelfAttentionBlock, carry: jax.Array | np.ndarray):
        out = block(
            carry, train_size=train_size
        )
        return out

      final_out = scan_fn(self.blocks, src)
      return final_out


# Column-wise embedding


class YEmbeddingScheme(enum.Enum):
  """Defines how target values (y) are embedded and combined with features (X)."""

  NONE = 'none'
  """y is not used in the cell embedding stage."""

  ADD_Y_TO_X_POST_EMBEDDING = 'add_y_to_x_post_embedding'
  """y is embedded to a vector of size embed_dim and added to the cell embedding *after* the cell embedding projection."""


class CellEmbedder(nnx.Module):
  """Embeds individual cells.

  Parameters
  ----------
  embed_dim : int
      Embedding dimension.
  y_embedding_scheme : YEmbeddingScheme
      The scheme to use for embedding y values.
  max_classes : int
      Number of classes for y embedding lookup table.
  y_col_embedder_encoder_nhid : int
      Number of hidden units for the y encoder MLP.
  is_classifier : bool
      Whether the model is a classifier.
  feature_group : bool, default=False
      Whether to group features with overlap using shifts (equivalent to 'same').
  feature_group_size : int, default=3
      The number of features in each group when feature_group is True.
  fourier_features_num_frequencies : int, default=32
      Number of frequencies for Fourier features.
  fourier_features_sigma : float, default=1.0
      Sigma for initializing Fourier feature frequencies.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  dtype : DType
      Data type for calculations.
  """

  @jt.typed
  def __init__(
      self,
      embed_dim: int,
      y_embedding_scheme: YEmbeddingScheme,
      max_classes: int,
      y_col_embedder_encoder_nhid: int,
      *,
      is_classifier: bool,
      feature_group: Union[bool, str] = False,
      feature_group_size: int = 3,
      use_fourier_features: bool = True,
      fourier_features_num_frequencies: int = 32,
      fourier_features_sigma: float = 1.0,
      rngs: Optional[nnx.Rngs] = None,
      dtype: DType = jnp.bfloat16,
  ) -> None:
    self.embed_dim = embed_dim
    self.dtype = dtype
    self.y_embedding_scheme = y_embedding_scheme
    if isinstance(feature_group, str):
      if feature_group.lower() in ['none', 'false', '']:
        feature_group = False
      elif feature_group.lower() in ['same', 'true']:
        feature_group = True
      else:
        raise ValueError(f'Invalid feature_group string: {feature_group}')

    self.feature_group = feature_group
    self.feature_group_size = feature_group_size
    self.use_fourier_features = use_fourier_features
    self.fourier_features_num_frequencies = fourier_features_num_frequencies
    self.fourier_features_sigma = fourier_features_sigma

    self.is_classifier = is_classifier

    in_dim = feature_group_size if self.feature_group else 1

    if self.use_fourier_features:
      # --- New path: Fourier feature projection ---
      # Projects raw input values through learned frequency banks before the
      # linear embedding, giving the model a richer spectral representation.
      linear_in_dim = self.fourier_features_num_frequencies * 2
      if rngs is not None:
        freq_key = rngs.params()
      else:
        freq_key = jax.random.key(42)
      self.fourier_frequencies = nnx.Param(
          jax.random.normal(
              freq_key, (in_dim, self.fourier_features_num_frequencies)
          )
          * self.fourier_features_sigma
      )
      # Separate frequency bank for categorical features so num and cat paths
      # can independently optimize their frequency scales without gradient
      # conflict. Numerics want metric-preserving (moderate ω); cats want
      # high ω to decorrelate adjacent integer-encoded category indices.
      if rngs is not None:
        freq_key_cat = rngs.params()
      else:
        freq_key_cat = jax.random.key(43)
      self.fourier_frequencies_cat = nnx.Param(
          jax.random.normal(
              freq_key_cat, (in_dim, self.fourier_features_num_frequencies)
          )
          * self.fourier_features_sigma
      )
      # Numerical input projection
      self.in_linear = nnx.Linear(
          linear_in_dim,
          embed_dim,
          rngs=rngs,
          dtype=self.dtype,
          bias_init=nnx.initializers.normal(stddev=1e-6),
      )
      # Separate categorical projection: each feature type gets its own learned
      # linear head, giving strictly more expressive power than a shared proj.
      self.in_linear_cat = nnx.Linear(
          linear_in_dim,
          embed_dim,
          rngs=rngs,
          dtype=self.dtype,
          bias_init=nnx.initializers.normal(stddev=1e-6),
      )
    else:
      # --- Old (legacy) path: direct linear projection ---
      # Used when loading checkpoints trained before Fourier features were
      # introduced. Produces the same parameter tree as the old model:
      #   in_linear: (in_dim -> embed_dim)
      # No fourier_frequencies, fourier_frequencies_cat, or in_linear_cat.
      self.in_linear = nnx.Linear(
          in_dim,
          embed_dim,
          rngs=rngs,
          dtype=self.dtype,
          bias_init=nnx.initializers.normal(stddev=1e-6),
      )

    if self.y_embedding_scheme == YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING:
      if self.is_classifier:
        self.y_embedder_lookup = nnx.Embed(
            num_embeddings=max_classes,
            features=embed_dim,
            rngs=rngs,
            dtype=self.dtype,
        )
      else:
        self.y_embedder_lookup = MLP(
            in_dim=1,
            out_dim=embed_dim,
            hidden_dims=(y_col_embedder_encoder_nhid,),
            activation='gelu',
            rngs=rngs,
            dtype=self.dtype,
        )

  @jt.typed
  def feature_grouping(
      self,
      X: jt.Shaped[jax.Array | np.ndarray, 'B T H'],
      d: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None
  ) -> jt.Shaped[jax.Array | np.ndarray, 'B T H G']:
    """Groups features with overlap using shifts.

    Args:
      X: Input feature tensor.
      d: The number of active features for each batch element.

    Returns:
      Grouped feature tensor.
    """
    if not self.feature_group:
      return jnp.expand_dims(X, -1)

    B, T, H = X.shape
    size = self.feature_group_size

    idxs = jnp.arange(H, dtype=jnp.int32)
    stacked = []
    for i in range(size):
      # i=0 -> offset=0, i=1 -> offset=1, i=2 -> offset=3, i=3 -> offset=7 ...
      offset = (2**i) - 1
      if d is not None:
        # Guard against mod-by-zero: XLA silently returns x for x%0.
        d_safe = jnp.maximum(d, 1)
        idx = (idxs[None, None, :] + offset) % d_safe[:, None, None]
        # Extract dynamic indices across the batch gracefully
        stacked.append(jnp.take_along_axis(X, idx, axis=-1))
      else:
        idx = (idxs[None, None, :] + offset) % H
        stacked.append(jnp.take_along_axis(X, idx, axis=-1))
    return jnp.stack(stacked, axis=-1)

  @jt.typed
  def __call__(
      self,
      features: jt.Float[jax.Array | np.ndarray, 'B T H'],
      y: jt.Shaped[Array, 'B T'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      d: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      cat_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B H']] = None,
  ) -> jt.Float[jax.Array | np.ndarray, 'B T H E']:
    """Embeds the input features and target values.

    Args:
      features: Input feature tensor.
      y: Target value tensor.
      train_size: The number of training samples for each batch element.
      d: The number of active features for each batch element.
      cat_mask: Boolean mask indicating categorical features.

    Returns:
      Cell embeddings.
    """
    B, T, H = features.shape
    E = self.embed_dim

    features_grouped = self.feature_grouping(features, d=d)

    HC = features_grouped.shape[-2]

    if self.use_fourier_features:
      # --- New path: Fourier feature projection ---
      features_expanded_raw = features_grouped  # (B, T, HC, in_dim)

      # features_expanded shape: (B, T, HC, in_dim)
      # fourier_frequencies shape: (in_dim, num_frequencies)
      x_proj = jnp.einsum(
          '...i,if->...if', features_expanded_raw, self.fourier_frequencies.value
      )
      # Always keep fourier_feats in per-slot shape: (B, T, HC, G, num_freq*2).
      # When feature_group=False, feature_grouping returns (B, T, HC, 1), so G=1
      # (a singleton). Summing over axis=-2 later collapses it correctly in both
      # cases, removing the need to branch on self.feature_group here.
      features_expanded = jnp.concatenate(
          [jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1
      )  # (B, T, HC, G, num_freq*2)

      # Route through numerical or categorical projection per slot, then sum over G.
      num_out = self.in_linear(features_expanded)  # (B, T, HC, G, E)

      if cat_mask is not None:
        # Group cat_mask exactly like features (X) were grouped.
        # Expand to (B, 1, HC) so feature_grouping sees shape (B, T, HC).
        cat_mask_grouped = self.feature_grouping(cat_mask[:, None, :], d=d)  # (B, 1, HC, G)
        # Broadcast to (B, 1, HC, G, 1) to match (B, T, HC, G, E)
        cat_mask_per_slot = cat_mask_grouped[:, :, :, :, None]

        # Cat-specific Fourier bank: project raw values through ω_cat.
        x_proj_cat = jnp.einsum(
            '...i,if->...if',
            features_expanded_raw,
            self.fourier_frequencies_cat.value,
        )  # (B, T, HC, G, num_freq)
        fe_cat_fourier = jnp.concatenate(
            [jnp.sin(x_proj_cat), jnp.cos(x_proj_cat)], axis=-1
        )  # (B, T, HC, G, num_freq*2)
        cat_out = self.in_linear_cat(fe_cat_fourier)  # (B, T, HC, G, E)

        selected = jnp.where(cat_mask_per_slot, cat_out, num_out)
        cell_embeddings = selected.sum(axis=-2)  # (B, T, HC, E)
      else:
        # No cat_mask: treat all as numerical and sum across slots (or singleton).
        cell_embeddings = num_out.sum(axis=-2)  # (B, T, HC, E)
    else:
      # --- Old (legacy) path: direct linear projection ---
      # Matches the parameter tree of checkpoints trained before Fourier
      # features were introduced. in_linear projects (in_dim -> embed_dim)
      # directly from raw scalar/grouped values.
      cell_embeddings = self.in_linear(features_grouped)  # (B, T, HC, E)

    if self.y_embedding_scheme == YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING:
      if self.is_classifier:
        # Embed y using a lookup table.
        y_embedded: Float[Array, '... T E'] = self.y_embedder_lookup(
            y.astype(jnp.int32)
        )
      else:
        # Embed y as a continuous feature.
        # TODO: Try using Fourier features for the continuous y embedding (as
        # done for X in feature_grouping) instead of a plain MLP, so the model
        # gets the same frequency-rich representation for target values.
        y_embedded: Float[Array, '... T E'] = self.y_embedder_lookup(y[..., None])

      if train_size is not None:
        # Create a mask for the training samples.
        train_mask = jnp.arange(T)[None, :] < train_size[:, None]  # (B, T)
        train_mask = train_mask[..., None, None]  # (B, T, 1, 1)
        # Add the embedded y only to the training samples.
        cell_embeddings = jnp.where(
            train_mask,
            cell_embeddings + y_embedded[:, :, None, :],
            cell_embeddings,
        )
      else:
        # If train_size is not provided, add to all samples.
        cell_embeddings = cell_embeddings + y_embedded[:, :, None, :]

    if d is not None:
      # (B, 1, HC, 1)
      mask = jnp.arange(HC)[None, None, :, None] < d[:, None, None, None]
      cell_embeddings = jnp.where(mask, cell_embeddings, 0.0)

    chex.assert_shape(cell_embeddings, (B, T, HC, E))
    return cell_embeddings


class ColEmbedding(nnx.Module):
  """Distribution-aware column-wise embedding in Flax.

  This module maps each scalar cell in a column to a high-dimensional embedding
  while capturing statistical regularities within the column. Unlike traditional
  approaches that use separate embedding layers per column, it employs a shared
  set transformer to process all features.

  ColEmbedding operates as follows:
  1. Each scalar cell is first linearly projected into the embedding dimension.
  2. The set transformer generates distribution-aware weights and biases for
  each column.
  3. The final column embeddings are computed as: column * weights + biases.

  Parameters
  ----------
  embed_dim : int
      Embedding dimension.
  num_blocks : int
      Number of induced self-attention blocks in the set transformer.
  nhead : int
      Number of attention heads of the set transformer.
  dim_feedforward : int
      Dimension of the feedforward network of the set transformer.
  num_inds : int
      Number of inducing points used in self-attention blocks of the set
      transformer.
  activation : str or callable, default="gelu"
      The activation function used in the feedforward network, can be
      either a string ("relu" or "gelu") or a unary callable.
  reserve_cls_tokens : int, default=4
      Number of slots to reserve for CLS tokens to avoid concatenation.
  rngs : nnx.Rngs, optional
      The random number generators for the module.
  """

  @jt.typed
  def __init__(
      self,
      embed_dim: int,
      num_blocks: int,
      nhead: int,
      dim_feedforward: int,
      num_inds: int,
      activation: str | Callable = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Optional[nnx.Rngs] = None,
      dtype: DType = jnp.bfloat16,
  ) -> None:
    self.embed_dim = embed_dim
    self.dtype = dtype

    self.tf_col = SetTransformer(
        num_blocks=num_blocks,
        d_model=embed_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_inds=num_inds,
        activation=activation,
        rngs=rngs,
        dtype=self.dtype,
        attention_impl=attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )

    self.out_w = nnx.Linear(
        embed_dim,
        embed_dim,
        rngs=rngs,
        dtype=self.dtype,
        bias_init=nnx.initializers.normal(stddev=1e-6),
    )
    self.ln_w = nnx.RMSNorm(embed_dim, rngs=rngs, dtype=self.dtype)

  @jt.typed
  def __call__(
      self,
      X: jt.Float[jax.Array | np.ndarray, 'B T H E'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      feature_shuffles: Optional[List[List[int]]] = None,
      *,
      cached_repr: Optional[jt.Float[jax.Array | np.ndarray, 'Y B*H I E']] = None,
      return_repr: bool = False,
  ) -> Union[jt.Float[jax.Array | np.ndarray, 'B T H E'],  # Output
             Tuple[jt.Float[jax.Array | np.ndarray, 'B T H E'], # Output
                   jt.Float[jax.Array | np.ndarray, 'Y B*H I E']  # cached_repr
                  ]]:
    """Transform input table into embeddings.

    Args:
      X: Input tensor.
      train_size: Position to split the input into training and test data.
      feature_shuffles: A list of feature shuffle patterns for each table.
      cached_repr: Transformer attention cache.
      return_repr: If True, return the new representations computed in this call.

    Returns:
      Embeddings or (embeddings, new_repr).
    """
    assert not (cached_repr is not None and return_repr), (
        'Cannot have both cached_repr not None and return_repr set to True.'
    )
    B, T, HC, E = X.shape
    padded_train_size = T
    if cached_repr is not None:
      assert train_size is None, (
          'train_size must be None when cached_repr is not None.'
      )

    assert padded_train_size is not None
    train_size_expanded = None
    if train_size is not None:
      if train_size.ndim == 2:
        train_size = jnp.squeeze(train_size, axis=-1)
      chex.assert_shape(train_size, (B,))
      chex.assert_shape(padded_train_size, ())
      # Expand train_size to match the flattened batch dimension (B * HC)
      # Repeat for each feature column: (B,) -> (B * HC,)
      train_size_expanded = jnp.repeat(train_size, HC)

    X = X.transpose((0, 2, 1, 3))
    src = X.reshape(B * HC, T, E)

    representations: Array
    new_repr: Optional[Array] = None

    tf_col_st = typing.cast(SetTransformer, self.tf_col)
    if cached_repr is not None:    # Decode.
      representations = typing.cast(
          Array,
          tf_col_st(
              src,
              # train_size is not used in SetTransformer during decode.
              train_size=None,
              cached_inducing_repr=cached_repr,
          ),
      )
    elif return_repr:             # Prefill.
      representations, new_repr = tf_col_st(
          src,
          train_size=train_size_expanded,
          return_inducing_repr=True,
      )
    else:                        # Train.
      representations = typing.cast(
          Array,
          tf_col_st(
              src,
              train_size=train_size_expanded,
          ),
      )

    # 3. Output Projection & Normalization
    embeddings = self.ln_w(self.out_w(representations))  # (B * HC, T, E)

    # Reshape back to original table structure
    # (B * HC, T, E) -> (B, HC, T, E)
    embeddings = embeddings.reshape((B, HC, T, self.embed_dim))

    # Transpose to (B, T, HC, E)
    final_embeddings = embeddings.transpose((0, 2, 1, 3))

    if return_repr:
      return final_embeddings, typing.cast(Array, new_repr)
    return final_embeddings

# Row-wise embedding

class RowInteraction(nnx.Module):
  """Context-aware row-wise interaction, rewritten in Flax NNX.

  This module captures interactions between features within each row using a
  transformer
  encoder. It prepends learnable class tokens to the feature embeddings and uses
  these tokens to aggregate information.
  """

  @jt.typed
  def __init__(
      self,
      embed_dim: int,
      num_blocks: int,
      nhead: int,
      dim_feedforward: int,
      num_cls: int = 4,
      rope_base: float = 100000,
      activation: str | Callable = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      output_full_sequence: bool = False,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.embed_dim = embed_dim
    self.num_cls = num_cls
    self.dtype = dtype
    self.output_full_sequence = output_full_sequence

    self.tf_row = Encoder(
        num_blocks=num_blocks,
        d_model=embed_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        activation=activation,
        use_rope=True,
        rope_base=rope_base,
        attention_impl=attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
        rngs=rngs,
        dtype=self.dtype,
    )

    self.out_ln = nnx.RMSNorm(embed_dim, rngs=rngs, dtype=self.dtype)

  @jt.typed
  def __call__(
      self,
      embeddings: jt.Float[jax.Array | np.ndarray, 'B T H E'],
      d: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T H E']
        | jt.Float[jax.Array | np.ndarray, 'B T C_TIMES_E']): # if output_full_sequence is False.
    """Captures interactions between features within each row.

    Args:
      embeddings: Input feature embeddings.
      d: The number of active features for each batch element.

    Returns:
      Context-aware representations.
    """
    B, T, HC, E = embeddings.shape
    embeddings_reshaped = embeddings.reshape(B * T, HC, E)

    # Create mask to prevent attention to padding features if `d` is provided
    attn_mask = None
    if d is not None:
      if d.ndim == 2:
        d = jnp.squeeze(d, axis=-1)
      # `d` is (B,), representing number of features for each table in the batch
      d_padded = d + self.num_cls
      indices = jnp.arange(HC)
      # Mask is True for positions to be attended to. Shape: (B, HC)
      mask_per_table = indices < d_padded[:, None]
      # Expand and reshape to match the transformer input: (B*T, HC)
      attn_mask = jnp.repeat(mask_per_table, T, axis=0)
      # Expand to (B*T, 1, HC) for attention mask
      attn_mask = attn_mask[:, None, None, :]

    # Process through the transformer
    outputs = self.tf_row(
        embeddings_reshaped,
        attn_mask=attn_mask,
    )
    if self.output_full_sequence:
      outputs = self.out_ln(outputs)

      # Reshape flattened symbols back to (B, T, HC, E)
      representations = outputs.reshape(B, T, HC, -1)
    else:
      # Extract, normalize, and flatten CLS token outputs
      cls_outputs = outputs[:, : self.num_cls, :]
      cls_outputs = self.out_ln(cls_outputs)

      # Reshape flattened CLS outputs back to (B, T, C*E)
      representations = cls_outputs.reshape(B, T, -1)

    return representations


@nnx.dataclass
class ICLearningCache:
  layer_caches: (
      jt.Float[jax.Array | np.ndarray, 'Y B T_prefill E']  # For cache_icl_input_only
      | Tuple[
          jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D'],  # Key cache
          jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D'],  # Value cache
      ]
  )
  prefill_train_size: jt.Int[jax.Array | np.ndarray, 'B']


class ICLearning(nnx.Module):
  """Dataset-wise in-context learning.

  This module is rewritten in Flax NNX and implements in-context learning that:
  1. Takes row representations and training labels as input.
  2. Conditions the model on training examples.
  3. Makes predictions for test examples based on learned patterns.

  parameters
  ----------
  max_classes : int
      Number of classes that the model supports.
  d_model : int
      Model dimension.
  num_blocks : int
      Number of blocks used in the ICL encoder.
  nhead : int
      Number of attention heads of the ICL encoder.
  dim_feedforward : int
      Dimension of the feedforward network of the ICL encoder.
  activation : str, default="gelu"
      The activation function used in the feedforward network.
  rngs : nnx.Rngs, optional
      RNGs for initialization.
  """
  @jt.typed
  def __init__(
      self,
      loss: str,
      max_classes: int,
      d_model: int,
      num_blocks: int,
      nhead: int,
      dim_feedforward: int,
      activation: str = 'gelu',
      zero_out_proj_init: bool = False,
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      use_bias: bool = True,
      cache_icl_input_only: bool = False,
      *,
      rngs: Any | None = None,
      dtype: DType = jnp.bfloat16,
  ):
    super().__init__()
    self.attention_impl = attention_impl
    self.max_classes = max_classes
    self.dtype = dtype
    self.loss = loss
    self.is_classifier = self.loss == 'cross_entropy'
    self.cache_icl_input_only = cache_icl_input_only

    self.tf_icl = Encoder(
        num_blocks=num_blocks,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        activation=activation,
        use_rope=False,
        attention_impl=attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
        cache_icl_input_only=cache_icl_input_only,
        rngs=rngs,
        dtype=self.dtype,
    )
    self.ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=self.dtype)
    if self.is_classifier:
      self.y_encoder = OneHotAndLinear(
          max_classes, d_model, rngs=rngs, dtype=self.dtype
      )
      self.decoder = MLP(
          in_dim=d_model,
          out_dim=max_classes,
          hidden_dims=(d_model * 2,),
          rngs=rngs,
          dtype=self.dtype,
      )
    else:
      self.y_encoder = MLP(
          in_dim=1,
          out_dim=d_model,
          hidden_dims=(d_model * 2,),
          rngs=rngs,
          dtype=self.dtype,
      )
      if self.loss == 'rmse' or self.loss == 'mse':
        out_dim = 1
      else:
        raise ValueError(f'Unsupported regression loss: {self.loss}')
      self.decoder = MLP(
          in_dim=d_model,
          out_dim=out_dim,
          hidden_dims=(d_model * 2,),
          rngs=rngs,
          dtype=self.dtype,
      )

  @jt.typed
  def _prefill_sequence_length_from_cache(
      self, cache: ICLearningCache
  ) -> Any:
    """Returns the sequence length of the prefill cache."""
    assert cache.layer_caches is not None, 'Layer caches must be non-empty.'
    if self.cache_icl_input_only:
      # Shape (Y, B, T, E)
      return cache.layer_caches.shape[2]
    else:
      # Tuple of (K, V), each shape (Y, B, T, N, D)
      k, _ = cache.layer_caches
      return k.shape[2]

  @jt.typed
  def __call__(
      self,
      R: jt.Float[jax.Array | np.ndarray, 'B T E'],
      y: jt.Shaped[jax.Array | np.ndarray, 'B T'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']],
      *,
      cache: Optional[ICLearningCache] = None,
      return_cache: bool = False,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T K'] | jt.Float[jax.Array | np.ndarray, 'B T E']
        | Tuple[
            jt.Float[jax.Array | np.ndarray, 'B T K'] | jt.Float[jax.Array | np.ndarray, 'B T E'],
            ICLearningCache
          ]):
    """Forward pass for ICLearning."""
    is_prefill = return_cache
    is_decode = cache is not None
    is_train = not is_prefill and not is_decode
    assert int(is_prefill) + int(is_decode) + int(is_train) == 1, (
        'Exactly one of is_prefill, is_decode, and is_train can be True.'
        + f' Got {is_prefill=}, {is_decode=}, {is_train=}.'
    )

    y = y.astype(self.dtype)
    if self.is_classifier:
      y_encoded = self.y_encoder(y.astype(jnp.int32))
    else:
      y_encoded = self.y_encoder(y[..., None])

    B, sequence_length, _ = R.shape

    if is_train or is_prefill:
      assert train_size is not None
      if train_size.ndim == 2:
        train_size = jnp.squeeze(train_size, axis=-1)
      train_mask = (
          jnp.arange(sequence_length)[None, :] < train_size[:, None]
      )
      full_attn_mask = train_mask[:, None, None, :]
      y_encoded = y_encoded * train_mask.reshape(B, sequence_length, 1)
      R = R + y_encoded
    else:
      assert is_decode
      prefill_train_size = cache.prefill_train_size
      prefill_sequence_length = self._prefill_sequence_length_from_cache(cache)
      train_mask = (
          jnp.arange(prefill_sequence_length)[None, :] < prefill_train_size[:, None]
      )
      full_attn_mask = train_mask[:, None, None, :]

    new_cache = None
    if is_train:
      result = self.tf_icl(R, attn_mask=full_attn_mask)
    elif is_prefill:
      result, new_layer_caches = self.tf_icl(
          R, attn_mask=full_attn_mask, return_kv=True
      )
      new_cache = ICLearningCache(
          layer_caches=new_layer_caches,
          prefill_train_size=train_size
      )
    else:
      assert is_decode
      assert train_size is None
      result = self.tf_icl(
          R, attn_mask=full_attn_mask, cached_kv=cache.layer_caches
      )



    result = self.ln(result)
    result = self.decoder(result)
    if return_cache:
      assert new_cache is not None
      return result, new_cache
    return result

# TabFM

class TabFM(nnx.Module):
  """A Tabular Foundation Model (TabFM), rewritten in Flax NNX.

  TabFM is a transformer-based architecture for in-context learning on tabular
  data to make
  predictions without fine-tuning. It processes tabular data through three
  sequential stages:

  1. Column-wise embedding creates distribution-aware embeddings.
  2. Row-wise interaction captures interactions between features within each
  row.
  3. Dataset-wise in-context learning learns patterns from labeled examples and
  makes predictions.

  Datasets with more than `max_classes` classes are not supported.

  Parameters
  ----------
  max_classes : int, default=10
      Number of classes that the model supports.
  embed_dim : int, default=128
      Model dimension used in the column/row embedding transformers. For the
      in-context
      learning transformer, the dimension is this value multiplied by the number
      of CLS tokens.
  col_num_blocks : int, default=3
      Number of induced self-attention blocks in the column embedding
      transformer.
  col_nhead : int, default=4
      Number of attention heads in the column embedding transformer.
  col_num_inds : int, default=128
      Number of inducing points in the column embedding transformer.
  row_num_blocks : int, default=3
      Number of attention blocks in the row interaction transformer.
  row_nhead : int, default=8
      Number of attention heads in the row interaction transformer.
  row_num_cls : int, default=4
      Number of learnable CLS tokens used to aggregate feature information per
      row.
  row_rope_base : float, default=100000
      Base scaling factor for rotary position encoding in the row interaction
      transformer.
  icl_num_blocks : int, default=12
      Number of transformer blocks in the in-context learning transformer.
  icl_nhead : int, default=4
      Number of attention heads in the in-context learning transformer.
  ff_factor : int, default=2
      Expansion factor for feedforward networks across all components.
  activation : str or callable, default="gelu"
      Activation function used throughout the model.
  y_embedding_scheme : YEmbeddingScheme, default=YEmbeddingScheme.NONE
      The scheme to use for embedding y values.
  y_col_embedder_encoder_nhid: int, default=6
      Number of hidden units for the encoder in the y
      column embedding MLP. Only used when y_embedding_scheme is
      ADD_Y_TO_X_POST_EMBEDDING with a regression target.
  feature_group : bool, default=False
      Whether to group features for cell embedding. If 'none', features are
      embedded independently. If 'same', groups features with overlap using
      shifts.
  feature_group_size : int, default=3
      The number of features in each group when feature_group is not False.
  col_attention_impl : AttentionImplementation, default=AttentionImplementation.JAX
      Which attention implementation to use for column embedding.
  row_attention_impl : AttentionImplementation, default=AttentionImplementation.JAX
      Which attention implementation to use for row interaction.
  icl_attention_impl : AttentionImplementation, default=AttentionImplementation.JAX
      Which attention implementation to use for attention layers in ICL.
  use_fourier_features : bool, default=True
      If True (default), CellEmbedder uses learned Fourier frequency banks to
      project raw feature values before the linear embedding. Set to False when
      loading checkpoints trained before this feature was introduced; doing so
      reconstructs the original parameter tree (single ``in_linear`` mapping
      ``in_dim -> embed_dim`` directly), allowing Orbax to restore the state
      without key-shape mismatches.
  rngs : nnx.Rngs
      The random number generators for initialization.
  """

  @jt.typed
  def __init__(
      self,
      loss: str = 'cross_entropy',
      max_classes: int = 10,
      embed_dim: int = 128,
      col_num_blocks: int = 3,
      col_nhead: int = 4,
      col_num_inds: int = 128,
      row_num_blocks: int = 3,
      row_nhead: int = 8,
      row_num_cls: int = 4,
      row_rope_base: float = 100000,
      icl_num_blocks: int = 12,
      icl_nhead: int = 4,
      col_attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      row_attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      icl_attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      ff_factor: int = 2,
      activation: str | Callable = 'gelu',
      y_embedding_scheme: YEmbeddingScheme = YEmbeddingScheme.NONE,
      y_col_embedder_encoder_nhid: int = 6,
      cache_icl_input_only: bool = False,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      feature_group: Union[bool, str] = False,
      feature_group_size: int = 3,
      use_fourier_features: bool = True,
      fourier_features_num_frequencies: int = 32,
      fourier_features_sigma: float = 1.0,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    super().__init__()
    self.max_classes = max_classes
    self.embed_dim = embed_dim
    self.row_num_cls = row_num_cls
    self.dtype = dtype
    self.loss = loss
    self.is_classifier = self.loss == 'cross_entropy'

    self.cls_tokens = nnx.Param(
        nnx.initializers.truncated_normal(stddev=0.02)(
            rngs.params(), (row_num_cls, embed_dim)
        ).astype(self.dtype)
    )

    # Stage 1: Column-wise Embedding
    self.cell_embedder = CellEmbedder(
        embed_dim=embed_dim,
        y_embedding_scheme=y_embedding_scheme,
        max_classes=max_classes,
        y_col_embedder_encoder_nhid=y_col_embedder_encoder_nhid,
        is_classifier=self.is_classifier,
        feature_group=feature_group,
        feature_group_size=feature_group_size,
        use_fourier_features=use_fourier_features,
        fourier_features_num_frequencies=fourier_features_num_frequencies,
        fourier_features_sigma=fourier_features_sigma,
        rngs=rngs,
        dtype=self.dtype,
    )
    self.col_embedder = ColEmbedding(
        embed_dim=embed_dim,
        num_blocks=col_num_blocks,
        nhead=col_nhead,
        num_inds=col_num_inds,
        dim_feedforward=embed_dim * ff_factor,
        activation=activation,
        rngs=rngs,
        dtype=self.dtype,
        attention_impl=col_attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )
    self.col_embedder_2 = ColEmbedding(
        embed_dim=embed_dim,
        num_blocks=col_num_blocks,
        nhead=col_nhead,
        num_inds=col_num_inds,
        dim_feedforward=embed_dim * ff_factor,
        activation=activation,
        rngs=rngs,
        dtype=self.dtype,
        attention_impl=col_attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )

    # Stage 2: Row-wise Interaction
    self.row_interactor = RowInteraction(
        embed_dim=embed_dim,
        num_blocks=row_num_blocks,
        nhead=row_nhead,
        num_cls=row_num_cls,
        rope_base=row_rope_base,
        dim_feedforward=embed_dim * ff_factor,
        activation=activation,
        attention_impl=row_attention_impl,
        rngs=rngs,
        dtype=self.dtype,
        output_full_sequence=True,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )
    self.row_interactor_2 = RowInteraction(
        embed_dim=embed_dim,
        num_blocks=row_num_blocks,
        nhead=row_nhead,
        num_cls=row_num_cls,
        rope_base=row_rope_base,
        dim_feedforward=embed_dim * ff_factor,
        activation=activation,
        attention_impl=row_attention_impl,
        rngs=rngs,
        dtype=self.dtype,
        output_full_sequence=False,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )
    # Stage 3: Dataset-wise In-Context Learning
    icl_dim = embed_dim * row_num_cls  # CLS tokens are concatenated for ICL
    self.icl_predictor = ICLearning(
        loss=loss,
        max_classes=max_classes,
        d_model=icl_dim,
        num_blocks=icl_num_blocks,
        nhead=icl_nhead,
        dim_feedforward=icl_dim * ff_factor,
        activation=activation,
        rngs=rngs,
        dtype=self.dtype,
        zero_out_proj_init=zero_out_proj_init,
        attention_impl=icl_attention_impl,
        use_bias=use_bias,
        cache_icl_input_only=cache_icl_input_only,
    )
    self.y_embedding_scheme = y_embedding_scheme

  @jt.typed
  def __call__(
      self,
      X: jt.Float[jax.Array | np.ndarray, 'B T H'],
      y: jt.Shaped[Array, 'B T'],
      train_size: jt.Int[jax.Array | np.ndarray, 'B'] | jt.Int[jax.Array | np.ndarray, 'B 1'],
      d: jt.Int[jax.Array | np.ndarray, 'B'] | jt.Int[jax.Array | np.ndarray, 'B 1'] | None = None,
      cat_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B H']] = None,
      softmax_temperature: float = 0.9,
      num_classes: Optional[int] = None,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T E']      # Regression
        | jt.Float[jax.Array | np.ndarray, 'B T K']):  # Classification
    """Processes tabular data through nested encoders and ICL predictor.

    Args:
      X: Input feature tensor.
      y: Target value tensor.
      train_size: Number of training samples per batch element.
      d: Number of features per dataset.
      cat_mask: Boolean mask indicating categorical features.
      softmax_temperature: Temperature for softmax (inference only).
      num_classes: Number of classes in the dataset.

    Returns:
      Logits or probabilities for the sequence.
    """
    # Cast input to bfloat16 at the beginning of the model call
    X = jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)

    _, T, _ = X.shape
    if train_size.ndim == 2:
      train_size = jnp.squeeze(train_size, axis=-1)
    if d is not None:
      if d.ndim == 2:
        d = jnp.squeeze(d, axis=-1)
    valid_mask = X != -100.0

    train_size = train_size.astype(jnp.int32)

    # Stage 1: Cell-wise embedding
    embeddings = self.cell_embedder(
        X, y, train_size=train_size, d=d, cat_mask=cat_mask
    )  # (B, T, HC, E)


    # Column-wise embedding
    embeddings = typing.cast(
        Array,
        self.col_embedder(
            embeddings,
            train_size=train_size,
        ),
    )
    # Prepend CLS tokens
    B1, T1, _, _ = embeddings.shape
    cls_tokens_expanded = jnp.broadcast_to(
        self.cls_tokens[...], (B1, T1, self.row_num_cls, self.embed_dim)
    )

    embeddings = jnp.concatenate(
        [cls_tokens_expanded, embeddings], axis=-2
    )

    embeddings = self.row_interactor(
        embeddings,
        d=d,
    )


    embeddings = typing.cast(
        Array,
        self.col_embedder_2(
            embeddings,
            train_size=train_size,
        ),
    )

    # Stage 2: Row-wise interaction
    representations = self.row_interactor_2(
        embeddings,
        d=d,
    )

    if (
        self.is_classifier
        and num_classes is not None
        and num_classes > self.icl_predictor.max_classes
    ):
      raise ValueError(
          f'Number of classes ({num_classes}) exceeds the maximum supported '
          f'({self.icl_predictor.max_classes}).'
      )
    out = self.icl_predictor(
        representations,
        y,
        train_size=train_size,
    )

    return out

  @jt.typed
  def prefill(
      self,
      X: jt.Float[jax.Array | np.ndarray, 'B T H'],
      y: jt.Shaped[jax.Array | np.ndarray, 'B T'],
      d: jt.Int[jax.Array | np.ndarray, 'B'] = None,
      cat_mask: jt.Bool[jax.Array | np.ndarray, 'B H'] | None = None,
  ) -> (
      Tuple[(jt.Float[jax.Array | np.ndarray, 'B T K'] | jt.Float[jax.Array | np.ndarray, 'B T E']),
             Dict[str, Any] # cache
      ]):
    """Prefills the model with training data and returns the cache.

    Args:
      X: Input feature tensor.
      y: Target value tensor.
      d: Number of features per dataset.
      cat_mask: Boolean mask indicating categorical features.

    Returns:
      Tuple of (logits, updated_cache).
    """
    X = jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)
    y = y.astype(self.dtype)

    _, T, _ = X.shape

    # Pad T to be a multiple of 128 for efficiency.
    # TODO: Do this only when use_flash_attention is True.
    block_size = 128
    pad_len = ((T - 1) // block_size + 1) * block_size - T
    paddings_X = [(0, 0), (0, pad_len), (0, 0)]
    X = jnp.pad(X, paddings_X, constant_values=-100.0)
    paddings_y = [(0, 0), (0, pad_len)]
    y = jnp.pad(y, paddings_y, constant_values=-100.0)
    total_padded_length = X.shape[1]

    # For prefill, all data is training data.
    # We calculate train_size based on valid y values to handle external padding.
    # TODO: Pass 'train_size' as an argument to prefill so the '-100'
    # value can be used for regression tasks.
    assert y.ndim == 2, 'y should have shape (B, T)'
    is_valid = jnp.not_equal(y, -100.0)
    train_size = jnp.sum(is_valid, axis=-1, dtype=jnp.int32).astype(jnp.int32)

    # Stage 0: Cell-wise embedding
    cell_embeddings = self.cell_embedder(
        X, y, train_size=train_size, d=d, cat_mask=cat_mask
    )


    # Stage 1: Column-wise embedding
    res = typing.cast(
        Tuple[jnp.ndarray, Array],
        self.col_embedder(
            cell_embeddings,
            train_size=train_size,
            return_repr=True,
        ),
    )
    embeddings, cache_col1 = res


    # Prepend CLS tokens to embeddings before row interactor
    B1, T1, _, _ = embeddings.shape
    cls_tokens_expanded = jnp.broadcast_to(
        self.cls_tokens[...], (B1, T1, self.row_num_cls, self.embed_dim)
    )
    embeddings = jnp.concatenate([cls_tokens_expanded, embeddings], axis=-2)

    embeddings = self.row_interactor(
        embeddings, d=d
    )

    res2 = typing.cast(
        Tuple[jnp.ndarray, Array],
        self.col_embedder_2(
            embeddings,
            train_size=train_size,
            return_repr=True,
        ),
    )
    embeddings, cache_col2 = res2

    representations = self.row_interactor_2(
        embeddings, d=d
    )

    # Stage 3: ICL
    logits_icl, cache_icl = self.icl_predictor(
        representations,
        y,
        train_size=train_size,
        return_cache=True,
    )

    cache = {
        'col1': cache_col1,
        'col2': cache_col2,
        'icl': cache_icl,
    }
    # Unpad to the original 'T'.
    return logits_icl[:, :T, :], cache

  @jt.typed
  def decode(
      self,
      X: jt.Float[jax.Array | np.ndarray, 'B T H'],
      cache: Dict[str, Any],
      d: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      cat_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B H']] = None,
      softmax_temperature: float = 0.9,
      num_classes: Optional[int] = None,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T K'] | jt.Float[jax.Array | np.ndarray, 'B T E']):
    """Generates predictions for test data using cached KVs.

    Args:
      X: Input feature tensor.
      cache: KV cache from prefill.
      d: Number of features per dataset.
      cat_mask: Boolean mask indicating categorical features.
      softmax_temperature: Temperature for softmax.
      num_classes: Number of classes in the dataset.

    Returns:
      Logits for the test data.
    """
    X = jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)

    B, T, _ = X.shape

    # Pad T to be a multiple of 128
    # TODO: Do this only when use_flash_attention is True.
    block_size = 128
    pad_len = ((T - 1) // block_size + 1) * block_size - T
    paddings_X = [(0, 0), (0, pad_len), (0, 0)]
    X = jnp.pad(X, paddings_X, constant_values=-100.0)

    # y is not used for features in decode (test rows).
    y = jnp.full((B, X.shape[1]), -100.0, dtype=self.dtype)

    # Stage 0: Cell-wise embedding for test rows
    cell_embeddings = self.cell_embedder(
        X, y,
        # train_size is 0 for the current batch (all are test rows relative to the cache)
        train_size=jnp.zeros((B,),
                             dtype=jnp.int32),
        d=d, cat_mask=cat_mask
    )


    # Stage 1: Column-wise embedding with cache
    embeddings = typing.cast(
        Array,
        self.col_embedder(
            cell_embeddings,
            train_size=None,
            cached_repr=cache['col1'],
        ),
    )


    # Prepend CLS tokens
    B1, T1, _, _ = embeddings.shape
    cls_tokens_expanded = jnp.broadcast_to(
        self.cls_tokens[...], (B1, T1, self.row_num_cls, self.embed_dim)
    )
    embeddings = jnp.concatenate([cls_tokens_expanded, embeddings], axis=-2)

    embeddings = self.row_interactor(
        embeddings, d=d
    )

    embeddings = typing.cast(
        Array,
        self.col_embedder_2(
            embeddings,
            train_size=None,
            cached_repr=cache['col2'],
        ),
    )

    # Stage 2: Row-wise interaction
    representations = self.row_interactor_2(
        embeddings, d=d
    )

    # Stage 3: ICL with cache
    out = self.icl_predictor(
        representations,
        y,
        train_size=None,
        cache=cache['icl'],
    )

    # Unpad
    out = out[:, :T, ...]
    return out

```

## File: tabfm/src/jax/model_test.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib

from absl import logging
import chex
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tabfm.src.jax import model as tabfm_model
from tabfm.src.jax.model import AttentionImplementation, YEmbeddingScheme

from absl.testing import absltest
from absl.testing import parameterized


def _init_model_deterministically(model: nnx.Module):
  """Initializes model parameters independently of their creation order."""
  state = nnx.state(model, nnx.Param)
  master_key = jax.random.key(42)

  new_state_dict = {}
  flat_state = nnx.to_flat_state(state)
  for path, var in flat_state:
    # Derive a deterministic seed from the parameter name.
    logging.info('Initializing parameter: %s', '.'.join(str(p) for p in path))
    key = jax.random.fold_in(
        master_key,
        int(
            hashlib.sha256('.'.join(str(p) for p in path).encode('utf-8')).hexdigest(),
            16,
        )
        % (2**32),
    )
    # var.value.size)
    new_state_dict[path] = jax.random.normal(
        key, var.value.shape, var.value.dtype
    )

  nnx.update(model, nnx.State.from_flat_path(new_state_dict))


class TabFMDependenciesTest(parameterized.TestCase):
  """Tests the input/output dependencies of the TabFM model."""

  @parameterized.product(
      y_embedding_scheme=[
          YEmbeddingScheme.NONE,
          YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
      ],
      attention_impl=[
          AttentionImplementation.JAX
      ],
  )
  def test_call_dependencies_with_perturbation(
      self,
      y_embedding_scheme: YEmbeddingScheme = YEmbeddingScheme.NONE,
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
  ):
    # --- 1. Test Setup ---
    # Define small dimensions for the test case.
    B, T, H = 2, 8, 4  # Batch size, sequence length, number of features
    train_size = jnp.array([4, 5])
    max_classes = 5
    embed_dim = 6

    # Create a random key for reproducibility.
    key = jax.random.PRNGKey(0)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    # Instantiate the TabFM model with a minimal configuration.
    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=3,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=3,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
        y_embedding_scheme=y_embedding_scheme,
        dtype=jnp.float32,
        col_attention_impl=attention_impl,
        row_attention_impl=AttentionImplementation.JAX,
        icl_attention_impl=attention_impl,
    )
    _init_model_deterministically(model)

    # Create dummy input tensors.
    X_key, y_key = jax.random.split(data_key)
    X = jax.random.normal(X_key, (B, T, H))
    y = jax.random.randint(y_key, (B, T), 0, max_classes)

    # --- 2. Baseline Output ---
    @nnx.jit
    def get_logits(model, x_input, y_input, train_size):
      return model(
          X=x_input,
          y=y_input,
          train_size=train_size,
      )

    # Compute the original output for comparison.
    original_output = get_logits(model, X, y, train_size)

    # --- 3. Perturbation Test for X (Features) ---
    delta = 1
    batch_idx = np.arange(B)
    # Iterate over every feature in the input tensor X.
    for b in range(B):
      for t in range(T):
        for h in range(H):
          logging.info('Perturbing feature %d, %d, %d', b, t, h)
          # Perturb a single element of X by a small amount.
          X_perturbed = X.at[b, t, h].add(delta)  # pylint: disable=invalid-name
          perturbed_output = get_logits(model, X_perturbed, y, train_size)
          output_diff = jnp.abs(perturbed_output - original_output)
          # Make sure the outputs for other batch elements didn't change.
          self.assertTrue(np.all(output_diff[batch_idx != b, :, :] == 0))
          # Consider a prediction changed if any of its logits changed.
          prediction_changed = np.any(output_diff[b, train_size[b]:, :] != 0,
                                      axis=-1)
          chex.assert_shape(prediction_changed, (T - train_size[b],))
          if t < train_size[b]:
            # Test 1: Perturbing a training feature should affect
            # ALL test predictions
            self.assertTrue(np.all(prediction_changed))
          else:
            # Test 2: Perturbing a test feature should ONLY affect its
            # own prediction.
            test_idx = t - train_size[b]
            # Check that the corresponding test output changed.
            self.assertTrue(
                prediction_changed[test_idx],
                'Perturbing a test feature should affect its own prediction.',
            )
            # Check that other test outputs did NOT change.
            mask = np.ones(T - train_size[b], dtype=bool)
            mask[test_idx] = False
            self.assertTrue(
                np.all(prediction_changed[mask] == False),   # pylint: disable=g-explicit-bool-comparison
                'Perturbing a test feature should not affect other'
                ' test predictions.',
            )

    # --- 4. Perturbation Test for y (Labels) ---
    # Iterate over every label in the input tensor y.
    for b in range(B):
      for t in range(T):
        # Perturb a single label by changing it to the next class index.
        y_perturbed = y.at[b, t].set((y[b, t] + 1) % max_classes)
        perturbed_output = get_logits(model, X, y_perturbed, train_size)
        output_diff = jnp.abs(perturbed_output - original_output)
        # Make sure the outputs for other batch elements didn't change.
        self.assertTrue(np.all(output_diff[batch_idx != b, :, :] == 0))
        # Consider a prediction changed if any of its logits changed.
        prediction_changed = np.any(output_diff[b, train_size[b]:, :] != 0,
                                    axis=-1)

        if t < train_size[b]:
          # Test 3: Perturbing a training label should affect ALL test predictions.
          self.assertTrue(
              np.all(prediction_changed),
              'Perturbing a training label should affect all test predictions.',
          )
        else:
          # Test 4: Perturbing a test label should affect NO test predictions.
          self.assertTrue(
              np.all(prediction_changed == False),  # pylint: disable=g-explicit-bool-comparison
              'Perturbing a test label should not affect any test predictions.',
          )

  def test_regression_add_y_to_x_post_embedding(self):
    B, T, H = 2, 8, 4
    train_size = jnp.array([4, 5])
    max_classes = 5
    embed_dim = 6

    key = jax.random.PRNGKey(0)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    model = tabfm_model.TabFM(
        loss='rmse',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=2,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=1,
        rngs=rngs,
        y_embedding_scheme=YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
    )

    X_key, y_key = jax.random.split(data_key)
    X = jax.random.normal(X_key, (B, T, H))
    y = jax.random.normal(y_key, (B, T))

    @nnx.jit
    def get_logits(model, x_input, y_input, train_size):
      return model(
          X=x_input,
          y=y_input,
          train_size=train_size,
      )

    output = get_logits(model, X, y, train_size)
    self.assertEqual(output.shape, (B, T, 1))
    self.assertFalse(jnp.isnan(output).any())

  @parameterized.product(
      y_embedding_scheme=[
          YEmbeddingScheme.NONE,
          YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
      ],
  )
  def test_call_dependencies_with_padding_perturbation(
      self,
      y_embedding_scheme: YEmbeddingScheme = YEmbeddingScheme.NONE,
  ):
    # --- 1. Test Setup ---
    B, T, H = 2, 16, 8  # Batch size, sequence length, number of features
    T_actual = jnp.array([4, 5])  # Actual sequence length before padding
    H_actual = jnp.array([3, 4])  # Actual number of features before padding
    max_classes = 5
    embed_dim = 6

    key = jax.random.PRNGKey(0)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=3,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=3,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
        y_embedding_scheme=y_embedding_scheme,
        dtype=jnp.float32,
    )
    _init_model_deterministically(model)

    X_key, y_key = jax.random.split(data_key)
    X = jax.random.normal(X_key, (B, T, H))
    y = jax.random.randint(y_key, (B, T), 0, max_classes)

    # --- 2. Baseline Output ---
    @nnx.jit
    def get_logits(model, x_input, y_input, train_size, d):
      return model(
          X=x_input,
          y=y_input,
          train_size=train_size,
          d=d,
      )

    original_output = get_logits(model, X, y, T_actual, H_actual)

    # --- 3. Perturbation Test for X ---
    delta = 1.0
    batch_idx = np.arange(B)
    # Iterate over every element in the input tensor X.
    for b in range(B):
      for t in range(T):
        for h in range(H):
          X_perturbed = X.at[b, t, h].add(delta)
          perturbed_output = get_logits(
              model, X_perturbed, y, T_actual, H_actual
          )
          output_diff = jnp.abs(perturbed_output - original_output)

          # Make sure the outputs for other batch elements didn't change.
          self.assertTrue(np.all(output_diff[batch_idx != b, :, :] == 0))

          # Indices of test samples for the current batch element.
          test_indices = np.arange(T_actual[b], T)
          # Consider a prediction changed if any of its logits changed.
          prediction_changed = np.any(
              output_diff[b, test_indices, :] != 0, axis=-1
          )
          chex.assert_shape(prediction_changed, (T - T_actual[b],))

          is_padding_h = h >= H_actual[b]
          is_train = t < T_actual[b]

          if is_padding_h:
            logging.info('Perturbing padded H element %d, %d, %d', b, t, h)
            # Assert that NO output changed for any batch element.
            self.assertTrue(
                np.all(output_diff == 0),
                f'Perturbing padded element {b}, {t}, {h} changed the output.',
            )
          elif is_train:
            logging.info('Perturbing training element %d, %d, %d', b, t, h)
            # Perturbing a training feature should affect ALL test predictions.
            self.assertTrue(
                np.all(prediction_changed),
                f'Perturbing train X[{b}, {t}, {h}] did not affect all test'
                ' predictions.'
                f'\n X: {X}'
                f'\n X_perturbed: {X_perturbed}'
                f'\n original_output: {original_output}'
                f'\n perturbed_output: {perturbed_output}',
            )
          else:  # Test element (t >= T_actual[b])
            logging.info('Perturbing test element %d, %d, %d', b, t, h)
            # Perturbing a test feature should ONLY affect its own prediction.
            test_idx = t - T_actual[b]
            self.assertTrue(
                prediction_changed[test_idx],
                f'Perturbing test X[{b}, {t}, {h}] should affect its own'
                ' prediction.',
            )
            mask = np.ones(T - T_actual[b], dtype=bool)
            mask[test_idx] = False
            self.assertTrue(
                np.all(~prediction_changed[mask]),
                f'Perturbing test X[{b}, {t}, {h}] should not affect other'
                ' test predictions.',
            )

    # --- 4. Perturbation Test for y (Labels) ---
    for b in range(B):
      for t in range(T):
        y_perturbed = y.at[b, t].set((y[b, t] + 1) % max_classes)
        perturbed_output = get_logits(model, X, y_perturbed, T_actual, H_actual)
        output_diff = jnp.abs(perturbed_output - original_output)

        # Make sure the outputs for other batch elements didn't change.
        self.assertTrue(np.all(output_diff[batch_idx != b, :, :] == 0))

        test_indices = np.arange(T_actual[b], T)
        prediction_changed = np.any(
            output_diff[b, test_indices, :] != 0, axis=-1
        )
        chex.assert_shape(prediction_changed, (T - T_actual[b],))

        if t < T_actual[b]:
          # Perturbing a training label should affect ALL test predictions.
          self.assertTrue(
              np.all(prediction_changed),
              f'Perturbing train y[{b}, {t}] did not affect all test'
              ' predictions.',
          )
        else:
          # Perturbing a test label should affect NO test predictions.
          self.assertTrue(
              np.all(~prediction_changed),
              f'Perturbing test y[{b}, {t}] should not affect any test'
              ' predictions.',
          )

  @parameterized.product(
      y_embedding_scheme=[
          YEmbeddingScheme.NONE,
          YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
      ],
  )
  def test_padding_invariance(
      self,
      y_embedding_scheme: YEmbeddingScheme = YEmbeddingScheme.NONE,
  ):
    # --- 1. Test Setup ---
    B, T = 2, 16
    H_actual = jnp.array([3, 4])  # Actual number of features
    H1, H2 = 8, 12  # Different padding sizes
    T_actual = jnp.array([4, 5])  # Train sizes
    max_classes = 5
    embed_dim = 6

    key = jax.random.PRNGKey(1)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=3,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=3,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
        y_embedding_scheme=y_embedding_scheme,
        dtype=jnp.float32,
    )
    _init_model_deterministically(model)

    X_key, y_key = jax.random.split(data_key)
    # Common non-padded data
    X_core = jax.random.normal(X_key, (B, T, np.max(H_actual)))
    y = jax.random.randint(y_key, (B, T), 0, max_classes)

    # --- 2. Create Inputs with Different Padding ---
    X1 = jnp.zeros((B, T, H1))
    X2 = jnp.zeros((B, T, H2))

    for b in range(B):
      X1 = X1.at[b, :, :H_actual[b]].set(X_core[b, :, :H_actual[b]])
      X2 = X2.at[b, :, :H_actual[b]].set(X_core[b, :, :H_actual[b]])
      # Add some noise to padding to make sure it's not just zeros
      X1 = X1.at[b, :, H_actual[b]:].add(
          jax.random.normal(X_key, (T, (H1 - H_actual[b]).item()))
      )
      X2 = X2.at[b, :, H_actual[b]:].add(
          jax.random.normal(X_key, (T, (H2 - H_actual[b]).item()))
      )

    # --- 3. Compute Outputs ---
    @nnx.jit
    def get_logits(model, x_input, y_input, train_size, d):
      return model(
          X=x_input,
          y=y_input,
          train_size=train_size,
          d=d,
      )

    output1 = get_logits(model, X1, y, T_actual, H_actual)
    output2 = get_logits(model, X2, y, T_actual, H_actual)

    # --- 4. Assertion ---
    # The outputs should be identical since the non-padding part is the same
    np.testing.assert_array_almost_equal(output1, output2, decimal=5)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NONE',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'expected_output': [[
              [2.35432, 0.403802, -1.134074],
              [1.628627, 0.455984, -1.079756],
              [2.470968, 0.44621, 0.312044],
              [2.584721, 0.455619, 0.324722],
          ]],
      },
      {
          'testcase_name': 'ADD',
          'y_embedding_scheme': YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
          'expected_output': [[
              [2.1624, 0.405149, -1.114775],
              [1.543, 0.468879, -1.069826],
              [2.62295, 0.459419, 0.371579],
              [2.715013, 0.465245, 0.365446],
          ]],
      },
      {
          'testcase_name': 'activation_relu',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'activation': 'relu',
          'expected_output': [[
              [2.314818, 0.415296, -1.289602],
              [0.24321, 0.901325, -1.016127],
              [3.422867, 0.42615, 1.615581],
              [3.503174, 0.423493, 1.591408],
          ]],
      },
      {
          'testcase_name': 'activation_swiglu',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'activation': 'swiglu',
          'expected_output': [[
              [2.12426, 0.439869, -1.233905],
              [1.812752, 0.483976, -1.207603],
              [-1.323864, 1.370978, -0.101007],
              [-1.358536, 1.385991, -0.088265],
          ]],
      },
      {
          'testcase_name': 'feature_group_true',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'feature_group': True,
          'expected_output': [[
              [0.891614, 0.604129, -0.962587],
              [0.867184, 0.612097, -0.963016],
              [0.596546, 0.412646, 0.349397],
              [0.639643, 0.403368, 0.3539],
          ]],
      },
      {
          'testcase_name': 'no_fourier_features',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'use_fourier_features': False,
          'expected_output': [[
              [0.813858, 0.62535, -0.945271],
              [0.74903, 0.641479, -0.927889],
              [1.282514, 0.353903, 0.455361],
              [0.966555, 0.363276, 0.374227],
          ]],
      },
  )
  def test_golden_output(
      self,
      y_embedding_scheme: YEmbeddingScheme,
      expected_output: list[list[list[float]]],
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      activation: str = 'gelu',
      feature_group: bool = False,
      use_fourier_features: bool = True,
      icl_num_blocks: int = 1,
  ):
    # --- 1. Test Setup ---
    B, T, H = 1, 4, 3  # Batch size, sequence length, number of features
    train_size = jnp.array([2])
    max_classes = 3
    embed_dim = 4

    key = jax.random.PRNGKey(42)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=icl_num_blocks,
        icl_nhead=1,
        rngs=rngs,
        y_embedding_scheme=y_embedding_scheme,
        icl_attention_impl=attention_impl,
        dtype=jnp.float32,
        activation=activation,
        feature_group=feature_group,
        use_fourier_features=use_fourier_features,

    )
    _init_model_deterministically(model)

    X_key, y_key = jax.random.split(data_key)
    X = jax.random.normal(X_key, (B, T, H))
    y = jax.random.randint(y_key, (B, T), 0, max_classes)

    # --- 2. Compute Output ---
    @nnx.jit
    def get_logits(model, x_input, y_input, train_size):
      return model(
          X=x_input,
          y=y_input,
          d=jnp.array([H]),
          train_size=train_size,
      )

    output = get_logits(model, X, y, train_size)
    logging.info('test_golden_output: %s', output)





  def test_head_sharding_consistency(self):
    # Setup minimal dimensions.
    # Setup mesh dimensions based on available devices.
    num_devices = jax.device_count()
    if num_devices >= 2:
      heads_dim = 2
      data_dim = num_devices // 2
    else:
      heads_dim = 1
      data_dim = 1

    devices = np.array(jax.devices()).reshape(data_dim, heads_dim)
    mesh = jax.sharding.Mesh(devices, ('data', 'heads'))

    # Ensure B and nhead are multiples of mesh dimensions.
    B = data_dim * 1
    T, H = 2, 2
    nhead = heads_dim * 2
    max_classes = 2
    embed_dim = nhead * 4

    key = jax.random.PRNGKey(0)
    init_key, data_key = jax.random.split(key)

    # 1. Baseline model (non-sharded)
    model_baseline = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        col_nhead=nhead,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=nhead,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=nhead,
        rngs=nnx.Rngs(params=init_key),
        # shard_attention_over_heads=False,
        col_attention_impl=AttentionImplementation.JAX,
        row_attention_impl=AttentionImplementation.JAX,
        icl_attention_impl=AttentionImplementation.JAX,
        dtype=jnp.float32,
    )
    _init_model_deterministically(model_baseline)

    # 2. Sharded model
    model_sharded = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        col_nhead=nhead,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=nhead,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=nhead,
        rngs=nnx.Rngs(params=init_key),
        # shard_attention_over_heads=True,
        col_attention_impl=AttentionImplementation.JAX,
        row_attention_impl=AttentionImplementation.JAX,
        icl_attention_impl=AttentionImplementation.JAX,
        dtype=jnp.float32,
    )
    # Ensure both models have the exact same weights.
    nnx.update(model_sharded, nnx.state(model_baseline))
    chex.assert_trees_all_close(
        nnx.state(model_baseline), nnx.state(model_sharded), atol=1e-8
    )

    data_sharding = jax.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(
            'data',
        ),
    )

    def run_model(model, X, y, train_size):
      return model(X=X, y=y, train_size=train_size)

    with mesh:
      X = jax.device_put(jax.random.normal(data_key, (B, T, H)), data_sharding)
      y = jax.device_put(jnp.zeros((B, T), dtype=jnp.int32), data_sharding)
      train_size = jax.device_put(jnp.array([2] * B), data_sharding)
      output_baseline = run_model(model_baseline, X, y, train_size)
      output_sharded = run_model(model_sharded, X, y, train_size)

    # If running on a single device, they MUST be exactly identical.
    # On multiple devices, allow for some numerical drift due to distributed computation.
    atol = 1e-6
    chex.assert_trees_all_close(output_baseline, output_sharded, atol=atol)


class CellEmbedderInformationFlowTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.B = 2
    self.T = 4
    self.H = 8
    self.max_classes = 5
    self.embed_dim = 6
    self.y_embedding_scheme = YEmbeddingScheme.NONE
    self.y_col_embedder_encoder_nhid = 6

  @parameterized.product(
      feature_group=[True, False],
      use_cat_mask=[True, False],
      use_d=[True, False],
  )
  def test_information_flow(self, feature_group, use_cat_mask, use_d):
    key = jax.random.PRNGKey(42)
    init_key, data_key, perturb_key = jax.random.split(key, 3)
    rngs = nnx.Rngs(params=init_key)
    feature_group_size = 3 if feature_group else 1

    model = tabfm_model.CellEmbedder(
        embed_dim=self.embed_dim,
        y_embedding_scheme=self.y_embedding_scheme,
        max_classes=self.max_classes,
        y_col_embedder_encoder_nhid=self.y_col_embedder_encoder_nhid,
        is_classifier=True,
        feature_group=feature_group,
        feature_group_size=feature_group_size,
        rngs=rngs,
    )
    _init_model_deterministically(model)

    X = jax.random.normal(data_key, (self.B, self.T, self.H))
    y = jax.random.randint(data_key, (self.B, self.T,), 0, self.max_classes)

    if use_d:
      # Use different d for each example in batch. Example 0: d=6, Example 1: d=7.
      d = jnp.array([6, 7])
    else:
      d = None

    if use_cat_mask:
      cat_mask = jax.random.bernoulli(data_key, 0.5, (self.B, self.H))
    else:
      cat_mask = None

    # Output baseline
    out = model(X, y, cat_mask=cat_mask, d=d) # Shape (B, T, H, E)

    # 1. Perturb input column h_in within the active range (h_in < min(d))
    h_in = 2
    perturbation = jax.random.normal(perturb_key, (self.B, self.T)) * 0.1
    perturbation = jnp.where(jnp.abs(perturbation) < 1e-4, 0.1, perturbation)
    X_perturbed = X.at[:, :, h_in].add(perturbation)

    # Output perturbed
    out_perturbed = model(X_perturbed, y, cat_mask=cat_mask, d=d)

    # Determine which output indices should change
    # Note: if d is used, modulo is by d, not H.
    def get_expected_changed(b_idx):
      d_val = d[b_idx].item() if use_d else self.H
      if h_in >= d_val:
        return set()
      if not feature_group:
        return {h_in}
      offsets = [(2**i) - 1 for i in range(feature_group_size)]
      return {(h_in - offset) % d_val for offset in offsets}

    # Verify output changes for each example in batch
    for b in range(self.B):
      expected_changed = get_expected_changed(b)
      d_limit = d[b] if use_d else self.H

      for h in range(self.H):
        diff = jnp.abs(out[b, :, h, :] - out_perturbed[b, :, h, :])
        max_diff = jnp.max(diff)

        if h >= d_limit:
          # Padding features should always be 0
          self.assertEqual(jnp.max(jnp.abs(out[b, :, h, :])), 0.0)
          self.assertEqual(max_diff, 0.0, f"Expected padded output column {h} in example {b} to remain zero.")
        elif h in expected_changed:
          self.assertGreater(max_diff, 1e-5, f"Expected output column {h} in example {b} to change, but it didn't.")
        else:
          self.assertEqual(max_diff, 0.0, f"Expected output column {h} in example {b} to remain unchanged, but it changed.")

    # 2. If using d, perturb a feature that is OUTSIDE the active range for at least one example
    if use_d:
      h_in_padded = 6 # >= d[0] (which is 6), so it shouldn't affect example 0
      X_perturbed_padded = X.at[:, :, h_in_padded].add(perturbation)
      out_perturbed_padded = model(X_perturbed_padded, y, cat_mask=cat_mask, d=d)

      # Example 0: h_in_padded >= d[0], so nothing should change
      diff0 = jnp.abs(out[0, :, :, :] - out_perturbed_padded[0, :, :, :])
      self.assertEqual(jnp.max(diff0), 0.0, "Perturbing a feature >= d[0] should not affect example 0.")

      # Example 1: h_in_padded < d[1] (which is 7), so something SHOULD change
      diff1 = jnp.abs(out[1, :, :, :] - out_perturbed_padded[1, :, :, :])
      self.assertGreater(jnp.max(diff1), 1e-5, "Perturbing a feature < d[1] should affect example 1.")


class TabFMPrefillDecodeTest(parameterized.TestCase):
  """Tests the consistency between full forward pass and prefill/decode."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(0)

  @parameterized.product(
      cache_icl_input_only=[True, False],
  )
  def test_prefill_decode_consistency(self, cache_icl_input_only):
    # This test verifies that prefill + decode produces similar results to a
    # full forward pass for the test portion of the data.
    B, T, H = 1, 20, 4
    train_len = 10
    embed_dim = 32
    max_classes = 5

    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=2,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        rngs=self.rngs,
        cache_icl_input_only=cache_icl_input_only,
        dtype=jnp.float32,
    )
    _init_model_deterministically(model)
    # Scale down weights to avoid chaotic numerical noise amplification on TPU.
    state = nnx.state(model, nnx.Param)
    flat_state = nnx.to_flat_state(state)
    new_state_dict = {}
    for path, var in flat_state:
      new_state_dict[path] = var.value * 0.02
    nnx.update(model, nnx.State.from_flat_path(new_state_dict))

    X = jax.random.normal(self.rngs.params(), (B, T, H))
    y = jax.random.randint(self.rngs.params(), (B, T), 0, max_classes)
    train_size = jnp.array([train_len])

    # 1. Full forward pass — reference output on the test portion.
    out_full = model(X, y, train_size)
    out_full_test = out_full[:, train_len:, :]

    # 2. Prefill + Decode split pass.
    X_train = X[:, :train_len, :]
    y_train = y[:, :train_len]
    X_test = X[:, train_len:, :]

    _, cache = model.prefill(X_train, y_train)

    # Verify the cache has the expected structure.
    icl_cache = cache['icl']
    self.assertIsInstance(icl_cache, tabfm_model.ICLearningCache)

    out_decode = model.decode(X_test, cache)

    # The split prefill+decode path should produce results close to the full pass.
    chex.assert_trees_all_close(out_full_test, out_decode, atol=0.2, rtol=0.1)


if __name__ == '__main__':
  absltest.main()

```

## File: tabfm/src/jax/tabfm_v1_0_0.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TabFM v1.0.0 Model Release.

This module provides simple access to the TabFM v1.0.0 model architecture and
allows downloading/restoring pre-trained weights from Hugging Face or loading
them from a local path.
"""

from dataclasses import dataclass
import os
import threading
from typing import Any, Dict, Optional
from absl import logging
from flax import nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from huggingface_hub import ModelHubMixin, snapshot_download
from tabfm.src.jax import checkpointing
from tabfm.src.jax.model import TabFM, YEmbeddingScheme

# Hugging Face repository ID for TabFM v1.0.0
HF_REPO_ID = "google/tabfm-1.0.0-jax"


@dataclass(frozen=True)
class Config:
  """Hardcoded architecture configuration for TabFM v1.0.0."""

  loss: str = "cross_entropy"
  max_classes: int = 10
  embed_dim: int = 256
  col_num_blocks: int = 3
  col_nhead: int = 4
  col_num_inds: int = 256
  row_num_blocks: int = 3
  row_nhead: int = 8
  row_num_cls: int = 8
  row_rope_base: float = 100000.0
  icl_num_blocks: int = 24
  icl_nhead: int = 8
  ff_factor: int = 4
  activation: str = "swiglu"
  feature_group: bool = True
  feature_group_size: int = 3
  use_fourier_features: bool = True
  fourier_features_num_frequencies: int = 32
  fourier_features_sigma: float = 1.0
  cache_icl_input_only: bool = False
  y_embedding_scheme: YEmbeddingScheme = (
      YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING
  )
  use_bias: bool = False

  def to_dict(self) -> Dict[str, Any]:
    return {
        "loss": self.loss,
        "max_classes": self.max_classes,
        "embed_dim": self.embed_dim,
        "col_num_blocks": self.col_num_blocks,
        "col_nhead": self.col_nhead,
        "col_num_inds": self.col_num_inds,
        "row_num_blocks": self.row_num_blocks,
        "row_nhead": self.row_nhead,
        "row_num_cls": self.row_num_cls,
        "row_rope_base": self.row_rope_base,
        "icl_num_blocks": self.icl_num_blocks,
        "icl_nhead": self.icl_nhead,
        "ff_factor": self.ff_factor,
        "activation": self.activation,
        "feature_group": self.feature_group,
        "feature_group_size": self.feature_group_size,
        "use_fourier_features": self.use_fourier_features,
        "fourier_features_num_frequencies": (
            self.fourier_features_num_frequencies
        ),
        "fourier_features_sigma": self.fourier_features_sigma,
        "cache_icl_input_only": self.cache_icl_input_only,
        "y_embedding_scheme": self.y_embedding_scheme,
        "use_bias": self.use_bias,
    }


@dataclass(frozen=True)
class ClassificationConfig(Config):
  """Architecture configuration for TabFM v1.0.0 classification model."""

  loss: str = "cross_entropy"


@dataclass(frozen=True)
class RegressionConfig(Config):
  """Architecture configuration for TabFM v1.0.0 regression model."""

  loss: str = "rmse"


def _restore_from_dir(
    model: TabFM,
    checkpoint_path: str,
    model_type: str,
    step: Optional[int],
) -> TabFM:
  """Restores Orbax checkpoint into model in place and returns model."""
  if not os.path.exists(os.path.join(checkpoint_path, "orbax")):
    potential_path = os.path.join(checkpoint_path, model_type)
    if os.path.exists(os.path.join(potential_path, "orbax")):
      checkpoint_path = potential_path

  checkpoint_manager = checkpointing.create_checkpoint_manager(
      checkpoint_path, read_only=True
  )
  if step is None:
    step = checkpoint_manager.latest_step()
    if step is None:
      raise ValueError(f"No checkpoints found in {checkpoint_path}/orbax")

  state = nnx.state(model)
  restored = checkpoint_manager.restore(
      step,
      args=ocp.args.Composite(
          params=ocp.args.StandardRestore(state, strict=False)
      ),
  )
  nnx.update(model, restored["params"])
  return model


class TabFM_HF(TabFM, ModelHubMixin):
  """JAX TabFM model with HuggingFace Hub support.

  Subclasses TabFM directly rather than wrapping it, since NNX has special
  handling for nnx.Module subclasses that a wrapper breaks under JIT.
  Provides from_pretrained(), save_pretrained(), and push_to_hub() via
  ModelHubMixin.
  """

  def __init__(self, *args, model_type: str = "classification", **kwargs):
    super().__init__(*args, **kwargs)
    object.__setattr__(self, "_model_type", model_type)

  def __setattr__(self, name: str, value: Any):
    if name in ("_hub_mixin_config", "_model_type"):
      # Bypass NNX checking entirely and write directly to Python's object dict.
      object.__setattr__(self, name, value)
    else:
      super().__setattr__(name, value)

  @classmethod
  def _from_pretrained(
      cls,
      *,
      model_id: str,
      revision: Optional[str],
      cache_dir,
      force_download: bool,
      local_files_only: bool,
      token,
      model_type: str = "classification",
      step: Optional[int] = None,
      col_attention_impl: str = "flash",
      row_attention_impl: str = "jax",
      icl_attention_impl: str = "flash",
      dtype: Any = jnp.bfloat16,
      **kwargs,
  ) -> "TabFM_HF":
    from tabfm.src.jax.model import AttentionImplementation  # pylint: disable=g-import-not-at-top

    if model_type == "classification":
      config = ClassificationConfig()
    elif model_type == "regression":
      config = RegressionConfig()
    else:
      raise ValueError(
          f"Unsupported model_type: {model_type!r}. "
          "Must be 'classification' or 'regression'."
      )

    config_dict = config.to_dict()
    config_dict["col_attention_impl"] = AttentionImplementation(col_attention_impl)
    config_dict["row_attention_impl"] = AttentionImplementation(row_attention_impl)
    config_dict["icl_attention_impl"] = AttentionImplementation(icl_attention_impl)

    model = cls(rngs=nnx.Rngs(0), dtype=dtype, model_type=model_type, **config_dict)

    if os.path.isdir(model_id):
      checkpoint_path = model_id
    else:
      logging.info(
          "Downloading TabFM v1.0.0 JAX %s weights from Hugging Face...",
          model_type,
      )
      base_path = snapshot_download(
          repo_id=model_id,
          revision=revision,
          cache_dir=cache_dir,
          force_download=force_download,
          local_files_only=local_files_only,
          token=token,
          allow_patterns=[f"{model_type}/**"],
      )
      checkpoint_path = os.path.join(base_path, model_type)

    return _restore_from_dir(model, checkpoint_path, model_type, step)

  def _save_pretrained(self, save_directory):
    """Save Orbax checkpoint to save_directory/<model_type>/orbax/."""
    out_path = os.path.join(save_directory, self._model_type)
    os.makedirs(out_path, exist_ok=True)
    manager = checkpointing.create_checkpoint_manager(out_path, read_only=False)
    state = nnx.state(self)
    manager.save(0, args=ocp.args.Composite(params=ocp.args.StandardSave(state)))
    manager.wait_until_finished()


# Process-wide memo of restored models (shared by load() and TabFM_HF).
_LOAD_CACHE_LOCK = threading.Lock()
_LOAD_CACHE: Dict[Any, "TabFM_HF"] = {}


def load(
    model_type: str = "classification",
    checkpoint_path: Optional[str] = None,
    step: Optional[int] = None,
    *,
    col_attention_impl: str = "flash",
    row_attention_impl: str = "jax",
    icl_attention_impl: str = "flash",
    dtype: Any = jnp.bfloat16,
    use_cache: bool = True,
) -> "TabFM_HF":
  """Loads the TabFM v1.0.0 JAX model with pre-trained weights.

  If `checkpoint_path` is not provided, downloads weights from Hugging Face
  (google/tabfm-1.0.0-jax), fetching only the requested model_type subfolder.

  Args:
    model_type: 'classification' or 'regression'.
    checkpoint_path: Local directory containing the 'orbax/' checkpoint, or
      None to download from Hugging Face.
    step: Checkpoint step to restore (for local loading).
    col_attention_impl: Attention impl for column-attention layers ('jax' or
      'flash').
    row_attention_impl: Attention impl for row-attention layers.
    icl_attention_impl: Attention impl for ICL layers.
    dtype: JAX compute dtype.
    use_cache: Reuse a process-wide cached model for identical settings.

  Returns:
    The restored TabFM_HF model.
  """
  cache_key = (
      model_type,
      checkpoint_path,
      step,
      col_attention_impl,
      row_attention_impl,
      icl_attention_impl,
      str(dtype),
  )
  if use_cache:
    _LOAD_CACHE_LOCK.acquire()
  try:
    if use_cache and cache_key in _LOAD_CACHE:
      return _LOAD_CACHE[cache_key]

    result = TabFM_HF.from_pretrained(
        HF_REPO_ID if checkpoint_path is None else checkpoint_path,
        model_type=model_type,
        step=step,
        col_attention_impl=col_attention_impl,
        row_attention_impl=row_attention_impl,
        icl_attention_impl=icl_attention_impl,
        dtype=dtype,
    )

    if use_cache:
      _LOAD_CACHE[cache_key] = result
    return result
  finally:
    if use_cache:
      _LOAD_CACHE_LOCK.release()

```

## File: tabfm/src/pytorch/BUILD

```
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//tabfm:__subpackages__"])

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "@pip//torch",
    ],
)

py_library(
    name = "tabfm_v1_0_0",
    srcs = ["tabfm_v1_0_0.py"],
    deps = [
        ":model",
        "@pip//absl_py",
        "@pip//torch",
        "@pip//huggingface_hub",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":model",
        "//tabfm/src/jax:model",
        "//tabfm/src:torch_convert",
        "@pip//numpy",
        "@pip//torch",
        "@pip//flax",
        "@pip//jax",
    ],
)

```

## File: tabfm/src/pytorch/__init__.py

```
# tabfm src pytorch package

```

## File: tabfm/src/pytorch/model.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EXPERIMENTAL faithful-ish PyTorch port of the TabFM architecture (fwd path).

Module/param names mirror the JAX model (tabfm/src/model.py) so weight
conversion is mechanical. The transformer math (attention with RoPE +
PerDimScale + q/k RMSNorm + SDPA at scale=1.0, swiglu FFN, full MAB) has been
parity-verified against JAX to ~1e-6 (float32). The embedding/ICL paths mirror
the JAX code but are validated by the end-to-end converter parity test.

NOT yet wired to Orbax weights; see torch_parity_harness.py for the converter
and parity gates.
"""

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _gelu_tanh(x):
  # jax.nn.gelu defaults to the tanh approximation -> match it.
  return F.gelu(x, approximate="tanh")


def get_activation(name):
  # Activations are stored as module attributes (e.g. MLP.act), so they must be
  # picklable for AutoGluon/TabArena's pickle-based save. A module-level
  # function pickles by reference; a lambda would not.
  return {"relu": F.relu,
          "gelu": _gelu_tanh,
          "silu": F.silu}[name]


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x):
    # Normalize entirely in float32 (x * rsqrt * weight), cast back at the end --
    # matches JAX/Flax, which keeps x*rsqrt in float32. Doing the multiply in bf16
    # (casting rsqrt down first) loses precision and accumulates across the ~36
    # RMSNorms per transformer stack.
    dt = x.dtype
    xf = x.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    return ((xf * torch.rsqrt(v + self.eps)) * self.weight.float()).to(dt)


def rope_interleaved(x, base):
  """Interleaved RoPE over the T axis of [B, T, N, Dh] (lucidrains convention)."""
  dh, t = x.shape[-1], x.shape[1]
  inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
  f = torch.outer(torch.arange(t, device=x.device).float(), inv)
  cos = f.cos().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
  sin = f.sin().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
  x1, x2 = x[..., 0::2], x[..., 1::2]
  rot = torch.stack((-x2, x1), -1).reshape_as(x)
  return x * cos + rot * sin


class RoPE(nn.Module):
  """One RoPE per Encoder, holding the inverse-frequency buffer loaded FROM the
  checkpoint (JAX stores `rope.freqs`, computed in bf16 at train time -- recomputing
  it in fp32 differs by ~1e-3 and that error grows with sequence length)."""

  def __init__(self, dim, base):
    super().__init__()
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # init = formula; overwritten on load
    self.register_buffer("freqs", inv)

  def rotate(self, x):  # x: [B, T, N, Dh], rotate over the T axis
    t = x.shape[1]
    f = torch.outer(torch.arange(t, device=x.device).float(), self.freqs.float())
    cos = f.cos().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
    sin = f.sin().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), -1).reshape_as(x)
    return x * cos + rot * sin


class MultiheadAttention(nn.Module):
  def __init__(self, d_model, nhead, rope_base=None):
    super().__init__()
    self.nhead, self.hd = nhead, d_model // nhead
    self.rope_base = rope_base  # None => no RoPE
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)
    self.query_ln = RMSNorm(self.hd)
    self.key_ln = RMSNorm(self.hd)
    self.per_dim_scale = nn.Parameter(torch.zeros(self.hd))

  def forward(self, query, key, value, attn_mask=None, rope=None,
              cached_kv=None, return_kv=False):
    """Computes multi-head attention, optionally with a K/V cache.

    At most one of cached_kv, return_kv is set. cached_kv is a (k, v) tuple of
    already-projected (and, where used, rotated and key-normalized) tensors of
    shape [B, T_src, N, D] from a prior call; key and value are then None and
    their projections are skipped. return_kv returns the freshly computed
    (k, v) in that layout for a later call to reuse.
    """
    b, tq, d = query.shape
    q = self.q_proj(query).view(b, tq, self.nhead, self.hd)

    if cached_kv is not None:
      assert key is None and value is None, (
          "key/value must be None when cached_kv is provided.")
      cached_k, cached_v = cached_kv
      # Cached K/V may be int8-quantized; dequantize to compute dtype before use.
      k = cached_k.dequantize(q.dtype) if isinstance(cached_k, QuantizedTensor) else cached_k
      v = cached_v.dequantize(q.dtype) if isinstance(cached_v, QuantizedTensor) else cached_v
    else:
      assert key is not None and value is not None, (
          "key/value must not be None when cached_kv is absent.")
      k = self.k_proj(key).view(b, key.shape[1], self.nhead, self.hd)
      v = self.v_proj(value).view(b, value.shape[1], self.nhead, self.hd)

    if self.rope_base is not None:
      # Cached K is already post-RoPE, so only rotate freshly-computed K.
      q = rope.rotate(q) if rope is not None else rope_interleaved(q, self.rope_base)
      if cached_kv is None:
        k = rope.rotate(k) if rope is not None else rope_interleaved(k, self.rope_base)

    q = self.query_ln(q)
    if cached_kv is None:
      k = self.key_ln(k)
    # per-dim scale in float32 (softplus), then cast to compute dtype -- matches JAX PerDimScale.
    scale = 1.442695041 / math.sqrt(self.hd) * F.softplus(self.per_dim_scale.float())
    q = q * scale.to(q.dtype)

    new_k, new_v = k, v  # cache format: [B, T_src, N, D], pre-transpose.

    q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # [B,N,T,D]
    # bf16 SDPA (flash already does the softmax in float32 internally).
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
    out = self.out_proj(o.transpose(1, 2).reshape(b, tq, d))
    if return_kv:
      return out, (new_k, new_v)
    return out


class MultiheadAttentionBlock(nn.Module):
  def __init__(self, d_model, nhead, dim_ff, activation="swiglu", rope_base=None):
    super().__init__()
    self.attn = MultiheadAttention(d_model, nhead, rope_base)
    self.pre_attn_ln = RMSNorm(d_model)
    self.post_attn_ln = RMSNorm(d_model)
    self.pre_ff_ln = RMSNorm(d_model)
    self.post_ff_ln = RMSNorm(d_model)
    self.swiglu = activation == "swiglu"
    self.linear1 = nn.Linear(d_model, dim_ff)
    if self.swiglu:
      self.linear1_gate = nn.Linear(d_model, dim_ff)
      self.act = F.silu
    else:
      self.act = get_activation(activation)
    self.linear2 = nn.Linear(dim_ff, d_model)
    self.ffn_chunk_size = None  # set to an int to chunk the FFN over tokens

  def _ff_impl(self, x):
    xn = self.pre_ff_ln(x)
    if self.swiglu:
      x = self.act(self.linear1_gate(xn)) * self.linear1(xn)
    else:
      x = self.act(self.linear1(xn))
    return self.post_ff_ln(self.linear2(x))

  def _ff(self, x):
    # Eager FFN chunking: process tokens in slices so the expanded
    # [tokens, dim_feedforward] activation is never materialized in full.
    if self.ffn_chunk_size is None:
      return self._ff_impl(x)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    out = torch.empty(flat.shape[0], self.linear2.out_features,
                      dtype=flat.dtype, device=flat.device)
    for s in range(0, flat.shape[0], self.ffn_chunk_size):
      out[s:s + self.ffn_chunk_size] = self._ff_impl(flat[s:s + self.ffn_chunk_size])
    return out.reshape(shape)

  def forward(self, q, k=None, v=None, attn_mask=None, rope=None,
              cached_kv=None, return_kv=False):
    q_n = self.pre_attn_ln(q)
    if cached_kv is not None:
      assert k is None and v is None, (
          "k/v must be None when cached_kv is provided.")
      k_n, v_n = None, None
    else:
      k = q if k is None else k
      v = q if v is None else v
      k_n = self.pre_attn_ln(k)
      v_n = self.pre_attn_ln(v)
    attn_res = self.attn(q_n, k_n, v_n, attn_mask, rope=rope,
                          cached_kv=cached_kv, return_kv=return_kv)
    if return_kv:
      attn_out, new_kv = attn_res
    else:
      attn_out = attn_res
    a = self.post_attn_ln(attn_out)
    x = q + a
    x = x + self._ff(x)
    if return_kv:
      return x, new_kv
    return x


class InducedSelfAttentionBlock(nn.Module):
  def __init__(self, d_model, nhead, dim_ff, num_inds, activation="swiglu"):
    super().__init__()
    self.ind_vectors = nn.Parameter(torch.zeros(num_inds, d_model))
    self.mab1 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)
    self.mab2 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)

  def forward(self, src, attn_mask=None, cached_hidden=None, return_hidden=False):
    """Applies induced self-attention, optionally reusing a cached hidden.

    If cached_hidden (the mab1 output from a prior call) is given, mab1 is
    skipped and only mab2 runs; return_hidden returns the freshly computed
    hidden for a later call to reuse.
    """
    if cached_hidden is not None:
      hidden = cached_hidden
    else:
      ind = self.ind_vectors.unsqueeze(0).expand(src.shape[0], -1, -1)
      hidden = self.mab1(ind, src, src, attn_mask=attn_mask)
    out = self.mab2(src, hidden, hidden)
    if return_hidden:
      return out, hidden
    return out


class Encoder(nn.Module):
  def __init__(self, num_blocks, d_model, nhead, dim_ff, activation="swiglu",
               rope_base=100000.0):
    super().__init__()
    # One RoPE per Encoder (mirrors JAX `tf_row.rope.freqs`), shared by all blocks.
    self.rope = RoPE(d_model // nhead, rope_base) if rope_base is not None else None
    self.blocks = nn.ModuleList([
        MultiheadAttentionBlock(d_model, nhead, dim_ff, activation, rope_base)
        for _ in range(num_blocks)
    ])

  def forward(self, x, attn_mask=None, cached_kv=None, return_kv=False):
    """Runs the stacked attention blocks, optionally with a per-block K/V cache.

    cached_kv, if given, is a per-block list of (k, v) that each block uses in
    place of its k/v projections; return_kv collects the freshly computed
    per-block (k, v). At most one of the two is set.
    """
    if cached_kv is not None:
      assert not return_kv, "Cannot both use cached_kv and return_kv."
      for blk, kv in zip(self.blocks, cached_kv):
        x = blk(x, attn_mask=attn_mask, rope=self.rope, cached_kv=kv)
      return x
    if return_kv:
      kvs = []
      for blk in self.blocks:
        x, kv = blk(x, attn_mask=attn_mask, rope=self.rope, return_kv=True)
        kvs.append(kv)
      return x, kvs
    for blk in self.blocks:
      x = blk(x, attn_mask=attn_mask, rope=self.rope)
    return x


class SetTransformer(nn.Module):
  def __init__(self, num_blocks, d_model, nhead, dim_ff, num_inds,
               activation="swiglu"):
    super().__init__()
    self.blocks = nn.ModuleList([
        InducedSelfAttentionBlock(d_model, nhead, dim_ff, num_inds, activation)
        for _ in range(num_blocks)
    ])

  def forward(self, src, attn_mask=None, cached_hidden=None, return_hidden=False):
    """Runs the stacked induced-attention blocks.

    cached_hidden, if given, is a per-block list of cached mab1 outputs (see
    InducedSelfAttentionBlock.forward()); return_hidden=True collects the
    freshly computed per-block hidden reprs for later reuse.
    """
    if cached_hidden is not None:
      assert not return_hidden, "Cannot both use cached_hidden and return_hidden."
      for blk, h in zip(self.blocks, cached_hidden):
        src = blk(src, cached_hidden=h)
      return src
    if return_hidden:
      hiddens = []
      for blk in self.blocks:
        src, h = blk(src, attn_mask=attn_mask, return_hidden=True)
        hiddens.append(h)
      return src, hiddens
    for blk in self.blocks:
      src = blk(src, attn_mask=attn_mask)
    return src


class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dims: List[int], out_dim, activation="gelu"):
    super().__init__()
    self.act = get_activation(activation)
    dims = [in_dim] + list(hidden_dims)
    self.layers = nn.ModuleList()  # only Linears; activation applied between
    for i in range(len(hidden_dims)):
      self.layers.append(nn.Linear(dims[i], dims[i + 1]))
    self.layers.append(nn.Linear(dims[-1], out_dim))

  def forward(self, x):
    for i, lin in enumerate(self.layers):
      x = lin(x)
      if i < len(self.layers) - 1:
        x = self.act(x)
    return x


class OneHotAndLinear(nn.Module):
  def __init__(self, num_classes, embed_dim):
    super().__init__()
    self.num_classes = num_classes
    self.projection = nn.Linear(num_classes, embed_dim)

  def forward(self, y):  # y: [B, T] int
    y_long = y.long()
    y_mapped = torch.where(
        (y_long >= 0) & (y_long < self.num_classes),
        y_long,
        torch.tensor(self.num_classes, device=y.device)
    )
    oh = F.one_hot(y_mapped, self.num_classes + 1).to(self.projection.weight.dtype)
    oh_sliced = oh[..., :self.num_classes]
    return self.projection(oh_sliced)


class CellEmbedder(nn.Module):
  def __init__(self, embed_dim, max_classes, feature_group_size=3, num_freq=32,
               is_classifier=True):
    super().__init__()
    self.embed_dim = embed_dim
    self.fgs = feature_group_size
    self.is_classifier = is_classifier
    in_dim = feature_group_size
    self.register_buffer("fourier_frequencies", torch.zeros(in_dim, num_freq))
    self.register_buffer("fourier_frequencies_cat", torch.zeros(in_dim, num_freq))
    self.in_linear = nn.Linear(num_freq * 2, embed_dim)
    self.in_linear_cat = nn.Linear(num_freq * 2, embed_dim)
    if is_classifier:  # classification: embedding lookup over class ids
      self.y_embedder_lookup = nn.Embedding(max_classes, embed_dim)
    else:  # regression: MLP over the scalar target (y_col_embedder_encoder_nhid=6)
      self.y_embedder_lookup = MLP(1, [6], embed_dim, activation="gelu")
    self.row_chunk_size = None  # chunk the Fourier expansion over rows

  def _group(self, x, d=None):  # x: [B,T,H] -> [B,T,H,G]
    h = x.shape[-1]
    idxs = torch.arange(h, device=x.device)
    stacked = []
    if d is not None:
      # Per-batch wrap-around over each member's ACTIVE feature count d (not the
      # padded width h). Mirrors the JAX `% d_safe` path so zero-padded slots are
      # filled with wrapped real features rather than mixing padding into groups.
      d_safe = torch.clamp(d.to(torch.long), min=1)  # [B]
      for i in range(self.fgs):
        offset = (2 ** i) - 1
        idx = (idxs[None, :] + offset) % d_safe[:, None]            # [B, H]
        idx = idx[:, None, :].expand(x.shape[0], x.shape[1], h)     # [B, T, H]
        stacked.append(torch.gather(x, -1, idx))
    else:
      for i in range(self.fgs):
        offset = (2 ** i) - 1
        stacked.append(x[..., (idxs + offset) % h])
    return torch.stack(stacked, dim=-1)

  def _cell(self, x, cat_mask, d=None):  # [B,t,H] -> [B,t,HC,E] (Fourier expansion + sum over G)
    g = self._group(x, d=d).unsqueeze(-1).float()  # float32 Fourier: args g*freq reach ~30,
    dt = x.dtype                                    # so sin/cos must run in fp32 (matches JAX,
    ff = self.fourier_frequencies.float()           # whose freq params stay float32). Cast the
    ffc = self.fourier_frequencies_cat.float()      # fourier features back to compute dtype before in_linear.
    num_out = self.in_linear(torch.cat([(g * ff).sin(), (g * ff).cos()], dim=-1).to(dt))
    if cat_mask is not None:
      cat_out = self.in_linear_cat(torch.cat([(g * ffc).sin(), (g * ffc).cos()], dim=-1).to(dt))
      cmg = self._group(cat_mask[:, None, :].float(), d=d).bool()[..., None]
      return torch.where(cmg, cat_out, num_out).sum(-2)
    return num_out.sum(-2)

  def forward(self, x, y, train_size, cat_mask=None, d=None):
    # The Fourier expansion materializes [B,T,HC,G,E]; chunk over rows so that
    # huge intermediate never exists in full (rows are independent here).
    if self.row_chunk_size is None:
      cell = self._cell(x, cat_mask, d=d)
    else:
      parts = [self._cell(x[:, s:s + self.row_chunk_size], cat_mask, d=d)
               for s in range(0, x.shape[1], self.row_chunk_size)]
      cell = torch.cat(parts, dim=1)
    if self.is_classifier:
      y_clean = torch.clamp(y.long(), 0, self.y_embedder_lookup.num_embeddings - 1)
      y_emb = self.y_embedder_lookup(y_clean)  # [B,T,E]
    else:
      y_emb = self.y_embedder_lookup(y[..., None].to(cell.dtype))  # scalar -> [B,T,E]
    t = x.shape[1]
    tm = (torch.arange(t, device=x.device)[None, :] < train_size[:, None])[..., None, None]
    out = torch.where(tm, cell + y_emb[:, :, None, :], cell)
    if d is not None:
      # Zero the padded feature columns (cols >= d): the % d wrap above fills them
      # with real features for valid indexing, but they must not enter attention.
      hc = out.shape[2]
      colmask = (torch.arange(hc, device=out.device)[None, :]
                 < d[:, None])[:, None, :, None]  # [B, 1, HC, 1]
      out = torch.where(colmask, out, torch.zeros_like(out))
    return out


class ColEmbedding(nn.Module):
  def __init__(self, d_model, num_blocks, nhead, dim_ff, num_inds):
    super().__init__()
    self.tf_col = SetTransformer(num_blocks, d_model, nhead, dim_ff, num_inds)
    self.out_w = nn.Linear(d_model, d_model)
    self.ln_w = RMSNorm(d_model)
    self.col_chunk_size = None  # chunk the independent column axis (B*HC)

  def _stage(self, src, mask=None, cached_hidden=None, return_hidden=False):
    out = self.tf_col(src, attn_mask=mask, cached_hidden=cached_hidden,
                       return_hidden=return_hidden)
    if return_hidden:
      out, hidden = out
      return self.ln_w(self.out_w(out)), hidden
    return self.ln_w(self.out_w(out))

  def forward(self, x, train_size, *, cached_repr=None, return_repr=False):
    """Transform input table into column-wise embeddings.

    train_size is None exactly when cached_repr is given (decode): cached_repr
    supplies the induced-point hidden, so mab1 and its mask are skipped.
    cached_repr and return_repr are mutually exclusive.
    """
    assert not (cached_repr is not None and return_repr), (
        "Cannot have both cached_repr not None and return_repr True.")
    assert (cached_repr is not None) == (train_size is None), (
        "train_size must be None iff cached_repr is given.")
    b, t, hc, e = x.shape
    src = x.permute(0, 2, 1, 3).reshape(b * hc, t, e)  # [B*HC, T, E]
    cc = self.col_chunk_size

    if cached_repr is not None:  # Decode: reuse cached induced-point hidden.
      if cc is None or src.shape[0] <= cc:
        out = self._stage(src, None, cached_hidden=cached_repr)
      else:
        parts = []
        for s in range(0, src.shape[0], cc):
          chunk_repr = [h[s:s + cc] for h in cached_repr]
          parts.append(self._stage(src[s:s + cc], None, cached_hidden=chunk_repr))
        out = torch.cat(parts, dim=0)
      return out.reshape(b, hc, t, e).permute(0, 2, 1, 3)

    ts = train_size.repeat_interleave(hc)  # [B*HC]
    mask = (torch.arange(t, device=x.device)[None, :] < ts[:, None])[:, None, None, :]

    if return_repr:  # Prefill: also return the induced-point hidden per block.
      if cc is None or src.shape[0] <= cc:
        out, hidden = self._stage(src, mask, return_hidden=True)
      else:
        out_parts = []
        hidden_parts = None
        for s in range(0, src.shape[0], cc):
          o, h = self._stage(src[s:s + cc], mask[s:s + cc], return_hidden=True)
          out_parts.append(o)
          if hidden_parts is None:
            hidden_parts = [[] for _ in h]
          for i, hh in enumerate(h):
            hidden_parts[i].append(hh)
        out = torch.cat(out_parts, dim=0)
        hidden = [torch.cat(parts, dim=0) for parts in hidden_parts]
      out = out.reshape(b, hc, t, e).permute(0, 2, 1, 3)
      return out, hidden

    if cc is None or src.shape[0] <= cc:
      out = self._stage(src, mask)
    else:
      out = torch.cat([self._stage(src[s:s + cc], mask[s:s + cc])
                       for s in range(0, src.shape[0], cc)], dim=0)
    return out.reshape(b, hc, t, e).permute(0, 2, 1, 3)


class RowInteraction(nn.Module):
  def __init__(self, d_model, num_blocks, nhead, dim_ff, num_cls,
               rope_base=100000.0, output_full=True):
    super().__init__()
    self.tf_row = Encoder(num_blocks, d_model, nhead, dim_ff, rope_base=rope_base)
    self.out_ln = RMSNorm(d_model)
    self.num_cls = num_cls
    self.output_full = output_full
    self.row_chunk_size = None  # chunk the independent row axis (B*T)

  def _stage(self, src, mask=None):
    out = self.tf_row(src, attn_mask=mask)
    return self.out_ln(out if self.output_full else out[:, : self.num_cls, :])

  def forward(self, x, d=None):  # x: [B,T,HC,E]
    b, t, hc, e = x.shape
    src = x.reshape(b * t, hc, e)
    # Mask cross-column attention to the valid columns (CLS + d real features);
    # padded columns (>= d + num_cls) must not be attended to. Matches JAX.
    mask = None
    if d is not None:
      d_padded = d.to(torch.long) + self.num_cls  # [B]
      valid = torch.arange(hc, device=x.device)[None, :] < d_padded[:, None]  # [B, HC]
      mask = valid.repeat_interleave(t, dim=0)[:, None, None, :]  # [B*T, 1, 1, HC]
    rc = self.row_chunk_size
    if rc is None or src.shape[0] <= rc:
      out = self._stage(src, mask)
    else:
      out = torch.cat([self._stage(src[s:s + rc],
                                   None if mask is None else mask[s:s + rc])
                       for s in range(0, src.shape[0], rc)], dim=0)
    if self.output_full:
      return out.reshape(b, t, hc, e)
    return out.reshape(b, t, -1)


@dataclasses.dataclass
class QuantizedTensor:
  """Per-tensor symmetric integer quantization of a cached K or V tensor.

  data holds the quantized codes; scale is the per-tensor absmax / max_val
  factor, so the float value is data.to(dtype) * scale.
  """
  data: torch.Tensor
  scale: torch.Tensor

  def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
    """Dequantizes back to a floating-point tensor of the given dtype."""
    return self.data.to(dtype) * self.scale.to(dtype)


# Per dtype: (lo, hi, max_val) clamp range, one code below the dtype's full
# range so -max and +max map symmetrically and dequantization cannot exceed
# the original absmax in magnitude.
_QUANTIZATION_RANGES: Dict[torch.dtype, Tuple[int, int, int]] = {
    torch.int8: (-127, 127, 127),
}


def _quantize_tensor(t: torch.Tensor,
                      dtype: torch.dtype = torch.int8) -> QuantizedTensor:
  """Per-tensor symmetric integer quantization to dtype."""
  if dtype not in _QUANTIZATION_RANGES:
    raise ValueError(f"Unsupported quantization dtype {dtype}; supported: "
                      f"{list(_QUANTIZATION_RANGES.keys())}")
  lo, hi, max_val = _QUANTIZATION_RANGES[dtype]
  absmax = t.abs().amax()
  scale = absmax / max_val
  # Avoid division by zero for all-zero tensors.
  scale = torch.clamp(scale, min=torch.finfo(scale.dtype).tiny)
  data = (t / scale).round().clamp(lo, hi).to(dtype)
  return QuantizedTensor(data=data, scale=scale)


def move_cache_to_device(cache, device):
  """Recursively moves a TabFM.prefill() cache dict to device.

  Handles the nested structure of the cache returned by TabFM.prefill(): a
  dict (top-level col1/col2/icl keys), a list/tuple (per-block K/V pairs,
  per-block induced-point hidden reprs), the ICLearningCache/QuantizedTensor
  dataclasses, and plain tensor leaves. Needed for the
  keep_cache_on_device=False path (CPU-offload between predict calls).
  """
  if isinstance(cache, torch.Tensor):
    return cache.to(device)
  if isinstance(cache, dict):
    return {k: move_cache_to_device(v, device) for k, v in cache.items()}
  if isinstance(cache, list):
    return [move_cache_to_device(v, device) for v in cache]
  if isinstance(cache, tuple):
    return tuple(move_cache_to_device(v, device) for v in cache)
  if dataclasses.is_dataclass(cache):
    kwargs = {f.name: move_cache_to_device(getattr(cache, f.name), device)
              for f in dataclasses.fields(cache)}
    return type(cache)(**kwargs)
  return cache


@dataclasses.dataclass
class ICLearningCache:
  """Per-block ICL K/V cache produced at prefill and reused at decode.

  layer_caches is a per-block list of (k, v) of shape [B, T_prefill, N, D],
  each a QuantizedTensor after quantize(). prefill_train_size is the [B] count
  of valid training rows used to build the decode attention mask; it stays
  full precision after quantize().
  """
  layer_caches: List[Tuple[Any, Any]]
  prefill_train_size: torch.Tensor

  @property
  def prefill_seq_len(self) -> int:
    """Sequence length of the prefill cache, derived from tensor shape."""
    k, _ = self.layer_caches[0]
    data = k.data if isinstance(k, QuantizedTensor) else k
    return data.shape[1]

  def quantize(self, dtype: torch.dtype = torch.int8) -> "ICLearningCache":
    """Returns a copy with the per-block ICL K/V quantized to dtype.

    Only the attention K/V is quantized; prefill_train_size stays full
    precision.
    """
    quantized = [(_quantize_tensor(k, dtype), _quantize_tensor(v, dtype))
                 for k, v in self.layer_caches]
    return ICLearningCache(layer_caches=quantized,
                            prefill_train_size=self.prefill_train_size)


class ICLearning(nn.Module):
  def __init__(self, d_model, num_blocks, nhead, max_classes, dim_ff,
               decoder_hidden, is_classifier=True):
    super().__init__()
    self.tf_icl = Encoder(num_blocks, d_model, nhead, dim_ff, rope_base=None)  # ICL has no RoPE
    self.ln = RMSNorm(d_model)
    self.is_classifier = is_classifier
    if is_classifier:  # one-hot y-encode; decode to per-class logits
      self.y_encoder = OneHotAndLinear(max_classes, d_model)
      self.decoder = MLP(d_model, [decoder_hidden], max_classes)
    else:  # MLP y-encode the scalar target; decode to a single value
      self.y_encoder = MLP(1, [decoder_hidden], d_model)
      self.decoder = MLP(d_model, [decoder_hidden], 1)

  def forward(self, reps, y, train_size, *,
              cache: Optional[ICLearningCache] = None,
              return_cache: bool = False):
    """Forward pass for ICLearning. reps: [B, T, E] row representations.

    train_size is None exactly when cache is given (decode): the attention mask
    is then derived from the cached prefill's train size and sequence length
    rather than this call's, and y is unused. return_cache (prefill only) also
    returns the ICLearningCache for this call alongside the decoded output.
    """
    assert (cache is not None) == (train_size is None), (
        "train_size must be None iff cache is given.")
    b, t, _ = reps.shape

    if cache is not None:  # Decode.
      prefill_seq_len = cache.prefill_seq_len
      tm_ctx = (torch.arange(prefill_seq_len, device=reps.device)[None, :]
                < cache.prefill_train_size[:, None])
      mask = tm_ctx[:, None, None, :]
      out = self.tf_icl(reps, attn_mask=mask, cached_kv=cache.layer_caches)
      return self.decoder(self.ln(out))

    tm = (torch.arange(t, device=reps.device)[None, :] < train_size[:, None])
    if self.is_classifier:
      y_enc = self.y_encoder(y)
    else:
      y_enc = self.y_encoder(y[..., None].to(reps.dtype))
    r = reps + y_enc * tm[..., None]
    mask = tm[:, None, None, :]
    if return_cache:  # Prefill.
      out, kvs = self.tf_icl(r, attn_mask=mask, return_kv=True)
      new_cache = ICLearningCache(layer_caches=kvs, prefill_train_size=train_size)
      return self.decoder(self.ln(out)), new_cache
    out = self.tf_icl(r, attn_mask=mask)
    return self.decoder(self.ln(out))


# Always-on activation-chunk sizes. TabFM runs the whole training fold as one
# in-context sequence, so a single forward materialises activations that grow
# with rows * features and OOM the GPU on large tasks. These sizes split each
# stage's largest activation along an independent axis, bounding peak memory.
# Chunking is exact (identical outputs) and a no-op when an input is smaller
# than the chunk size, and it costs <1% runtime otherwise, so it is always on.
# Sizes are chosen for memory safety on a 40 GB GPU across TabArena-scale tasks.
_ROW_CHUNK_SIZE = 4096   # rows per chunk (Fourier cell embedding + row interaction)
_COL_CHUNK_SIZE = 16     # feature-instances per chunk (column set-transformer)
_FFN_CHUNK_SIZE = 8192   # tokens per chunk (feed-forward expansion in every block)


class TabFM(nn.Module):
  def __init__(self, *, embed_dim=8, max_classes=10, col_num_blocks=2,
               col_nhead=2, col_num_inds=4, row_num_blocks=2, row_nhead=2,
               row_num_cls=2, icl_num_blocks=2, icl_nhead=2, ff_factor=2,
               feature_group_size=3, num_freq=32, decoder_hidden=None,
               is_classifier=True):
    super().__init__()
    self.max_classes = max_classes
    self.is_classifier = is_classifier
    ff = embed_dim * ff_factor
    icl_dim = embed_dim * row_num_cls
    self.cell_embedder = CellEmbedder(embed_dim, max_classes, feature_group_size,
                                      num_freq, is_classifier)
    self.col_embedder = ColEmbedding(embed_dim, col_num_blocks, col_nhead, ff, col_num_inds)
    self.col_embedder_2 = ColEmbedding(embed_dim, col_num_blocks, col_nhead, ff, col_num_inds)
    self.row_interactor = RowInteraction(embed_dim, row_num_blocks, row_nhead, ff,
                                         row_num_cls, output_full=True)
    self.row_interactor_2 = RowInteraction(embed_dim, row_num_blocks, row_nhead, ff,
                                           row_num_cls, output_full=False)
    self.cls_tokens = nn.Parameter(torch.zeros(row_num_cls, embed_dim))
    self.icl_predictor = ICLearning(icl_dim, icl_num_blocks, icl_nhead, max_classes,
                                    icl_dim * ff_factor,
                                    decoder_hidden or icl_dim * 2, is_classifier)

    # Enable activation chunking by default (see the module constants above).
    # The per-block knobs stay settable (e.g. to None) for benchmarking.
    for module in self.modules():
      if hasattr(module, "row_chunk_size"):
        module.row_chunk_size = _ROW_CHUNK_SIZE
      if hasattr(module, "col_chunk_size"):
        module.col_chunk_size = _COL_CHUNK_SIZE
      if hasattr(module, "ffn_chunk_size"):
        module.ffn_chunk_size = _FFN_CHUNK_SIZE

  def forward(self, x, y, train_size, cat_mask=None, d=None):
    # Mirror the JAX model's entry: replace NaN with the -100 sentinel and cast
    # to the compute dtype (JAX: `jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)`).
    # NaN is already imputed in the shared preprocessing, so nan_to_num is a
    # no-op in the normal flow, but it keeps the model robust + JAX-faithful.
    x = torch.nan_to_num(x, nan=-100.0).to(self.cls_tokens.dtype)
    emb = self.cell_embedder(x, y, train_size, cat_mask, d=d)
    emb = self.col_embedder(emb, train_size)
    b, t, _, e = emb.shape
    cls = self.cls_tokens.expand(b, t, -1, -1)
    emb = torch.cat([cls, emb], dim=2)
    emb = self.row_interactor(emb, d=d)
    emb = self.col_embedder_2(emb, train_size)
    reps = self.row_interactor_2(emb, d=d)
    return self.icl_predictor(reps, y, train_size)

  def prefill(self, x, y, cat_mask=None, d=None):
    """Encodes context (training) rows once and returns (logits, cache).

    x is [B, T, H] context rows and y is [B, T] context labels; cat_mask is an
    optional [B, H] categorical-feature mask and d an optional [B] active-
    feature count. Pads the sequence to a multiple of 128 with the -100
    sentinel, derives train_size from the non-sentinel y entries, runs the full
    pipeline while collecting the col-embedder induced-point reprs and the ICL
    encoder per-layer K/V, and unpads the logits to the original length. cache
    is a dict with keys 'col1', 'col2' (per-block induced-point reprs) and
    'icl' (an ICLearningCache).
    """
    x = torch.nan_to_num(x, nan=-100.0).to(self.cls_tokens.dtype)
    y = y.to(self.cls_tokens.dtype)

    t_orig = x.shape[1]
    block_size = 128
    pad_len = ((t_orig - 1) // block_size + 1) * block_size - t_orig
    if pad_len > 0:
      x = F.pad(x, (0, 0, 0, pad_len), value=-100.0)
      y = F.pad(y, (0, pad_len), value=-100.0)

    # All data is training data; train_size counts non-sentinel y entries so
    # externally-padded rows (or the padding just added above) are excluded.
    is_valid = y != -100.0
    train_size = is_valid.sum(dim=-1).to(torch.long)

    cell = self.cell_embedder(x, y, train_size, cat_mask, d=d)
    emb, cache_col1 = self.col_embedder(cell, train_size, return_repr=True)
    b1, t1, _, e = emb.shape
    cls = self.cls_tokens.expand(b1, t1, -1, -1)
    emb = torch.cat([cls, emb], dim=2)
    emb = self.row_interactor(emb, d=d)
    emb, cache_col2 = self.col_embedder_2(emb, train_size, return_repr=True)
    reps = self.row_interactor_2(emb, d=d)
    logits, cache_icl = self.icl_predictor(reps, y, train_size, return_cache=True)

    cache = {"col1": cache_col1, "col2": cache_col2, "icl": cache_icl}
    return logits[:, :t_orig, :], cache

  def decode(self, x, cache, cat_mask=None, d=None):
    """Generates predictions for test rows using a cache from prefill.

    x is [B, T, H] test rows and cache is the dict returned by prefill;
    cat_mask and d are as in prefill. Pads to a multiple of 128, re-runs the
    row-independent cell embedder and row interactors on the test rows, reuses
    the cached col-embedder reprs and ICL per-layer K/V instead of recomputing
    them from the context, and unpads to the original test length. Returns
    [B, T, K_or_1] logits.
    """
    x = torch.nan_to_num(x, nan=-100.0).to(self.cls_tokens.dtype)
    b, t_orig, _ = x.shape

    block_size = 128
    pad_len = ((t_orig - 1) // block_size + 1) * block_size - t_orig
    if pad_len > 0:
      x = F.pad(x, (0, 0, 0, pad_len), value=-100.0)

    # y carries no information for test rows in decode (unlabeled); use the
    # -100 sentinel throughout, matching JAX.
    y = torch.full((b, x.shape[1]), -100.0, dtype=self.cls_tokens.dtype, device=x.device)
    train_size_zero = torch.zeros(b, dtype=torch.long, device=x.device)

    cell = self.cell_embedder(x, y, train_size_zero, cat_mask, d=d)
    emb = self.col_embedder(cell, None, cached_repr=cache["col1"])
    b1, t1, _, e = emb.shape
    cls = self.cls_tokens.expand(b1, t1, -1, -1)
    emb = torch.cat([cls, emb], dim=2)
    emb = self.row_interactor(emb, d=d)
    emb = self.col_embedder_2(emb, None, cached_repr=cache["col2"])
    reps = self.row_interactor_2(emb, d=d)
    out = self.icl_predictor(reps, y, None, cache=cache["icl"])

    return out[:, :t_orig, :]

```

## File: tabfm/src/pytorch/model_test.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from absl import logging
import numpy as np
import torch
from flax import nnx
import jax.numpy as jnp

from tabfm.src.jax.model import TabFM as JaxTabFM, YEmbeddingScheme
from tabfm.src.pytorch import model as PyTorchTabFM
from tabfm.src.hugging_face.torch_convert import convert, jax_params


class PyTorchModelTest(unittest.TestCase):

  def test_pytorch_model_instantiation(self):
    """Verifies that the PyTorch model instantiates with config and runs a forward pass."""
    model = PyTorchTabFM.TabFM(
        embed_dim=16,
        max_classes=10,
        col_num_blocks=2,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=2,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=2,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=3,
        is_classifier=True
    )
    self.assertIsNotNone(model)
    
    # Run dummy forward pass
    x = torch.randn(2, 4, 6) # [B, T, H]
    y = torch.randint(0, 3, (2, 4)) # [B, T]
    train_size = torch.tensor([2, 3]) # [B]
    out = model(x, y, train_size)
    self.assertEqual(out.shape, (2, 4, 10))

  def test_jax_pytorch_parity(self):
    """Verifies JAX vs PyTorch model outputs are numerically equal up to 1e-4."""
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        # 1. Config definitions
        cfg = dict(
            embed_dim=32,
            max_classes=4,
            col_num_blocks=2,
            col_nhead=4,
            col_num_inds=16,
            row_num_blocks=2,
            row_nhead=4,
            row_num_cls=4,
            icl_num_blocks=3,
            icl_nhead=4,
            ff_factor=4,
            feature_group_size=3,
            use_bias=False
        )

        # 2. Instantiate JAX model deterministically (random init)
        jax_model = JaxTabFM(
            loss="cross_entropy" if is_classifier else "rmse",
            activation="swiglu",
            feature_group=True,
            **cfg,
            y_embedding_scheme=YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
            rngs=nnx.Rngs(42),
            dtype=jnp.float32
        )

        # 3. Instantiate PyTorch model with matching config
        torch_model = PyTorchTabFM.TabFM(
            embed_dim=cfg["embed_dim"],
            max_classes=cfg["max_classes"],
            col_num_blocks=cfg["col_num_blocks"],
            col_nhead=cfg["col_nhead"],
            col_num_inds=cfg["col_num_inds"],
            row_num_blocks=cfg["row_num_blocks"],
            row_nhead=cfg["row_nhead"],
            row_num_cls=cfg["row_num_cls"],
            icl_num_blocks=cfg["icl_num_blocks"],
            icl_nhead=cfg["icl_nhead"],
            ff_factor=cfg["ff_factor"],
            feature_group_size=cfg["feature_group_size"],
            is_classifier=is_classifier
        )

        state_dict, missing = convert(jax_params(jax_model), torch_model)
        self.assertEqual(len(missing), 0)
        torch_model.load_state_dict(state_dict, strict=True)
        torch_model.eval()

        # 5. Prepare random input data
        b, t, h = 3, 5, 8
        np.random.seed(123)
        x_np = np.random.normal(size=(b, t, h)).astype(np.float32)
        
        if is_classifier:
          y_np = np.random.randint(0, cfg["max_classes"], size=(b, t)).astype(np.float32)
        else:
          y_np = np.random.normal(size=(b, t)).astype(np.float32)
          
        train_size_np = np.array([2, 3, 4], dtype=np.int32)
        d_np = np.array([5, 6, 7], dtype=np.int32) # active feature counts (d < h)
        cat_mask_np = np.zeros((b, h), dtype=bool)
        cat_mask_np[0, :3] = True
        cat_mask_np[1, :4] = True

        # JAX Inputs
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        train_size_jax = jnp.array(train_size_np)
        d_jax = jnp.array(d_np)
        cat_mask_jax = jnp.array(cat_mask_np)

        # PyTorch Inputs
        x_torch = torch.from_numpy(x_np)
        y_torch = torch.from_numpy(y_np)
        train_size_torch = torch.from_numpy(train_size_np)
        d_torch = torch.from_numpy(d_np)
        cat_mask_torch = torch.from_numpy(cat_mask_np)

        # 6. Forward passes
        # JAX
        jax_out = jax_model(x_jax, y_jax, train_size_jax, cat_mask=cat_mask_jax, d=d_jax)
        jax_out_np = np.asarray(jax_out)

        # PyTorch
        with torch.no_grad():
          torch_out = torch_model(x_torch, y_torch, train_size_torch, cat_mask=cat_mask_torch, d=d_torch)
          torch_out_np = torch_out.numpy()

        # 7. Compare JAX vs PyTorch outputs
        diff = np.abs(jax_out_np - torch_out_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Assert max difference is less than 1e-4
        self.assertLess(
            max_diff,
            1e-4,
            f"Fidelity discrepancy found: max diff = {max_diff}, mean diff = {mean_diff} for is_classifier={is_classifier}"
        )

  def _make_prefill_decode_model(self):
    model = PyTorchTabFM.TabFM(
        embed_dim=32,
        max_classes=5,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=2,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=3,
        is_classifier=True,
    )
    model.eval()
    with torch.no_grad():
      # fourier_frequencies{,_cat} are checkpoint-loaded buffers that default
      # to all-zero at random init, which would make the Fourier cell features
      # (and hence every row's embedding) identically constant regardless of
      # x -- a vacuous test. Seed them so cell embeddings genuinely depend on
      # per-row input, matching what a real (checkpoint-loaded) model does.
      model.cell_embedder.fourier_frequencies.normal_()
      model.cell_embedder.fourier_frequencies_cat.normal_()
      # per_dim_scale also defaults to all-zero; give it a small nonzero init
      # so attention scores are not degenerate.
      for name, p in model.named_parameters():
        if "per_dim_scale" in name:
          p.normal_(std=0.1)
    return model

  def test_prefill_decode_consistency(self):
    """decode(test | prefill(context)) should match forward(concat) on test rows."""
    torch.manual_seed(0)
    b, t, h = 1, 20, 4
    train_len = 10
    max_classes = 5

    model = self._make_prefill_decode_model()

    x = torch.randn(b, t, h)
    y = torch.randint(0, max_classes, (b, t)).float()
    train_size = torch.tensor([train_len])

    with torch.no_grad():
      out_full = model(x, y, train_size)
    out_full_test = out_full[:, train_len:, :]

    x_train, y_train = x[:, :train_len, :], y[:, :train_len]
    x_test = x[:, train_len:, :]

    with torch.no_grad():
      _, cache = model.prefill(x_train, y_train)
      self.assertIn("col1", cache)
      self.assertIn("col2", cache)
      self.assertIn("icl", cache)
      out_decode = model.decode(x_test, cache)

    self.assertEqual(out_decode.shape, out_full_test.shape)
    # Sanity check that the test data isn't in the degenerate "all rows
    # identical" regime (which would make this parity check vacuous).
    self.assertGreater(
        (out_full_test[:, 0] - out_full_test[:, 1]).abs().max().item(), 1e-4
    )
    diff_fp32 = (out_full_test - out_decode).abs()
    max_diff_fp32 = diff_fp32.max().item()
    logging.info(
        "[prefill/decode parity] fp32 max-abs-diff = %.6e", max_diff_fp32
    )
    # fp32 prefill/decode is algebraically exact; gate near machine precision.
    self.assertLess(
        max_diff_fp32, 1e-5,
        f"fp32 prefill/decode vs full-forward diff too large: {max_diff_fp32}"
    )

    # Separately report the bf16 max-abs-diff (not gated -- bf16 has ~1e-2/1e-3
    # relative precision, so this is diagnostic rather than a strict gate).
    model_bf16 = self._make_prefill_decode_model()
    model_bf16.load_state_dict(model.state_dict())
    model_bf16 = model_bf16.to(torch.bfloat16).eval()

    x_bf16 = x.to(torch.bfloat16)
    y_bf16 = y.to(torch.bfloat16)
    x_train_bf16, y_train_bf16 = x_bf16[:, :train_len, :], y_bf16[:, :train_len]
    x_test_bf16 = x_bf16[:, train_len:, :]

    with torch.no_grad():
      out_full_bf16 = model_bf16(x_bf16, y_bf16, train_size)
      out_full_test_bf16 = out_full_bf16[:, train_len:, :]
      _, cache_bf16 = model_bf16.prefill(x_train_bf16, y_train_bf16)
      out_decode_bf16 = model_bf16.decode(x_test_bf16, cache_bf16)

    max_diff_bf16 = (
        out_full_test_bf16.float() - out_decode_bf16.float()
    ).abs().max().item()
    logging.info(
        "[prefill/decode parity] bf16 max-abs-diff = %.6e", max_diff_bf16
    )

  def test_quantized_cache_decode_consistency(self):
    """decode() with an int8-quantized ICLearningCache stays close to fp32."""
    torch.manual_seed(0)
    b, t, h = 1, 20, 4
    train_len = 10
    max_classes = 5

    model = self._make_prefill_decode_model()

    x = torch.randn(b, t, h)
    y = torch.randint(0, max_classes, (b, t)).float()
    x_train, y_train = x[:, :train_len, :], y[:, :train_len]
    x_test = x[:, train_len:, :]

    with torch.no_grad():
      _, cache_fp32 = model.prefill(x_train, y_train)
      out_decode_fp32 = model.decode(x_test, cache_fp32)

      icl_cache = cache_fp32["icl"]
      quantized_icl_cache = icl_cache.quantize()
      for (k_q, v_q) in quantized_icl_cache.layer_caches:
        self.assertIsInstance(k_q, PyTorchTabFM.QuantizedTensor)
        self.assertEqual(k_q.data.dtype, torch.int8)
        self.assertIsInstance(v_q, PyTorchTabFM.QuantizedTensor)
        self.assertEqual(v_q.data.dtype, torch.int8)
      self.assertEqual(
          quantized_icl_cache.prefill_seq_len, icl_cache.prefill_seq_len
      )
      cache_quantized = dict(cache_fp32, icl=quantized_icl_cache)
      out_decode_quantized = model.decode(x_test, cache_quantized)

    self.assertEqual(out_decode_quantized.shape, out_decode_fp32.shape)
    max_diff = (out_decode_fp32 - out_decode_quantized).abs().max().item()
    fp32_scale = out_decode_fp32.abs().max().item()
    rel_diff = max_diff / (fp32_scale + 1e-12)
    logging.info(
        "[quantized cache] fp32-vs-int8 max-abs-diff = %.6e "
        "(rel = %.6e of output scale %.4f)",
        max_diff,
        rel_diff,
        fp32_scale,
    )
    # int8 is lossy; gate on relative deviation (2% >> int8's ~0.8% resolution).
    self.assertLess(
        rel_diff, 0.02,
        f"int8-quantized cache decode diverged too much from fp32: "
        f"rel={rel_diff:.3e} (abs={max_diff:.3e}, scale={fp32_scale:.3e})"
    )

  def test_quantize_unsupported_dtype_raises(self):
    """quantize() rejects a dtype not in the registered quantization ranges."""
    t = torch.randn(2, 3)
    with self.assertRaises(ValueError):
      PyTorchTabFM._quantize_tensor(t, dtype=torch.int16)

  def test_move_cache_to_device_roundtrip(self):
    """move_cache_to_device preserves values and structure (CPU roundtrip)."""
    torch.manual_seed(0)
    b, t, h = 1, 12, 4
    train_len = 8
    max_classes = 5
    model = self._make_prefill_decode_model()
    x = torch.randn(b, t, h)
    y = torch.randint(0, max_classes, (b, t)).float()
    x_train, y_train = x[:, :train_len, :], y[:, :train_len]
    x_test = x[:, train_len:, :]

    with torch.no_grad():
      _, cache = model.prefill(x_train, y_train)
      cache["icl"] = cache["icl"].quantize()
      moved = PyTorchTabFM.move_cache_to_device(cache, "cpu")
      out_direct = model.decode(x_test, cache)
      out_moved = model.decode(x_test, moved)

    self.assertTrue(torch.equal(out_direct, out_moved))
    for (k, v), (k_m, v_m) in zip(
        cache["icl"].layer_caches, moved["icl"].layer_caches
    ):
      self.assertTrue(torch.equal(k.data, k_m.data))
      self.assertTrue(torch.equal(k.scale, k_m.scale))
      self.assertEqual(str(k_m.data.device), "cpu")
      self.assertTrue(torch.equal(v.data, v_m.data))
      self.assertTrue(torch.equal(v.scale, v_m.scale))


if __name__ == "__main__":
  unittest.main()

```

## File: tabfm/src/pytorch/tabfm_v1_0_0.py

```
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import threading
from typing import Any, Dict, Optional
from absl import logging
import torch
from huggingface_hub import PyTorchModelHubMixin, constants, snapshot_download

from tabfm.src.pytorch.model import TabFM

HF_REPO_ID = "google/tabfm-1.0.0-pytorch"

_LOAD_CACHE_LOCK = threading.Lock()
_LOAD_CACHE: Dict[Any, "TabFM_HF"] = {}


class TabFM_HF(
    TabFM,
    PyTorchModelHubMixin,
    repo_url="https://github.com/google-research/tabfm",
    license="other",
):
  """PyTorch TabFM model with HuggingFace Hub support.

  Subclasses TabFM directly (rather than wrapping it) and mixes in
  PyTorchModelHubMixin, keeping the Hugging Face specific loading logic out
  of the plain model class.
  """

  @classmethod
  def _from_pretrained(
      cls,
      *,
      model_id,
      revision,
      cache_dir,
      force_download,
      local_files_only,
      token,
      map_location="cpu",
      strict=True,
      **model_kwargs,
  ):
    subfolder = model_kwargs.pop("subfolder", None)

    def _apply_config(cfg):
      if "is_classifier" not in model_kwargs and "task" in cfg:
        model_kwargs["is_classifier"] = cfg.pop("task") == "classification"
      for key in ("model_type", "version", "framework"):
        cfg.pop(key, None)
      for k, v in cfg.items():
        if k not in model_kwargs:
          model_kwargs[k] = v

    # translate config keys already merged into model_kwargs by from_pretrained()
    _apply_config(model_kwargs)

    if subfolder is None:
      local_id = model_id
    elif os.path.isdir(model_id):
      local_id = os.path.join(model_id, subfolder)
    else:
      base_path = snapshot_download(
          repo_id=model_id,
          revision=revision,
          cache_dir=cache_dir,
          force_download=force_download,
          local_files_only=local_files_only,
          token=token,
          allow_patterns=[f"{subfolder}/**"],
      )
      local_id = os.path.join(base_path, subfolder)

    if subfolder is not None:
      cfg_path = os.path.join(local_id, constants.CONFIG_NAME)
      if os.path.exists(cfg_path):
        with open(cfg_path) as f:
          _apply_config(json.load(f))
      else:
        logging.warning("No config.json found in %s", local_id)

    return super()._from_pretrained(
        model_id=local_id,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        map_location=map_location,
        strict=strict,
        **model_kwargs,
    )


def load(
    model_type: str = "classification",
    checkpoint_path: Optional[str] = None,
    *,
    device: Optional[str] = None,
    dtype: Any = torch.bfloat16,
    use_cache: bool = True,
) -> "TabFM_HF":
  """Loads the PyTorch TabFM v1.0.0 model with pre-trained weights.

  The checkpoint is stored in float32, but the model is designed to run in
  bfloat16 (matching the JAX release's ``dtype=jnp.bfloat16`` compute default),
  with a few internal fp32 upcasts. ``dtype`` casts the model accordingly; pass
  ``None`` to keep the float32 weights.

  ``dtype`` is provided for float32 debugging / quality comparison; the model is
  designed for bfloat16 and this option may be removed in a future release.

  Args:
    model_type: 'classification' or 'regression'.
    checkpoint_path: Local directory or weights file. If None, downloads from
      Hugging Face (google/tabfm-1.0.0-pytorch).
    device: Target device (e.g. 'cuda', 'cpu'). Defaults to 'cpu'.
    dtype: Compute dtype to cast the model to after loading. Defaults to
      bfloat16; pass None to keep the float32 weights.
    use_cache: Reuse a process-wide cached model for identical settings.

  Returns:
    An eval-mode TabFM_HF model with pre-trained weights loaded.
  """
  if model_type not in ("classification", "regression"):
    raise ValueError(
        f"Unsupported model_type: {model_type!r}. "
        "Must be 'classification' or 'regression'."
    )

  cache_key = (model_type, checkpoint_path, device, str(dtype))
  if use_cache:
    _LOAD_CACHE_LOCK.acquire()
  try:
    if use_cache and cache_key in _LOAD_CACHE:
      return _LOAD_CACHE[cache_key]

    if checkpoint_path is None:
      logging.info(
          "Downloading TabFM v1.0.0 PyTorch %s weights from Hugging Face...",
          model_type,
      )
      model = TabFM_HF.from_pretrained(HF_REPO_ID, subfolder=model_type)
    else:
      local_dir = checkpoint_path
      if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Local checkpoint path not found: {local_dir}")
      sub = os.path.join(local_dir, model_type)
      if os.path.isdir(sub):
        local_dir = sub

      if os.path.exists(os.path.join(local_dir, "config.json")):
        model = TabFM_HF.from_pretrained(local_dir)
      else:
        # no config.json: pass is_classifier explicitly
        model = TabFM_HF.from_pretrained(
            local_dir,
            is_classifier=(model_type == "classification"),
        )

    if dtype is not None:
      model = model.to(dtype)  # engage the bf16 compute design (see docstring)

    if device is not None:
      model = model.to(device)
    model.eval()

    if use_cache:
      _LOAD_CACHE[cache_key] = model
    return model
  finally:
    if use_cache:
      _LOAD_CACHE_LOCK.release()

```

