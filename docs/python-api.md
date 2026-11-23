# Python and JAX API

SpringMV exposes immutable CSR topology separately from edge values. All
numerical arrays use `float32`; CSR arrays use `int32`.

## CSR structure

Construct `CsrStructure` eagerly from CSR row offsets and column indices:

```python
import jax.numpy as jnp

from springmv import CsrStructure

row_offsets = jnp.array([0, 2, 3], dtype=jnp.int32)
column_indices = jnp.array([0, 2, 1], dtype=jnp.int32)

structure = CsrStructure.from_arrays(
    row_offsets,
    column_indices,
    num_cols=3,
)
```

Construction validates the topology, builds its transpose and edge
permutation, and records scheduling metadata. Values deliberately remain
outside the structure so one topology can be reused with changing or trainable
weights.

Structure construction and `structure.transpose()` are preprocessing
operations and cannot run under JAX transformations. Calling `transpose()`
returns a new `CsrStructure` backed by the already-computed transpose:

```python
transposed_structure = structure.transpose()
transposed_values = values[structure.transpose_permutation]
```

The lower-level `csr_transpose` array API is also available:

```python
from springmv import csr_transpose

t_row_offsets, t_column_indices = csr_transpose(
    row_offsets,
    column_indices,
    num_cols=3,
)
```

Supplying `values=` additionally returns transposed values. Temporary CUDA
workspace is managed internally. Set `return_permutation=False` when no edge
permutation is needed.

## Sparse reductions

The four public reductions have the same topology-first shape:

```python
from springmv import csr_logsumexp, csr_maxplus, csr_minplus, csr_spmv

y = csr_spmv(structure, x, values=values)
log_y = csr_logsumexp(structure, log_x, log_values=log_values)
minimum = csr_minplus(structure, x, edge_values=edge_values)
maximum = csr_maxplus(structure, x, edge_values=edge_values)
```

These calls use a valued (weighted) CSR matrix: each stored entry has an
explicit `float32` value. A binary (unweighted) CSR matrix stores only the
sparsity structure and treats every present entry as the multiplicative
identity of the selected semiring.

The value argument is optional in every case:

```python
y = csr_spmv(structure, x)
log_y = csr_logsumexp(structure, log_x)
minimum = csr_minplus(structure, x)
maximum = csr_maxplus(structure, x)
```

Omitting weights dispatches to a distinct unweighted CUDA target; SpringMV
does not allocate or read dummy unit/zero edge values.

All reductions compose with `jax.jit`:

```python
import jax

compiled = jax.jit(
    lambda current_x: csr_logsumexp(
        structure,
        current_x,
        log_values=log_values,
    )
)
result = compiled(log_x)
```

The optional `subwarp_size` argument accepts `0`, `1`, `2`, `4`, `8`, `16`,
or `32`. Zero selects a width automatically from the maximum row length and is
the normal default.

## Batching

The CSR topology remains shared under `jax.vmap`. Dense inputs and edge values
may be mapped independently:

```python
log_x_batch = jnp.stack([log_x, log_x + 0.5])
log_values_batch = jnp.stack([log_values, log_values - 0.25])

# Shared edge values, mapped inputs.
shared_weights = jax.jit(
    jax.vmap(
        lambda current_x: csr_logsumexp(
            structure,
            current_x,
            log_values=log_values,
        )
    )
)(log_x_batch)

# Mapped edge values and inputs.
batched_weights = jax.jit(
    jax.vmap(
        lambda current_values, current_x: csr_logsumexp(
            structure,
            current_x,
            log_values=current_values,
        )
    )
)(log_values_batch, log_x_batch)
```

Vmapped forward reductions use one batch-aware CUDA launch. Batched or ragged
`CsrStructure` objects are not supported.

## Differentiation

Arithmetic and log-sum-exp support first-order JVP, VJP, and `jax.grad`.
Weighted forms differentiate both the dense input and edge values:

```python
primal, tangent = jax.jvp(
    lambda current_values, current_x: csr_spmv(
        structure,
        current_x,
        values=current_values,
    ),
    (values, x),
    (jnp.ones_like(values), jnp.ones_like(x)),
)

values_grad, x_grad = jax.grad(
    lambda current_values, current_x: csr_spmv(
        structure,
        current_x,
        values=current_values,
    ).sum(),
    argnums=(0, 1),
)(values, x)
```

Arithmetic and log-sum-exp expose two input-gradient strategies for both
weighted and unweighted products:

- `input_vjp_strategy="scatter"` traverses original CSR and atomically
  accumulates into the input cotangent. This is the default because it avoids
  indirect edge-value reads.
- `input_vjp_strategy="transpose"` gathers over transposed CSR and reads
  weighted values through `transpose_permutation` when present. It is
  deterministic and can be faster for matrices with high atomic contention.

Both strategies use fused batched CUDA kernels for unweighted products under
`jax.vmap`. Weighted batches currently share edge values and use mapped
per-vector backward kernels.

```python
x_grad = jax.grad(
    lambda current_x: csr_logsumexp(
        structure,
        current_x,
        log_values=log_values,
        input_vjp_strategy="transpose",
    ).sum()
)(log_x)
```

Atomic scatter has a non-deterministic floating-point accumulation order.
Higher-order differentiation is not currently supported.

## Log-sum-exp boundary convention

At extended-real boundary points, log-sum-exp derivatives use a deterministic
symmetric convention:

- Multiple `+inf` terms split the derivative equally.
- If every term in a nonempty row is `-inf`, all terms split it equally.
- Empty rows have zero derivatives.
- A NaN term makes the row output and derivatives associated with that row NaN.

## Sampling

`csr_logsumexp_sample` performs weighted Gumbel-max selection followed by
Boolean propagation from active parents to children:

```python
from springmv import csr_logsumexp_sample

active_parent = jnp.array([True, False], dtype=jnp.bool_)
gumbels = jax.random.gumbel(
    jax.random.key(7),
    (structure.num_nonzeros,),
    dtype=jnp.float32,
)

active_child = csr_logsumexp_sample(
    structure,
    log_values,
    gumbels,
    active_parent,
)
```

The caller supplies already-transformed Gumbel variates. This keeps JAX PRNG
key ownership explicit and reproducible. Weights, Gumbels, and parent activity
can each be shared or independently mapped, provided mapped batch sizes agree:

```python
keys = jax.random.split(jax.random.key(8), 128)
gumbel_batch = jax.vmap(
    lambda key: jax.random.gumbel(
        key,
        (structure.num_nonzeros,),
        dtype=jnp.float32,
    )
)(keys)
active_parent_batch = jnp.broadcast_to(active_parent, (128, 2))

active_child_batch = jax.jit(
    jax.vmap(
        lambda noise, active: csr_logsumexp_sample(
            structure,
            log_values,
            noise,
            active,
        )
    )
)(gumbel_batch, active_parent_batch)
```

Selection reads weights in original CSR order and chooses exactly one edge per
nonempty parent. Exact ties choose the lowest original CSR edge index. NaN
scores are ignored; a row containing only NaNs or no edges selects nothing.
Multiple `+inf` scores and rows containing only `-inf` follow the same tie
rule.

Propagation is a row-owned Boolean gather over the stored transpose. It reads
neither edge weights nor `transpose_permutation`, and neither sampling stage
uses atomics. Sampling is intentionally non-differentiable.

## Transformation support

| Operation | Eager | `jax.jit` | `jax.jvp` | VJP/`jax.grad` | `jax.vmap` |
|---|---:|---:|---:|---:|---:|
| Arithmetic CSR reduction | Yes | Yes | Yes | Yes | Yes |
| Log-sum-exp CSR reduction | Yes | Yes | Yes | Yes | Yes |
| Min-plus and max-plus | Yes | Yes | No | No | Yes |
| Log-sum-exp sampling | Yes | Yes | No | No | Yes |
| CSR preprocessing | Yes | No | No | No | No |
