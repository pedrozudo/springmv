# SpringMV

SpringMV is a CUDA-accelerated JAX library for sparse CSR reductions over
multiple semirings. Its kernels run through JAX FFI without host copies and
compose with `jax.jit` and `jax.vmap`.

| Operation | Weighted reduction | Unweighted reduction |
|---|---|---|
| Arithmetic | `sum(values * x)` | `sum(x)` |
| Log-sum-exp | `logsumexp(log_values + log_x)` | `logsumexp(log_x)` |
| Min-plus | `min(edge_values + x)` | `min(x)` |
| Max-plus | `max(edge_values + x)` | `max(x)` |

SpringMV also provides atomic-free Gumbel-max sampling for weighted
log-sum-exp layers. Arithmetic and log-sum-exp support first-order forward- and
reverse-mode differentiation. Values are `float32`, CSR indices are `int32`,
and the current backend is NVIDIA CUDA.

## Installation

SpringMV requires Python 3.10 or newer, a CUDA-enabled JAX installation, CMake
3.24 or newer, a C++17 compiler, and the NVIDIA CUDA Toolkit. Ninja is
recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "jax[cuda13]"
python -m pip install -e .
```

The extension is compiled locally and therefore needs `nvcc`, CUDA Runtime,
and CUB. It does not link cuSPARSE. If several CUDA toolkits are installed,
select one explicitly:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS='-DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_ARCHITECTURES=native' \
python -m pip install -e .
```

## Quick start

`CsrStructure` stores reusable topology and preprocessing metadata, but not
edge values. Construct it eagerly, then reuse it with changing or trainable
weights:

```python
import jax
import jax.numpy as jnp

from springmv import CsrStructure, csr_logsumexp

row_offsets = jnp.array([0, 2, 3], dtype=jnp.int32)
column_indices = jnp.array([0, 2, 1], dtype=jnp.int32)

structure = CsrStructure.from_arrays(
    row_offsets,
    column_indices,
    num_cols=3,
)

log_values = jnp.array([0.2, -0.4, 0.7], dtype=jnp.float32)
log_x = jnp.array([-1.0, -2.0, -3.0], dtype=jnp.float32)

weighted = csr_logsumexp(
    structure,
    log_x,
    log_values=log_values,
)

# Distinct unweighted kernel: no dummy edge-value array is stored or read.
unweighted = csr_logsumexp(structure, log_x)

compiled = jax.jit(lambda x: csr_logsumexp(structure, x))
compiled_result = compiled(log_x)
```

The other reductions follow the same interface:

```python
from springmv import csr_maxplus, csr_minplus, csr_spmv

y = csr_spmv(structure, log_x, values=log_values)
minimum = csr_minplus(structure, log_x, edge_values=log_values)
maximum = csr_maxplus(structure, log_x, edge_values=log_values)
```

Omit `values` or `edge_values` to use their unweighted kernels.

## Batching and differentiation

The topology stays shared under `jax.vmap`. Inputs and edge values can be
mapped independently:

```python
log_x_batch = jnp.stack([log_x, log_x + 0.5])

batched = jax.jit(
    jax.vmap(
        lambda current_x: csr_logsumexp(
            structure,
            current_x,
            log_values=log_values,
        )
    )
)(log_x_batch)
```

Arithmetic and log-sum-exp differentiate both inputs and edge values:

```python
log_values_grad, log_x_grad = jax.grad(
    lambda current_values, current_x: csr_logsumexp(
        structure,
        current_x,
        log_values=current_values,
    ).sum(),
    argnums=(0, 1),
)(log_values, log_x)
```

| Operation | `jax.jit` | `jax.jvp` | VJP/`jax.grad` | `jax.vmap` |
|---|---:|---:|---:|---:|
| Arithmetic CSR reduction | Yes | Yes | Yes | Yes |
| Log-sum-exp CSR reduction | Yes | Yes | Yes | Yes |
| Min-plus and max-plus | Yes | No | No | Yes |
| Log-sum-exp sampling | Yes | No | No | Yes |
| CSR preprocessing | No | No | No | No |

Higher-order differentiation is not currently supported.

## Sampling

Sampling accepts caller-generated Gumbel noise, selects exactly one edge per
nonempty parent, and propagates active selections to a Boolean child mask:

```python
from springmv import csr_logsumexp_sample

active_parent = jnp.array([True, False], dtype=jnp.bool_)
gumbels = jax.random.gumbel(
    jax.random.key(7),
    (structure.num_nonzeros,),
    dtype=jnp.float32,
)

active_child = jax.jit(
    lambda noise: csr_logsumexp_sample(
        structure,
        log_values,
        noise,
        active_parent,
    )
)(gumbels)
```

Weights, Gumbels, and parent activity may each be shared or independently
mapped. Selection runs over original CSR; propagation is an atomic-free
Boolean gather over the stored transpose. Exact ties choose the lowest
original edge index. Sampling is intentionally non-differentiable.

## Testing and benchmarking

Install test dependencies and run the Python suite:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Run the JAX and sampling benchmarks with:

```bash
python benchmarks/benchmark_jax.py
python benchmarks/benchmark_sampling.py
```

The native library and tests can be built separately:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
ctest --test-dir build --output-on-failure
```

cuSPARSE is optional and used only for three native reference-test executables
and the native comparison benchmark. Those targets are omitted automatically
when cuSPARSE is unavailable.

## Further documentation

- [Python and JAX API](docs/python-api.md)
- [Benchmark methodology and options](docs/benchmarks.md)
- [Testing and numerical validation](docs/testing.md)
- [Low-level C++/CUDA API](docs/cpp-api.md)

## Provenance and licensing

SpringMV is licensed under BSD-3-Clause. Its classical CSR row/subwarp
scheduling is adapted from Ginkgo v1.11.0, commit
`4c77a5d826bd1597fd093f39c47688f8cc735690`, specifically
`common/cuda_hip/matrix/csr_kernels.template.cpp`. The adapted source files
retain the Ginkgo copyright and provenance information.
