# Testing and numerical validation

SpringMV has Python/JAX and native CUDA test suites. The Python suite exercises
the public interface and JAX transformations; the native suite validates the
individual C++/CUDA entry points and stream behavior.

## Python tests

Install the package with test dependencies and run:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The JAX tests cover:

- Eager and `jax.jit` execution.
- Shared and independently mapped inputs and edge values under `jax.vmap`.
- First-order JVP, VJP, `grad`, and `value_and_grad` for arithmetic and
  log-sum-exp.
- JVP/VJP adjoint identities and finite-difference checks.
- Dynamic `CsrStructure` arguments.
- Duplicate indices, rectangular and empty structures, infinities, and NaNs.
- Sampling with independently shared or mapped weights, Gumbels, and parent
  activity.
- Sampling tie and special-value semantics plus an empirical categorical
  distribution check.

## Native tests

Configure and run the native suite with:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
ctest --test-dir build --output-on-failure
```

The native suites cover:

- Every compiled subwarp width and automatic width selection.
- Weighted/unweighted equivalence using unit arithmetic or zero log weights.
- Empty rows, irregular row lengths, duplicate columns, infinities, NaNs, and
  extreme finite values.
- Non-default CUDA streams.
- Arithmetic and log-sum-exp JVPs, input VJPs, and edge-value VJPs.
- Min-plus/max-plus identities, duality, and permutation invariance.
- Batched forward kernels with shared or mapped operands.
- Independent sampling selection and propagation stages.
- Sampling ties, inactive parents, duplicate edges, empty parents/children,
  and special values.

## Reference oracles

When system cuSPARSE development libraries are available, three additional
native executables are built:

- Arithmetic SpMV compares directly with cuSPARSE.
- Moderate finite log-sum-exp cases use a transformed double-precision
  cuSPARSE reference, complemented by a stable long-double oracle, analytic
  edge cases, and metamorphic properties.
- Weighted and structure-only transpose compares with cuSPARSE CSR-to-CSC and
  validates deterministic ordering, edge permutations, rectangular matrices,
  duplicates, and empty rows/columns.

cuSPARSE is not a SpringMV runtime dependency. If `CUDA::cusparse` is absent,
these three executables are omitted while all independent native and Python
tests remain enabled. The same path can be tested explicitly with:

```bash
cmake -S . -B build-no-cusparse -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DSPRINGMV_ENABLE_CUSPARSE_VALIDATION=OFF
cmake --build build-no-cusparse
ctest --test-dir build-no-cusparse --output-on-failure
```

## Numerical conventions

Arithmetic follows ordinary `float32` accumulation. Log-sum-exp uses an online
max/scaled-sum reduction for stability. Empty log-sum-exp rows produce `-inf`.
At derivative boundary points, multiple `+inf` terms split cotangents equally,
as do all terms in a nonempty all-`-inf` row. Empty rows have zero derivatives;
NaNs propagate through numerical reductions and their derivatives.

Min-plus empty rows produce `+inf`; max-plus empty rows produce `-inf`. Both
tropical reductions propagate NaNs.

Sampling differs deliberately: NaN scores are ignored, all-NaN and empty rows
select nothing, and exact ties choose the lowest original CSR edge index. This
produces one winner rather than the all-ties behavior of an equality mask
against a row maximum.
