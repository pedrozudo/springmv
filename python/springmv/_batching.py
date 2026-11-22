"""Shared batching helpers for SpringMV custom primitives."""

from __future__ import annotations

from collections.abc import Iterable

import jax
import jax.numpy as jnp
from jax.extend import core


def sequential_primitive_batcher(
    primitive: core.Primitive,
    batched_args,
    batch_dimensions,
    *,
    batchable_indices: Iterable[int],
    operation_name: str,
    **params,
):
    """Batch a single-vector primitive with an explicit ``lax.map``.

    CSR topology arrays are deliberately not batchable: a mapped SpringMV
    operation always shares one sparse structure across the batch.  Numeric
    operands may independently be shared or mapped.
    """
    batchable = frozenset(batchable_indices)
    mapped = [
        index
        for index, dimension in enumerate(batch_dimensions)
        if dimension is not None
    ]
    unsupported = [index for index in mapped if index not in batchable]
    if unsupported:
        raise NotImplementedError(
            f"{operation_name} requires one shared CsrStructure; "
            "batching CSR topology arrays is not supported"
        )
    if not mapped:
        return primitive.bind(*batched_args, **params), None

    batch_sizes = {
        batched_args[index].shape[batch_dimensions[index]] for index in mapped
    }
    if len(batch_sizes) != 1:
        raise ValueError("mapped SpringMV operands must have equal batch sizes")

    mapped_operands = tuple(
        jnp.moveaxis(batched_args[index], batch_dimensions[index], 0)
        for index in mapped
    )

    def apply_one(slices):
        arguments = list(batched_args)
        for index, value in zip(mapped, slices, strict=True):
            arguments[index] = value
        return primitive.bind(*arguments, **params)

    return jax.lax.map(apply_one, mapped_operands), 0
