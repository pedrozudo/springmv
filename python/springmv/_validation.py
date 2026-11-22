"""Shared validation helpers for the SpringMV Python API."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def as_vector(value: Any, name: str, dtype: np.dtype[Any]) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    return array


def as_structure_array(value: Any, name: str) -> jax.Array:
    return as_vector(value, name, np.dtype(np.int32))


def as_float_vector(value: Any, name: str) -> jax.Array:
    return as_vector(value, name, np.dtype(np.float32))


def require_eager_cuda(*arrays: jax.Array, operation: str) -> None:
    if any(isinstance(array, jax.core.Tracer) for array in arrays):
        raise TypeError(
            f"springmv.{operation} is a preprocessing operation and cannot "
            "run under JAX transformations"
        )
    devices: set[jax.Device] = set()
    for array in arrays:
        devices.update(array.devices())
    if len(devices) != 1:
        raise ValueError("all preprocessing inputs must be on one CUDA device")
    device = next(iter(devices))
    platform_version = getattr(device.client, "platform_version", "").lower()
    if device.platform != "gpu" or "cuda" not in platform_version:
        raise ValueError(f"springmv.{operation} requires an NVIDIA CUDA device")
