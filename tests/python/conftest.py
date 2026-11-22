# Copyright 2026 SpringMV contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import jax
import pytest


def _has_cuda_device() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if _has_cuda_device():
        return
    marker = pytest.mark.skip(
        reason="SpringMV's Python tests require an NVIDIA CUDA device"
    )
    for item in items:
        item.add_marker(marker)
