"""Registration of SpringMV's typed CUDA FFI handlers."""

from jax import ffi as jffi

from . import _springmv


for _name, _target in _springmv.registrations().items():
    jffi.register_ffi_target(_name, _target, platform="CUDA")
