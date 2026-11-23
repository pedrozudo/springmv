find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

execute_process(
    COMMAND
        "${Python_EXECUTABLE}" -c
        "import jax; print(jax.ffi.include_dir(), end='')"
    RESULT_VARIABLE SPRINGMV_JAX_INCLUDE_RESULT
    OUTPUT_VARIABLE SPRINGMV_JAX_INCLUDE_DIR
    ERROR_VARIABLE SPRINGMV_JAX_INCLUDE_ERROR)
if(NOT SPRINGMV_JAX_INCLUDE_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "Could not determine the JAX FFI include directory: "
        "${SPRINGMV_JAX_INCLUDE_ERROR}")
endif()

nanobind_add_module(
    _springmv
    src/module.cpp
    src/ffi_reductions.cpp
    src/ffi_logsumexp_backward.cpp
    src/ffi_sampling.cpp
    src/ffi_spmv_backward.cpp
    src/ffi_transpose.cpp)
target_compile_features(_springmv PRIVATE cxx_std_17)
target_include_directories(
    _springmv
    PRIVATE
        "${SPRINGMV_JAX_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(_springmv PRIVATE SpringMV::springmv CUDA::cudart)

install(
    TARGETS _springmv
    LIBRARY DESTINATION springmv COMPONENT python
    RUNTIME DESTINATION springmv COMPONENT python)
