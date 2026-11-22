cmake_minimum_required(VERSION 3.24)

project(SpringMV VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)

option(SPRINGMV_BUILD_TESTS "Build SpringMV correctness tests" ON)
option(SPRINGMV_BUILD_BENCHMARKS "Build SpringMV benchmarks" ON)
option(SPRINGMV_BUILD_JAX "Build the SpringMV JAX extension" OFF)

find_package(CUDAToolkit REQUIRED)

add_library(
    springmv
    src/csr_common.cpp
    src/csr_spmv.cu
    src/csr_spmv_backward.cu
    src/csr_logsumexp.cu
    src/csr_logsumexp_backward.cu
    src/csr_tropical.cu
    src/csr_transpose.cu)
add_library(SpringMV::springmv ALIAS springmv)

target_compile_features(springmv PUBLIC cxx_std_17 cuda_std_17)
target_include_directories(
    springmv
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(springmv PUBLIC CUDA::cudart)
set_target_properties(
    springmv
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(
        springmv PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()

if(SPRINGMV_BUILD_TESTS)
    enable_testing()
    add_executable(springmv_tests tests/test_csr_spmv.cpp)
    target_compile_features(springmv_tests PRIVATE cxx_std_17)
    target_link_libraries(
        springmv_tests PRIVATE SpringMV::springmv CUDA::cudart CUDA::cusparse)
    add_test(NAME springmv.csr_spmv COMMAND springmv_tests)

    add_executable(
        springmv_spmv_backward_tests tests/test_csr_spmv_backward.cpp)
    target_compile_features(
        springmv_spmv_backward_tests PRIVATE cxx_std_17)
    target_link_libraries(
        springmv_spmv_backward_tests
        PRIVATE SpringMV::springmv CUDA::cudart)
    add_test(
        NAME springmv.csr_spmv_backward
        COMMAND springmv_spmv_backward_tests)

    add_executable(
        springmv_logsumexp_tests tests/test_csr_logsumexp.cpp)
    target_compile_features(springmv_logsumexp_tests PRIVATE cxx_std_17)
    target_link_libraries(
        springmv_logsumexp_tests
        PRIVATE SpringMV::springmv CUDA::cudart CUDA::cusparse)
    add_test(NAME springmv.csr_logsumexp COMMAND springmv_logsumexp_tests)

    add_executable(
        springmv_logsumexp_backward_tests
        tests/test_csr_logsumexp_backward.cpp)
    target_compile_features(
        springmv_logsumexp_backward_tests PRIVATE cxx_std_17)
    target_link_libraries(
        springmv_logsumexp_backward_tests
        PRIVATE SpringMV::springmv CUDA::cudart)
    add_test(
        NAME springmv.csr_logsumexp_backward
        COMMAND springmv_logsumexp_backward_tests)

    add_executable(
        springmv_tropical_tests tests/test_csr_tropical.cpp)
    target_compile_features(springmv_tropical_tests PRIVATE cxx_std_17)
    target_link_libraries(
        springmv_tropical_tests
        PRIVATE SpringMV::springmv CUDA::cudart)
    add_test(NAME springmv.csr_tropical COMMAND springmv_tropical_tests)

    add_executable(
        springmv_transpose_tests tests/test_csr_transpose.cpp)
    target_compile_features(springmv_transpose_tests PRIVATE cxx_std_17)
    target_link_libraries(
        springmv_transpose_tests
        PRIVATE SpringMV::springmv CUDA::cudart CUDA::cusparse)
    add_test(NAME springmv.csr_transpose COMMAND springmv_transpose_tests)
endif()

if(SPRINGMV_BUILD_BENCHMARKS)
    add_executable(
        springmv_benchmark benchmarks/benchmark_csr_spmv.cpp)
    target_compile_features(springmv_benchmark PRIVATE cxx_std_17)
    target_link_libraries(
        springmv_benchmark
        PRIVATE SpringMV::springmv CUDA::cudart CUDA::cusparse)
endif()

install(
    TARGETS springmv
    EXPORT SpringMVTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT springmv_development
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT springmv_development
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT springmv_development
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT springmv_development)
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT springmv_development)
install(
    EXPORT SpringMVTargets
    NAMESPACE SpringMV::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SpringMV
    COMPONENT springmv_development)

if(SPRINGMV_BUILD_JAX)
    add_subdirectory(python)
endif()

