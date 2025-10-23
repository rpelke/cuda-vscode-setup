cmake_minimum_required(VERSION 3.18)
project(add_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is not set")
endif()

set(CMAKE_CUDA_FLAGS_DEBUG
    "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")

option(CMAKE_ENABLE_TRACING "Enable CUDA lineinfo for profiling/tracing" OFF)

find_library(CUBLAS_LIBRARY cublas ${CUDA_PATH}/lib64)

add_executable(
    sgemm
    main.cu
    src/benchmark.cu
    src/sgemm/sgemm_cpu.cpp
    src/sgemm/00_sgemm_simple.cu
    src/sgemm/01_sgemm_coalesced.cu
    src/sgemm/02_sgemm_tiled.cu
    src/sgemm/03_sgemm_tiled_2d.cu
    src/sgemm/04_sgemm_tiled_2d_vectorized_1.cu
    src/sgemm/05_sgemm_tiled_2d_vectorized_2.cu
    src/sgemm/06_sgemm_warptiling.cu
    src/sgemm/07_tensor_cores.cu
    src/softmax/softmax_cpu.cpp
    src/softmax/08_softmax_simple.cu
    src/softmax/09_softmax_block_sum.cu
    src/softmax/10_softmax_block_sum_binary.cu
    src/softmax/11_softmax_binary_non_divergent.cu
    src/softmax/12_softmax_sequential_access.cu
    src/softmax/12_softmax_warp_shuffle.cu
)

add_executable(
    benchmark_cublas_kernels
    src/benchmark_cublas_kernels.cu
    src/benchmark.cu
    src/sgemm/sgemm_cpu.cpp
    src/sgemm/00_sgemm_simple.cu
    src/sgemm/01_sgemm_coalesced.cu
    src/sgemm/02_sgemm_tiled.cu
    src/sgemm/03_sgemm_tiled_2d.cu
    src/sgemm/04_sgemm_tiled_2d_vectorized_1.cu
    src/sgemm/05_sgemm_tiled_2d_vectorized_2.cu
    src/sgemm/06_sgemm_warptiling.cu
    src/sgemm/07_tensor_cores.cu
    src/sgemm/07_tensor_cores.cu
    src/softmax/softmax_cpu.cpp
    src/softmax/08_softmax_simple.cu
    src/softmax/09_softmax_block_sum.cu
    src/softmax/10_softmax_block_sum_binary.cu
    src/softmax/11_softmax_binary_non_divergent.cu
    src/softmax/12_softmax_sequential_access.cu
    src/softmax/12_softmax_warp_shuffle.cu
)

if(CMAKE_ENABLE_TRACING)
  message(STATUS "CUDA tracing enabled: adding -lineinfo and -Xptxas=-v")
  target_compile_options(sgemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  )
  target_compile_options(benchmark_cublas_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  )
endif()

target_include_directories(
    sgemm
    PRIVATE inc
    PRIVATE ${CUDA_PATH}/include
)
target_link_libraries(sgemm
  PRIVATE ${CUBLAS_LIBRARY} cuda cudart
)
target_include_directories(
    benchmark_cublas_kernels
    PRIVATE inc
    PRIVATE ${CUDA_PATH}/include
)
target_link_libraries(benchmark_cublas_kernels
  PRIVATE ${CUBLAS_LIBRARY} cuda cudart
)
install(
    TARGETS sgemm benchmark_cublas_kernels
    DESTINATION bin
)
