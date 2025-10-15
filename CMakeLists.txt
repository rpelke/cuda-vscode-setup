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
    src/sgemm_cpu.cpp
    src/softmax_cpu.cpp
    src/00_sgemm_simple.cu
    src/01_sgemm_coalesced.cu
    src/02_sgemm_tiled.cu
    src/03_sgemm_tiled_2d.cu
    src/04_sgemm_tiled_2d_vectorized_1.cu
    src/05_sgemm_tiled_2d_vectorized_2.cu
    src/06_sgemm_warptiling.cu
    src/07_tensor_cores.cu
    src/08_softmax_simple.cu
    src/09_softmax_block_sum.cu
    src/10_softmax_block_sum_binary.cu
)

if(CMAKE_ENABLE_TRACING)
  message(STATUS "CUDA tracing enabled: adding -lineinfo and -Xptxas=-v")
  target_compile_options(sgemm PRIVATE
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
install(
    TARGETS sgemm
    DESTINATION bin
)
