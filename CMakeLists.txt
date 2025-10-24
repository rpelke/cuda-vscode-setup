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

add_subdirectory(src/sgemm)
add_subdirectory(src/softmax)

add_executable(
    benchmark_cublas_kernels
    src/benchmark_cublas_kernels.cu
    src/benchmark.cu
)

if(CMAKE_ENABLE_TRACING)
  message(STATUS "CUDA tracing enabled: adding -lineinfo and -Xptxas=-v")
  target_compile_options(benchmark_cublas_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  )
endif()

target_include_directories(
    benchmark_cublas_kernels
    PRIVATE inc
    PRIVATE ${CUDA_PATH}/include
)
target_link_libraries(benchmark_cublas_kernels
  PRIVATE ${CUBLAS_LIBRARY} cuda cudart
)
install(
    TARGETS sgemm softmax benchmark_cublas_kernels
    DESTINATION bin
)
