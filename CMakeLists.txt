cmake_minimum_required(VERSION 3.18)
project(add_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS_DEBUG
    "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")

option(CMAKE_ENABLE_TRACING "Enable CUDA lineinfo for profiling/tracing" OFF)

add_executable(
    sgemm
    main.cu
    src/sgemm_cpu.cpp
    src/sgemm_simple.cu
    src/sgemm_coalesced.cu
    src/sgemm_tiled.cu
    src/sgemm_tiled_2d.cu
    src/sgemm_tiled_2d_vectorized_1.cu
    src/sgemm_tiled_2d_vectorized_2.cu
    src/sgemm_warptiling.cu
)

if(CMAKE_ENABLE_TRACING)
  message(STATUS "CUDA tracing enabled: adding -lineinfo and -Xptxas=-v")
  target_compile_options(sgemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    #$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O0>
  )
endif()

message(STATUS "CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")

target_include_directories(
    sgemm
    PRIVATE inc
)
install(
    TARGETS sgemm
    DESTINATION bin
)
