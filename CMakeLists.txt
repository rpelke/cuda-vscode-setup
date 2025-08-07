cmake_minimum_required(VERSION 3.18)
project(add_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_TOOLKIT_ROOT_DIR "" CACHE PATH
    "Path to your CUDA Toolkit (e.g. /usr/local/cuda-12.4)")

if (CMAKE_CUDA_TOOLKIT_ROOT_DIR STREQUAL "")
  message(FATAL_ERROR
    "You must specify the CUDA root directory:\n"
    "  cmake -DCMAKE_CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4 ..")
endif()

set(ENV{CUDA_HOME} "${CMAKE_CUDA_TOOLKIT_ROOT_DIR}")
set(ENV{PATH} "${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/lib64:$ENV{LD_LIBRARY_PATH}")

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS_DEBUG
    "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")

add_executable(
    sgemm
    main.cu
    src/sgemm_cpu.cpp
    src/sgemm_simple.cu
    src/sgemm_coalesced.cu
    src/sgemm_tiled.cu
    src/sgemm_tiled_2d.cu
)
target_include_directories(
    sgemm
    PRIVATE inc
)
install(
    TARGETS sgemm
    DESTINATION bin
)
