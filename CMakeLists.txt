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

add_executable(
    add_cuda
    add.cu
)

set_target_properties(add_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE OFF
)

install(
    TARGETS add_cuda
    DESTINATION bin
)
