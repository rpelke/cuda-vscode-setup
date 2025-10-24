cmake_minimum_required(VERSION 3.18)

project(mymatmul LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is not set")
endif()

find_package(Python3 COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(PY_PACKAGE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../.venv/lib/python${PYTHON_VERSION}/site-packages)

message(STATUS "Use Python: ${PYTHON_VERSION}")
find_package(pybind11 CONFIG REQUIRED PATHS ${PY_PACKAGE_DIR}/pybind11/share/cmake/pybind11)

set(TORCH_INCLUDE_DIRS
  ${PY_PACKAGE_DIR}/torch/include
  ${PY_PACKAGE_DIR}/torch/include/torch/csrc/api/include
)

set(TORCH_LIB_DIR ${PY_PACKAGE_DIR}/torch/lib)

pybind11_add_module(mymatmul SHARED
    main.cu
    ../../src/sgemm/07_sgemm_tensor_cores.cu
)
include_directories(mymatmul PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_PATH}/include
    ../../inc
)

find_library(TORCH_CPU torch_cpu HINTS ${TORCH_LIB_DIR} REQUIRED)
find_library(C10 c10 HINTS ${TORCH_LIB_DIR} REQUIRED)
find_library(TORCH_PYTHON torch_python HINTS ${TORCH_LIB_DIR} REQUIRED)
message(STATUS "TORCH_CPU = ${TORCH_CPU}")
message(STATUS "C10 = ${C10}")
message(STATUS "TORCH_PYTHON = ${TORCH_PYTHON}")
target_link_libraries(mymatmul PRIVATE
    pybind11::module
    ${TORCH_CPU} ${C10} ${TORCH_PYTHON}
    cuda cudart
)
install(TARGETS mymatmul DESTINATION ${PY_PACKAGE_DIR})
