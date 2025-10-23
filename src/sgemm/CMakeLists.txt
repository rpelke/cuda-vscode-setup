add_executable(
    sgemm
    main.cu
    ../benchmark.cu
    benchmark_sgemm.cu
    sgemm_cpu.cpp
    00_sgemm_simple.cu
    01_sgemm_coalesced.cu
    02_sgemm_tiled.cu
    03_sgemm_tiled_2d.cu
    04_sgemm_tiled_2d_vectorized_1.cu
    05_sgemm_tiled_2d_vectorized_2.cu
    06_sgemm_warptiling.cu
    07_tensor_cores.cu
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
    PRIVATE ${PROJECT_SOURCE_DIR}/inc
    PRIVATE ${CUDA_PATH}/include
)
target_link_libraries(sgemm
  PRIVATE ${CUBLAS_LIBRARY} cuda cudart
)
