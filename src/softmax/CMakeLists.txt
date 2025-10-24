add_executable(
    softmax
    main.cu
    ../benchmark.cu
    benchmark_softmax.cu
    softmax_cpu.cpp
    00_softmax_simple.cu
    01_softmax_block_sum.cu
    02_softmax_block_sum_binary.cu
    03_softmax_binary_non_divergent.cu
    04_softmax_warp_shuffle.cu
)

if(CMAKE_ENABLE_TRACING)
  message(STATUS "CUDA tracing enabled: adding -lineinfo and -Xptxas=-v")
  target_compile_options(softmax PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  )
endif()

target_include_directories(
    softmax
    PRIVATE ${PROJECT_SOURCE_DIR}/inc
    PRIVATE ${CUDA_PATH}/include
)
target_link_libraries(softmax
  PRIVATE ${CUBLAS_LIBRARY} cuda cudart
)
