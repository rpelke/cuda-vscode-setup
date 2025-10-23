add_executable(
    softmax
    main.cu
    ../benchmark.cu
    benchmark_softmax.cu
    softmax_cpu.cpp
    08_softmax_simple.cu
    09_softmax_block_sum.cu
    10_softmax_block_sum_binary.cu
    11_softmax_binary_non_divergent.cu
    12_softmax_sequential_access.cu
    12_softmax_warp_shuffle.cu
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
