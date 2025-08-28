#!/bin/bash

if [ -z "$CUDA_PATH" ]; then
    echo "ERROR: CUDA_PATH is not set!"
    exit 1
fi

FILE="inc/sgemm_kernel_dimension.cuh"
FILE_BACKUP=$(cat "$FILE")

PARAMS=()

TM_VALUES=(1 2 4 6 8)
TN_VALUES=(1 2 4 6 8)
BK_VALUES=(8 16 32 64)
XY_VALUES=(1 2 3 4)

# Create configs
for TM in "${TM_VALUES[@]}"; do
  for TN in "${TN_VALUES[@]}"; do
    for BK in "${BK_VALUES[@]}"; do
      for X in "${XY_VALUES[@]}"; do
        for Y in "${XY_VALUES[@]}"; do
          BN=$((TN * BK))
          BM=$((TM * BK))

          WN=$((X * TN))
          WM=$((Y * TM))
          
          if [ "$WN" -eq 0 ] || [ "$WM" -eq 0 ]; then
            continue
          fi
          if [ $((BN % WN)) -ne 0 ] || [ $((BM % WM)) -ne 0 ]; then
            continue
          fi

          PARAMS+=("$TM $TN $WM $WN $BM $BN $BK")
          echo "$TM $TN $WM $WN $BM $BN $BK"
        done
      done
    done
  done
done

COMPILE_CMD="cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../ \
    -DCUDA_PATH=${CUDA_PATH} \
    -DCMAKE_CUDA_COMPILER=${CUDA_PATH}/bin/nvcc \
    ../../../"

RUN_CMD="build/release/bin/sgemm"

for p in "${PARAMS[@]}"; do
    read TM TN WM WN BM BN BK <<< "$p"
    echo "=== Kernel 06: BM=$BM BN=$BN TM=$TM TN=$TN BK=$BK WN=$WN WM=$WM ==="

    # Override values in cuh file
    sed -i "s/constexpr int BM_06 = .*/constexpr int BM_06 = $BM;/" "$FILE"
    sed -i "s/constexpr int BN_06 = .*/constexpr int BN_06 = $BN;/" "$FILE"
    sed -i "s/constexpr int TM_06 = .*/constexpr int TM_06 = $TM;/" "$FILE"
    sed -i "s/constexpr int TN_06 = .*/constexpr int TN_06 = $TN;/" "$FILE"
    sed -i "s/constexpr int BK_06 = .*/constexpr int BK_06 = $BK;/" "$FILE"
    sed -i "s/constexpr int WM_06 = .*/constexpr int WM_06 = $WM;/" "$FILE"
    sed -i "s/constexpr int WN_06 = .*/constexpr int WN_06 = $WN;/" "$FILE"

    # Compiler
    rm -rf build
    mkdir -p build/release/build && cd build/release/build
    $COMPILE_CMD > /dev/null 2>&1
    make > /dev/null 2>&1
    make install > /dev/null 2>&1
    cd ../../../

    # Execute
    $RUN_CMD | grep -E "Kernel 06: .* ms, .* GFLOPS"

done

# Restore
echo "$FILE_BACKUP" > "$FILE"
