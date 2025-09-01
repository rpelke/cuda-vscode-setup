#!/bin/bash
# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

rm -rf ${DIR}/build
mkdir -p ${DIR}/build
cd ${DIR}/build

if [ -z "${CUDA_PATH}" ]; then
  echo "Error: CUDA_PATH is not set" >&2
  exit 1
fi
cmake \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCUDA_PATH=${CUDA_PATH} \
    -DCMAKE_CUDA_COMPILER=${CUDA_PATH}/bin/nvcc \
    ..
make install
