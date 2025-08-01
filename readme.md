## Build venv (if needed)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Compile cuda example
```bash
mkdir -p build/debug/build && cd build/debug/build
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=../ \
    -DCMAKE_CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4 \
    ../../../
make
make install

# Execute in main directory
./build/debug/bin/add_cuda
```
