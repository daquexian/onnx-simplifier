#!/usr/bin/env bash
set -uex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR
pushd third_party/onnxruntime/cmake/external/protobuf/cmake
mkdir -p build
cd build
cmake -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_WITH_ZLIB_DEFAULT=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -GNinja ..
cmake --build . --parallel `nproc`
PROTOC=`pwd`/protoc
popd

mkdir -p build-wasm
cd build-wasm
emcmake cmake -DONNX_CUSTOM_PROTOC_EXECUTABLE=$PROTOC -DONNXSIM_WASM_NODE=OFF -GNinja ..
ninja onnxsim_bin
