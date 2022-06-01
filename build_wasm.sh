#!/usr/bin/env bash
set -uex

pushd onnxsim/third_party/onnxruntime/cmake/external/protobuf/cmake
mkdir -p build
cd build
cmake -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_WITH_ZLIB_DEFAULT=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -GNinja ..
cmake --build . --parallel `nproc`
PROTOC=`pwd`/protoc
popd

pushd onnxsim
mkdir -p build
cd build
emcmake cmake -DONNX_CUSTOM_PROTOC_EXECUTABLE=$PROTOC -DONNXSIM_WASM_NODE=OFF -GNinja ..
ninja onnxsim
