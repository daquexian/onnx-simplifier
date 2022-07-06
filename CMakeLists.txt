cmake_minimum_required(VERSION 3.22)

project(onnxsim CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

option(ONNXSIM_PYTHON "" OFF)

if (ONNXSIM_PYTHON AND EMSCRIPTEN)
  message(STATUS "python and emscripten cannot be built at the same time")
endif()

# For MessageDifferencer::Equals
option(onnxruntime_USE_FULL_PROTOBUF "" ON)
if (EMSCRIPTEN)
  if (NOT DEFINED ONNX_CUSTOM_PROTOC_EXECUTABLE)
    message(FATAL_ERROR "ONNX_CUSTOM_PROTOC_EXECUTABLE must be set for emscripten")
  endif()

  option(onnxruntime_BUILD_WEBASSEMBLY "" ON)
  option(onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB "" ON)
  option(onnxruntime_ENABLE_WEBASSEMBLY_SIMD "" OFF)
  option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING "" ON)
  option(onnxruntime_ENABLE_WEBASSEMBLY_THREADS "" OFF)
  option(onnxruntime_BUILD_UNIT_TESTS "" OFF)
  set(onnxruntime_EMSCRIPTEN_SETTINGS "MALLOC=dlmalloc")

  option(ONNXSIM_WASM_NODE "For node (enable NODERAWFS etc.)" OFF)

  # For custom onnx target in onnx optimizer
  set(ONNX_TARGET_NAME onnxruntime_webassembly)
else()
  # For native build, only shared libs is ok. Otherwise libonnx.a will be linked twice (in onnxruntime and in onnxsim)
  # For emscripten build, since the libonnxruntime_webassembly.a is bundled by `bundle_static_library`, onnxsim can link
  # to the single libonnxruntime_webassembly.a
  set(BUILD_SHARED_LIBS ON)
  option(onnxruntime_BUILD_SHARED_LIB "" ON)
endif()
add_subdirectory(third_party/onnxruntime/cmake)

# configure onnx-optimizer after onnxruntime, because they both depend on onnx and onnxruntime has its own flags for onnx
add_subdirectory(third_party/onnx-optimizer)

add_library(onnxsim onnxsim/onnxsim.cpp)
target_include_directories(onnxsim PRIVATE third_party/onnxruntime/onnxruntime third_party/onnxruntime/include/onnxruntime)
target_include_directories(onnxsim PUBLIC onnxsim)
if (EMSCRIPTEN)
  target_link_libraries(onnxsim onnxruntime_webassembly onnx_optimizer)
else()
  target_link_libraries(onnxsim onnxruntime onnx_optimizer onnx)
endif()

add_executable(onnxsim_bin onnxsim/bin/onnxsim_bin.cpp)
target_link_libraries(onnxsim_bin onnxsim)
set_target_properties(onnxsim_bin PROPERTIES OUTPUT_NAME onnxsim)
if (EMSCRIPTEN)
  if (ONNXSIM_WASM_NODE)
    set_target_properties(onnxsim_bin PROPERTIES LINK_FLAGS "-s NODERAWFS=1 -s ALLOW_MEMORY_GROWTH=1 -s ASSERTIONS=2 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=1")
  else()
    set_target_properties(onnxsim_bin PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_onnxsim\"' -s 'EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
  endif()
endif()

if (ONNXSIM_PYTHON)
  add_subdirectory(third_party/pybind11)
  pybind11_add_module(onnxsim_cpp2py_export onnxsim/cpp2py_export.cc)
  target_link_libraries(onnxsim_cpp2py_export PRIVATE onnxsim)
endif()
