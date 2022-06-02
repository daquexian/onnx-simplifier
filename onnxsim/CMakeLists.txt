cmake_minimum_required(VERSION 3.22)

project(onnxsim CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

if (EMSCRIPTEN)
  if (NOT DEFINED ONNX_CUSTOM_PROTOC_EXECUTABLE)
    message(FATAL_ERROR "ONNX_CUSTOM_PROTOC_EXECUTABLE must be set for emscripten")
  endif()
  add_compile_options("-fexceptions")

  option(onnxruntime_BUILD_WEBASSEMBLY "" ON)
  option(onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB "" ON)
  option(onnxruntime_ENABLE_WEBASSEMBLY_SIMD "" OFF)
  option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING "" ON)
  option(onnxruntime_ENABLE_WEBASSEMBLY_THREADS "" OFF)
  option(onnxruntime_BUILD_UNIT_TESTS "" OFF)
  set(onnxruntime_EMSCRIPTEN_SETTINGS "MALLOC=dlmalloc")

  option(ONNXSIM_WASM_NODE "For node (enable NODERAWFS etc.)" OFF)
else()
  # For native build, only shared libs is ok. Otherwise libonnx.a will be linked twice (in onnxruntime and in onnxsim)
  # For emscripten build, since the libonnxruntime_webassembly.a is bundled by `bundle_static_library`, onnxsim can link
  # to the single libonnxruntime_webassembly.a
  set(BUILD_SHARED_LIBS ON)
  option(onnxruntime_BUILD_SHARED_LIB "" ON)
  option(onnxruntime_USE_FULL_PROTOBUF "" ON)
endif()
add_subdirectory(third_party/onnxruntime/cmake)

# configure onnx-optimizer after onnxruntime, because they both depend on onnx and onnxruntime has its own flags for onnx
add_subdirectory(third_party/onnx-optimizer)

add_executable(onnxsim onnxsim.cpp)
target_include_directories(onnxsim PRIVATE third_party/onnxruntime/onnxruntime third_party/onnxruntime/include/onnxruntime)
if (EMSCRIPTEN)
  set(ONNX_TARGET_NAME onnxruntime_webassembly)

  target_link_libraries(onnxsim onnxruntime_webassembly onnx_optimizer)
  if (ONNXSIM_WASM_NODE)
    set_target_properties(onnxsim PROPERTIES LINK_FLAGS "-s NODERAWFS=1 -s ALLOW_MEMORY_GROWTH=1 -s SAFE_HEAP=1 -fexceptions")
  else()
    set_target_properties(onnxsim PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_onnxsim\"' -s 'EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
  endif()
else()
  target_link_libraries(onnxsim onnxruntime onnx_optimizer onnx)
endif()
