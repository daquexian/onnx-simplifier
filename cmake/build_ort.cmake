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

