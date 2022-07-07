cmake_minimum_required(VERSION 3.22)

# For std::filesystem in onnx optimizer
# Must be a cache variable and be set before project()
# Reference: https://cmake.org/cmake/help/latest/variable/CMAKE_OSX_DEPLOYMENT_TARGET.html
# It can be a normal variable if policy CMP0126 is set to NEW.
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15 CACHE STRING "Minimum OS X deployment version")

project(onnxsim CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

option(ONNXSIM_PYTHON "" OFF)
option(ONNXSIM_BUILTIN_ORT "" ON)
option(ONNXSIM_WASM_NODE "For node (enable NODERAWFS etc.)" OFF)

if (ONNXSIM_PYTHON AND EMSCRIPTEN)
  message(STATUS "python and emscripten cannot be built at the same time")
endif()

if (NOT ONNXSIM_BUILTIN_ORT AND EMSCRIPTEN)
  message(STATUS "emscripten needs builtin ort")
endif()

add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>>
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>>
  $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:Clang>:-fcolor-diagnostics>>)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (ONNXSIM_BUILTIN_ORT)
  include(cmake/build_ort.cmake)
  if (EMSCRIPTEN)
    set(ORT_NAME onnxruntime_webassembly)
  else()
    set(ORT_NAME onnxruntime)
  endif()
endif()

# configure onnx-optimizer after onnxruntime, because they both depend on onnx and onnxruntime has its own flags for onnx
add_subdirectory(third_party/onnx-optimizer)

add_library(onnxsim onnxsim/onnxsim.cpp)
if (ONNXSIM_BUILTIN_ORT)
  target_include_directories(onnxsim PRIVATE third_party/onnxruntime/onnxruntime third_party/onnxruntime/include/onnxruntime)
endif()
target_include_directories(onnxsim PUBLIC onnxsim)
if (NOT ONNXSIM_BUILTIN_ORT)
  target_compile_definitions(onnxsim PUBLIC NO_BUILTIN_ORT)
endif()
if (EMSCRIPTEN)
  target_link_libraries(onnxsim ${ORT_NAME} onnx_optimizer)
else()
  target_link_libraries(onnxsim ${ORT_NAME} onnx_optimizer onnx)
endif()

add_executable(onnxsim_bin onnxsim/bin/onnxsim_bin.cpp)
target_link_libraries(onnxsim_bin onnxsim)
set_target_properties(onnxsim_bin PROPERTIES OUTPUT_NAME onnxsim)
if (EMSCRIPTEN)
  if (ONNXSIM_WASM_NODE)
    set_target_properties(onnxsim_bin PROPERTIES LINK_FLAGS "-s NODERAWFS=1 -s ALLOW_MEMORY_GROWTH=1 -s ASSERTIONS=2 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=1 -sNO_DISABLE_EXCEPTION_CATCHING")
  else()
    set_target_properties(onnxsim_bin PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_onnxsim\"' -s 'EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
  endif()
endif()

if (ONNXSIM_PYTHON)
  add_subdirectory(third_party/pybind11)
  pybind11_add_module(onnxsim_cpp2py_export onnxsim/cpp2py_export.cc)
  target_link_libraries(onnxsim_cpp2py_export PRIVATE onnxsim)
  if(NOT "${PY_EXT_SUFFIX}" STREQUAL "")
    set_target_properties(onnxsim_cpp2py_export PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  endif()
endif()
