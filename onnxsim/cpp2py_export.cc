/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "onnx/py_utils.h"
#include "onnxsim.h"

namespace ONNX_NAMESPACE {
namespace py = pybind11;
using namespace pybind11::literals;
PYBIND11_MODULE(onnxsim_cpp2py_export, onnxsim_cpp2py_export) {
  onnxsim_cpp2py_export.doc() = "ONNX Simplifier";

  onnxsim_cpp2py_export.def(
      "simplify",
      [](const py::bytes& model_proto_bytes, bool opt, bool bin) {
        ModelProto model;
        ParseProtoFromPyBytes(&model, model_proto_bytes);
        auto const result = Simplify(model, opt, bin);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out);
      });
}
}  // namespace ONNX_NAMESPACE
