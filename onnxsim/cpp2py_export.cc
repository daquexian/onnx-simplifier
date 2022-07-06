/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "onnx/py_utils.h"
#include "onnxsim.h"

namespace py = pybind11;
using namespace pybind11::literals;

struct PyModelExecutor : public ModelExecutor {
  using ModelExecutor::ModelExecutor;

  std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const override {
    std::vector<py::bytes> inputs_bytes;
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(inputs_bytes),
                   [](const onnx::TensorProto& x) {
                     return py::bytes(x.SerializeAsString());
                   });
    std::string model_str = model.SerializeAsString();
    auto output_bytes = _PyRun(py::bytes(model_str), inputs_bytes);
    std::vector<onnx::TensorProto> output_tps;
    std::transform(output_bytes.begin(), output_bytes.end(),
                   std::back_inserter(output_tps), [](const py::bytes& x) {
                     onnx::TensorProto tp;
                     tp.ParseFromString(std::string(x));
                     return tp;
                   });
    return output_tps;
  }

  virtual std::vector<py::bytes> _PyRun(
      const py::bytes& model_bytes,
      const std::vector<py::bytes>& inputs_bytes) const = 0;
};

struct PyModelExecutorTrampoline : public PyModelExecutor {
  /* Inherit the constructors */
  using PyModelExecutor::PyModelExecutor;

  /* Trampoline (need one for each virtual function) */
  std::vector<py::bytes> _PyRun(
      const py::bytes& model_bytes,
      const std::vector<py::bytes>& inputs_bytes) const override {
    PYBIND11_OVERRIDE_PURE_NAME(
        std::vector<py::bytes>, /* Return type */
        PyModelExecutor,        /* Parent class */
        "Run",
        _PyRun, /* Name of function in C++ (must match Python name) */
        model_bytes, inputs_bytes /* Argument(s) */
    );
  }
};

PYBIND11_MODULE(onnxsim_cpp2py_export, m) {
  m.doc() = "ONNX Simplifier";

  m.def("simplify",
        [](const py::bytes& model_proto_bytes,
           std::optional<std::vector<std::string>> skip_optimizers,
           bool constant_folding, bool shape_inference,
           bool allow_large_tensor) -> std::pair<py::bytes, bool> {
          // force env initialization to register opset
          InitEnv();
          ONNX_NAMESPACE::ModelProto model;
          ParseProtoFromPyBytes(&model, model_proto_bytes);
          auto const result = Simplify(model, skip_optimizers, constant_folding,
                                       shape_inference, allow_large_tensor);
          std::string out;
          result.SerializeToString(&out);
          return {py::bytes(out), true};
        })
      .def("_set_model_executor", [](std::shared_ptr<PyModelExecutor> executor) {
        ModelExecutor::set_instance(std::move(executor));
      });

  py::class_<PyModelExecutor, PyModelExecutorTrampoline, std::shared_ptr<PyModelExecutor>>(
      m, "ModelExecutor")
      .def(py::init<>())
      .def("Run", &PyModelExecutor::_PyRun);
}
