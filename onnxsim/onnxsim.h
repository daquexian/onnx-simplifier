#pragma once

#include <optional>
#include <memory>
#include <vector>

#include <onnx/onnx_pb.h>

struct ModelExecutor {
  virtual ~ModelExecutor() = default;
  static void set_instance(std::shared_ptr<const ModelExecutor> instance) {
    instance_ = std::move(instance);
  }
  static std::vector<onnx::TensorProto> Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) {
    if (instance_ == nullptr) {
      throw std::runtime_error("empty instance");
    }
    return instance_->_Run(model, inputs);
  }

  // public it for pybind11
  virtual std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const = 0;

 private:
  static std::shared_ptr<const ModelExecutor> instance_;
};

void InitEnv();

onnx::ModelProto Simplify(
    const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, bool allow_large_tensor);
