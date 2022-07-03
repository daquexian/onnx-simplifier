#pragma once

#include <onnx/onnx_pb.h>

#include <memory>

namespace Ort {
class Env;
}

std::shared_ptr<Ort::Env> GetEnv();
onnx::ModelProto Simplify(
    const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, bool allow_large_tensor);
