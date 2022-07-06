#pragma once

#include <onnx/onnx_pb.h>

#include <memory>

void InitEnv();

onnx::ModelProto Simplify(
    const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, bool allow_large_tensor);
