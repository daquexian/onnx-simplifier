#pragma once

#include <onnx/onnx_pb.h>
#include <memory>

namespace Ort {
  class Env;
}

std::shared_ptr<Ort::Env> GetEnv();
onnx::ModelProto Simplify(const onnx::ModelProto& model, bool opt, bool sim);
