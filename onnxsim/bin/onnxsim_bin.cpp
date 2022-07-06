#include <fstream>

#include "cxxopts.hpp"
#include "onnx/common/file_utils.h"
#include "onnxsim.h"

int main(int argc, char** argv) {
  // force env initialization to register opset
  InitEnv();
  cxxopts::Options options("onnxsim", "Simplify your ONNX model");
  options.add_options()("no-opt", "No optimization",
                        cxxopts::value<bool>()->default_value("false"))(
      "no-sim", "No simplification",
      cxxopts::value<bool>()->default_value("false"));
  auto result = options.parse(argc, argv);
  const bool opt = !result["no-opt"].as<bool>();
  const bool sim = !result["no-sim"].as<bool>();

  onnx::ModelProto model;
  onnx::LoadProtoFromPath(argv[1], model);

  model = Simplify(
      model, opt ? std::make_optional<std::vector<std::string>>({}) : std::nullopt,
      sim, true, true);

  std::ofstream ofs(argv[2],
                    std::ios::out | std::ios::trunc | std::ios::binary);
  if (!model.SerializeToOstream(&ofs)) {
    throw std::invalid_argument("save model error");
  }
  return 0;
}
