#include <fstream>

#include "onnx/common/file_utils.h"
#include "onnxsim.h"
#include "onnxsim_option.h"

int main(int argc, char** argv) {
  // force env initialization to register opset
  InitEnv();
  OnnxsimOption option(argc, argv);
  bool no_opt = option.Get<bool>("no-opt");
  bool no_sim = option.Get<bool>("no-sim");
  auto input_model_filename = option.Get<std::string>("input-model");
  auto output_model_filename = option.Get<std::string>("output-model");

  onnx::ModelProto model;
  onnx::LoadProtoFromPath(input_model_filename, model);

  model = Simplify(
      model,
      no_opt ? std::nullopt : std::make_optional<std::vector<std::string>>({}),
      !no_sim, true, SIZE_MAX);

  std::ofstream ofs(output_model_filename,
                    std::ios::out | std::ios::trunc | std::ios::binary);
  if (!model.SerializeToOstream(&ofs)) {
    throw std::invalid_argument("save model error");
  }
  return 0;
}
