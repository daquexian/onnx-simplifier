#include <fstream>

#include "onnx/common/file_utils.h"
#include "onnxsim.h"
#include "onnxsim_option.h"

int main(int argc, char** argv) {
  // force env initialization to register opset
  InitEnv();
  OnnxsimOption option(argc, argv);
  bool is_no_opt = option.Get<bool>("no-opt");
  bool is_no_sim = option.Get<bool>("no-sim");
  auto input_model_filename = option.Get<std::string>("input-model");
  auto output_model_filename = option.Get<std::string>("output-model");

  onnx::ModelProto model;
  onnx::LoadProtoFromPath(input_model_filename, model);

  model = Simplify(model,
                   is_no_opt ? std::make_optional<std::vector<std::string>>({})
                             : std::nullopt,
                   is_no_sim, true, true);

  std::ofstream ofs(output_model_filename,
                    std::ios::out | std::ios::trunc | std::ios::binary);
  if (!model.SerializeToOstream(&ofs)) {
    throw std::invalid_argument("save model error");
  }
  return 0;
}
