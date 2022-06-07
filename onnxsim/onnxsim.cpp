#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <bit>
#include <fstream>
#include <numeric>

#include "cxxopts.hpp"
#include "onnx/common/file_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnxoptimizer/optimize.h"
#include "third_party/onnxruntime/include/onnxruntime/core/framework/endian.h"
#include "third_party/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h"

bool IsDeterministic(const std::string& domain, const std::string& op) {
  // Copy from onnxruntime/core/optimizer/utils.cc
  constexpr std::array kOnnxDomainNonDeterministicOps{
      "RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike",
      "Multinomial"};
  if (domain == "ai.onnx" || domain == "ai.onnx.ml" || domain.empty()) {
    auto iter = std::find(kOnnxDomainNonDeterministicOps.begin(),
                          kOnnxDomainNonDeterministicOps.end(), op);
    return iter == kOnnxDomainNonDeterministicOps.end();
  }
  // Unknown domain. Assume the op is not deterministic.
  return false;
}

bool IsQDQ(const std::string& domain, const std::string& op) {
  if (domain == "ai.onnx" || domain.empty()) {
    return op == "QuantizeLinear" || op == "DequantizeLinear";
  }
  return false;
}

auto FindInitializerByName(const onnx::ModelProto& model,
                           const std::string& name) {
  for (const auto& initializer : model.graph().initializer()) {
    if (initializer.name() == name) {
      return initializer;
    }
  }
  throw std::invalid_argument("no initializer " + name);
}

auto FindValueInfoProtoByName(const onnx::ModelProto& model,
                              const std::string& name) {
  for (const auto& vi : model.graph().value_info()) {
    if (vi.name() == name) {
      return vi;
    }
  }
  for (const auto& initializer : model.graph().initializer()) {
    if (initializer.name() == name) {
      onnx::ValueInfoProto vi;
      for (const auto& dim : initializer.dims()) {
        vi.mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
            ->add_dim()
            ->set_dim_value(dim);
      }
      vi.mutable_type()->mutable_tensor_type()->set_elem_type(
          initializer.data_type());
      vi.set_name(name);
      return vi;
    }
  }
  throw std::invalid_argument("no value info " + name);
}

onnx::TensorProto TensorToTensorProto(const Ort::Value& tensor) {
  onnx::TensorProto tensor_proto;
  for (const auto& dim : tensor.GetTensorTypeAndShapeInfo().GetShape()) {
    tensor_proto.add_dims(dim);
  }
  onnx::TensorProto::DataType onnx_dtype =
      (onnx::TensorProto::DataType)tensor.GetTensorTypeAndShapeInfo()
          .GetElementType();
  tensor_proto.set_data_type(onnx_dtype);

  switch (onnx_dtype) {
#define CASE_DTYPE(onnx_dtype, storage_dtype, cpp_type)                   \
  case onnx::TensorProto::onnx_dtype: {                                   \
    const auto* dptr = tensor.GetTensorData<cpp_type>();                  \
    for (size_t i = 0;                                                    \
         i < tensor.GetTensorTypeAndShapeInfo().GetElementCount(); i++) { \
      tensor_proto.add_##storage_dtype##_data(dptr[i]);                   \
    }                                                                     \
    break;                                                                \
  }

    CASE_DTYPE(FLOAT, float, float)
    CASE_DTYPE(DOUBLE, double, double)
    CASE_DTYPE(INT64, int64, int64_t)
    CASE_DTYPE(UINT64, uint64, uint64_t)
    CASE_DTYPE(INT32, int32, int32_t)
    CASE_DTYPE(UINT8, int32, uint8_t)
    CASE_DTYPE(INT8, int32, int8_t)
    CASE_DTYPE(UINT16, int32, uint16_t)
    CASE_DTYPE(INT16, int32, int16_t)
    CASE_DTYPE(BOOL, int32, int8_t)
#undef CASE_DTYPE
    default:
      throw std::invalid_argument("Unknown dtype " +
                                  std::to_string(tensor_proto.data_type()));
  }
  return tensor_proto;
}

Ort::Value TensorProtoToTensor(const onnx::TensorProto& tensor_proto) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto tensor = Ort::Value::CreateTensor(
      allocator, tensor_proto.dims().data(), tensor_proto.dims_size(),
      (ONNXTensorElementDataType)tensor_proto.data_type());
  if (tensor_proto.has_raw_data()) {
    if (onnxruntime::endian::native == onnxruntime::endian::big) {
      throw std::invalid_argument("only little endian is supported");
    }
    memcpy(tensor.GetTensorMutableData<void>(), tensor_proto.raw_data().data(),
           tensor_proto.raw_data().size());
  } else {
    switch (tensor_proto.data_type()) {
#define CASE_DTYPE(onnx_dtype, storage_dtype, cpp_type)         \
  case onnx::TensorProto::onnx_dtype: {                         \
    std::vector<cpp_type> vec;                                  \
    for (const auto& x : tensor_proto.storage_dtype##_data()) { \
      vec.push_back(x);                                         \
    }                                                           \
    memcpy(tensor.GetTensorMutableData<void>(), vec.data(),     \
           vec.size() * sizeof(cpp_type));                      \
    break;                                                      \
  }
      CASE_DTYPE(FLOAT, float, float)
      CASE_DTYPE(DOUBLE, double, double)
      CASE_DTYPE(INT64, int64, int64_t)
      CASE_DTYPE(UINT64, uint64, uint64_t)
      CASE_DTYPE(INT32, int32, int32_t)
      CASE_DTYPE(UINT8, int32, uint8_t)
      CASE_DTYPE(INT8, int32, int8_t)
      CASE_DTYPE(UINT16, int32, uint16_t)
      CASE_DTYPE(INT16, int32, int16_t)
      CASE_DTYPE(BOOL, int32, int8_t)
#undef CASE_DTYPE
      default:
        throw std::invalid_argument("Unknown dtype " +
                                    std::to_string(tensor_proto.data_type()));
    }
  }
  return tensor;
}

std::shared_ptr<Ort::Env> GetEnv() {
  static std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>();
  return env;
}

std::vector<onnx::TensorProto> RunOp(onnx::ModelProto& model,
                                     const onnx::NodeProto& op) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<Ort::Value> input_tensors;

  for (const auto& input : op.input()) {
    if (std::find(input_names.begin(), input_names.end(), input) != input_names.end()) {
      continue;
    }
    auto in_tp = FindInitializerByName(model, input);
    auto input_tensor = TensorProtoToTensor(in_tp);
    input_names.push_back(input);
    input_tensors.push_back(std::move(input_tensor));
  }
  onnx::ModelProto op_model;
  op_model.set_ir_version(model.ir_version());
  for (const auto& x : model.opset_import()) {
    *op_model.add_opset_import() = x;
  }
  *op_model.mutable_graph()->add_node() = op;
  for (const auto& x : input_names) {
    *op_model.mutable_graph()->add_input() = FindValueInfoProtoByName(model, x);
  }
  std::vector<std::string> output_names;
  for (const auto& x : op.output()) {
    onnx::ValueInfoProto vi;
    // In principle output ValueInfoProto must have type. But it is not checked.
    vi.set_name(x);
    *op_model.mutable_graph()->add_output() = vi;
    output_names.push_back(x);
  }

  std::vector<const char*> input_name_ptrs;
  std::vector<const char*> output_name_ptrs;
  std::transform(input_names.begin(), input_names.end(),
                 std::back_inserter(input_name_ptrs), [](const auto& x) {
                   return x.c_str();
                 });
  std::transform(output_names.begin(), output_names.end(),
                 std::back_inserter(output_name_ptrs), [](const auto& x) {
                   return x.c_str();
                 });
  auto op_model_str = op_model.SerializeAsString();
  Ort::SessionOptions sess_opts;
  sess_opts.SetLogSeverityLevel(3);
  sess_opts.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  Ort::Session session(*GetEnv(), op_model_str.data(), op_model_str.size(),
                       sess_opts);
  Ort::RunOptions run_opts;
  run_opts.SetRunLogSeverityLevel(3);
  auto output_tensors = session.Run(
      run_opts, input_name_ptrs.data(), input_tensors.data(),
      input_tensors.size(), output_name_ptrs.data(), output_name_ptrs.size());

  std::vector<onnx::TensorProto> output_tps;
  for (size_t i = 0; i < output_names.size(); i++) {
    onnx::TensorProto tp = TensorToTensorProto(output_tensors[i]);
    tp.set_name(output_names[i]);
    output_tps.push_back(tp);
  }
  return output_tps;
}

void RunOpAndAddInitializer(onnx::ModelProto& model,
                            const onnx::NodeProto& op) {
  const auto output_tps = RunOp(model, op);
  for (const auto& output_tp : output_tps) {
    *model.mutable_graph()->add_initializer() = output_tp;
  }
}

std::pair<std::vector<onnx::NodeProto>, std::vector<onnx::NodeProto>>
GetConstantNodes(const onnx::ModelProto& model) {
  std::vector<std::string> const_names;
  std::vector<onnx::NodeProto> const_nodes;
  std::vector<onnx::NodeProto> non_const_nodes;
  std::transform(
      model.graph().initializer().begin(), model.graph().initializer().end(),
      std::back_inserter(const_names), [](const auto& x) { return x.name(); });
  // node is already topo sorted
  for (const auto& node : model.graph().node()) {
    if (IsDeterministic(node.domain(), node.name()) &&
        !IsQDQ(node.domain(), node.name()) &&
        std::all_of(node.input().begin(), node.input().end(),
                    [&const_names](const auto& x) {
                      return std::find(const_names.begin(), const_names.end(),
                                       x) != const_names.end();
                    })) {
      const_names.insert(const_names.end(), node.output().begin(),
                         node.output().end());
      const_nodes.push_back(node);
    } else {
      non_const_nodes.push_back(node);
    }
  }
  return {const_nodes, non_const_nodes};
}

onnx::ModelProto InferShapes(const onnx::ModelProto& model) {
  onnx::ModelProto result;
  result.CopyFrom(model);
  onnx::shape_inference::InferShapes(result);
  return result;
}

onnx::ModelProto _FoldConstant(const onnx::ModelProto& model) {
  std::cout << "FoldConstant" << std::endl;
  const auto& tmp = model;
  {
    onnx::ModelProto model;
    model.CopyFrom(tmp);
    const auto [const_nodes, non_const_nodes] = GetConstantNodes(model);
    for (const auto& x : const_nodes) {
      RunOpAndAddInitializer(model, x);
    }
    model.mutable_graph()->clear_node();
    for (const auto& x : non_const_nodes) {
      *model.mutable_graph()->add_node() = x;
    }
    return model;
  }
}

onnx::ModelProto _Optimize(const onnx::ModelProto& model) {
  std::cout << "Optimize" << std::endl;
  return onnx::optimization::Optimize(
      model, onnx::optimization::GetFuseAndEliminationPass());
}

template <typename T>
std::function<T(const T&)> FixedPointFn(const std::function<T(const T&)>& f1,
                                        const std::function<T(const T&)>& f2,
                                        size_t max_iters) {
  return [f1, f2, max_iters](const T& x) {
    size_t _max_iters = max_iters;
    T tmp1 = f1(x);
    T tmp2 = f2(x);
    T& y1 = tmp1;
    T& y2 = tmp2;
    while (_max_iters-- > 0) {
      if (google::protobuf::util::MessageDifferencer::Equals(y1, y2)) {
        return y2;
      }
      y1 = f1(y2);
      if (google::protobuf::util::MessageDifferencer::Equals(y1, y2)) {
        return y1;
      }
      y2 = f2(y1);
    }
    return y2;
  };
}

onnx::ModelProto Identity(const onnx::ModelProto& model) { return model; }

void Check(const onnx::ModelProto& model) { onnx::checker::check_model(model); }

int main(int argc, char** argv) {
  // force env initialization to register opset
  GetEnv();
  cxxopts::Options options("onnxsim", "Simplify your ONNX model");
  options.add_options()("no-opt", "No optimization",
                        cxxopts::value<bool>()->default_value("false"))(
      "no-sim", "No simplification",
      cxxopts::value<bool>()->default_value("false"));
  auto result = options.parse(argc, argv);
  const bool opt = !result["no-opt"].as<bool>();
  const bool sim = !result["no-sim"].as<bool>();

  auto Optimize = opt ? _Optimize : Identity;
  auto FoldConstant = sim ? _FoldConstant : Identity;

  onnx::ModelProto model;
  onnx::LoadProtoFromPath(argv[1], model);
  Check(model);
  auto OptAndShape =
      FixedPointFn(std::function{InferShapes}, std::function{Optimize}, 5);
  auto OptAndShapeAndFold =
      FixedPointFn(std::function{OptAndShape}, std::function{FoldConstant}, 5);
  model = OptAndShapeAndFold(model);
  Check(model);
  std::ofstream ofs(argv[2],
                    std::ios::out | std::ios::trunc | std::ios::binary);
  if (!model.SerializeToOstream(&ofs)) {
    throw std::invalid_argument("save model error");
  }
  return 0;
}
