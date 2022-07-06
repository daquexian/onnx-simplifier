#include "onnxsim.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <fstream>
#include <numeric>

#ifndef NO_BUILTIN_ORT
#include "../third_party/onnxruntime/include/onnxruntime/core/framework/endian.h"
#include "../third_party/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
#endif
#include "onnx/common/file_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnxoptimizer/optimize.h"

struct Config {
  std::vector<std::string> optimizer_passes;
  bool allow_large_tensor = true;
};

Config config;

std::shared_ptr<const ModelExecutor> ModelExecutor::instance_ = nullptr;

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

#ifndef NO_BUILTIN_ORT
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

struct CppModelExecutor : public ModelExecutor {
  std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const override {
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
    std::transform(
        model.graph().input().begin(), model.graph().input().end(),
        std::back_inserter(input_name_ptrs),
        [](const onnx::ValueInfoProto& x) { return x.name().c_str(); });
    std::transform(
        model.graph().output().begin(), model.graph().output().end(),
        std::back_inserter(output_name_ptrs),
        [](const onnx::ValueInfoProto& x) { return x.name().c_str(); });
    Ort::SessionOptions sess_opts;
    sess_opts.SetLogSeverityLevel(3);
    sess_opts.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    std::string model_str = model.SerializeAsString();
    Ort::Session session(*GetEnv(), model_str.data(), model_str.size(),
                         sess_opts);
    Ort::RunOptions run_opts;
    run_opts.SetRunLogSeverityLevel(3);
    std::vector<Ort::Value> input_tensors;
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(input_tensors), TensorProtoToTensor);
    auto output_tensors = session.Run(
        run_opts, input_name_ptrs.data(), input_tensors.data(),
        input_tensors.size(), output_name_ptrs.data(), output_name_ptrs.size());

    std::vector<onnx::TensorProto> output_tps;
    std::transform(output_tensors.begin(), output_tensors.end(),
                   std::back_inserter(output_tps), TensorToTensorProto);
    return output_tps;
  }
};

static int __register_cpp_model_executor __attribute__((unused)) = []() {
  ModelExecutor::set_instance(std::make_shared<CppModelExecutor>());
  return 0;
}();

void InitEnv() { GetEnv(); }
#else
void InitEnv() {
  // do nothing
}
#endif

std::vector<onnx::TensorProto> RunOp(onnx::ModelProto& model,
                                     const onnx::NodeProto& op) {
  std::vector<std::string> input_names;
  std::vector<onnx::TensorProto> input_tps;

  for (const auto& input : op.input()) {
    if (std::find(input_names.begin(), input_names.end(), input) !=
        input_names.end()) {
      continue;
    }
    input_names.push_back(input);
    auto in_tp = FindInitializerByName(model, input);
    input_tps.push_back(in_tp);
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
  for (const auto& x : op.output()) {
    onnx::ValueInfoProto vi;
    // In principle output ValueInfoProto must have type. But it is not checked.
    vi.set_name(x);
    *op_model.mutable_graph()->add_output() = vi;
  }

  auto output_tps = ModelExecutor::Run(op_model, input_tps);
  for (size_t i = 0; i < op.output_size(); i++) {
    output_tps[i].set_name(op.output(i));
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

bool HasSubgraph(const onnx::NodeProto& node) {
  for (const auto& attr : node.attribute()) {
    if (attr.type() == onnx::AttributeProto::GRAPH ||
        attr.type() == onnx::AttributeProto::GRAPHS) {
      return true;
    }
  }
  return false;
}

bool ProdeuceLargeTensor(const onnx::NodeProto& node) {
  return node.op_type() == "Tile" || node.op_type() == "ConstantOfShape";
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
    // clang-format off
    if (IsDeterministic(node.domain(), node.name()) &&
        !IsQDQ(node.domain(), node.name()) &&
        !HasSubgraph(node) &&
        (config.allow_large_tensor || !ProdeuceLargeTensor(node)) &&
        // clang-format on
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

onnx::ModelProto _InferShapes(const onnx::ModelProto& model) {
  onnx::ModelProto result;
  result.CopyFrom(model);
  onnx::shape_inference::InferShapes(result);
  return result;
}

onnx::ModelProto _FoldConstant(const onnx::ModelProto& model) {
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

onnx::ModelProto Optimize(const onnx::ModelProto& model) {
  return onnx::optimization::Optimize(model, config.optimizer_passes);
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

onnx::ModelProto Simplify(
    const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, bool allow_large_tensor) {
  config.allow_large_tensor = allow_large_tensor;
  config.optimizer_passes.clear();
  // skip_optimizers == nullopt means skiping all optimizers, so
  // config.optimizer_passes is empty
  if (skip_optimizers) {
    std::vector<std::string> passes;
    const auto all_passes = onnx::optimization::GetFuseAndEliminationPass();
    for (const auto& pass : all_passes) {
      if (std::find(skip_optimizers->begin(), skip_optimizers->end(), pass) ==
          skip_optimizers->end()) {
        passes.push_back(pass);
      }
    }
    config.optimizer_passes = passes;
  }

  auto FoldConstant = constant_folding ? _FoldConstant : Identity;
  auto InferShapes = shape_inference ? _InferShapes : Identity;

  Check(model);
  auto OptAndShape =
      FixedPointFn(std::function{InferShapes}, std::function{Optimize}, 15);
  auto OptAndShapeAndFold =
      FixedPointFn(std::function{OptAndShape}, std::function{FoldConstant}, 15);
  auto sim_model = OptAndShapeAndFold(model);
  Check(sim_model);
  return sim_model;
}
