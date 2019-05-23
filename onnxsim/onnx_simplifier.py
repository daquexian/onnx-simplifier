from collections import OrderedDict

from typing import List, Dict, Union

import onnx
import onnx.helper
import onnx.optimizer
import onnx.shape_inference
import onnxruntime as rt

import numpy as np


def add_features_to_output(m: onnx.ModelProto) -> None:
    """
    Add features to output in pb, so that ONNX Runtime will output them.
    :param m: the model that will be run in ONNX Runtime
    """
    m.graph.output.extend(m.graph.value_info)


def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]


def get_value_info(m: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    for v in m.graph.value_info:
        if v.name == name:
            return v


def get_value_info_all(m: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    for v in m.graph.value_info:
        if v.name == name:
            return v

    for v in m.graph.input:
        if v.name == name:
            return v

    for v in m.graph.output:
        if v.name == name:
            return v


def get_shape(m: onnx.ModelProto, name: str) -> List[int]:
    """
    Note: This method relies on onnx shape inference, which is not reliable. So only use it on input or output tensors
    """
    v = get_value_info_all(m, name)
    if v is not None:
        return get_shape_from_value_info_proto(v)


def get_elem_type(m: onnx.ModelProto, name: str) -> int:
    v = get_value_info_all(m, name)
    if v is not None:
        return v.type.tensor_type.elem_type


def get_np_type_from_elem_type(elem_type: int) -> int:
    sizes = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, None, None,
             np.float16, np.double, np.uint32, np.uint64, None, None, np.float16)
    assert len(sizes) == 17
    size = sizes[elem_type]
    assert size is not None
    return size


def generate_rand_input(model):
    input_names = list(set([ipt.name for ipt in model.graph.input]) - set([x.name for x in model.graph.initializer]))
    inputs = {ipt: np.random.rand(*get_shape(model, ipt)).astype(np.float32) for ipt in
              input_names}
    return inputs


def get_constant_nodes(m: onnx.ModelProto) -> List[onnx.NodeProto]:
    const_nodes = []
    const_tensors = [x.name for x in m.graph.initializer]
    const_tensors.extend([node.output[0] for node in m.graph.node if node.op_type == 'Constant'])

    for node in m.graph.node:
        if node.op_type == 'Shape':
            const_nodes.append(node)
            const_tensors.extend(node.output)
        elif all([x in const_tensors for x in node.input]):
            const_nodes.append(node)
            const_tensors.extend(node.output)
    return const_nodes


def forward(model, inputs=None):
    sess = rt.InferenceSession(model.SerializeToString())
    if inputs is None:
        inputs = generate_rand_input(model)
    outputs = [x.name for x in sess.get_outputs()]
    res = OrderedDict(zip(outputs, sess.run(outputs, inputs)))
    return res


def forward_all(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    import copy
    model = copy.deepcopy(model)
    add_features_to_output(model)
    res = forward(model)
    return res


def eliminate_const_nodes(model: onnx.ModelProto, const_nodes: List[onnx.NodeProto],
                          res: Dict[str, np.ndarray]) -> onnx.ModelProto:
    """
    :param model: the original onnx model
    :param const_nodes: const nodes detected by `get_constant_nodes`
    :param res: The dict containing all tensors, got by `forward_all`
    :return: the simplified onnx model. Redundant ops are all removed.
    """
    for node in model.graph.node[:]:
        if node in const_nodes:
            assert len(node.output) == 1
            node.op_type = 'Constant'
            elem_type = get_elem_type(model, node.output[0])
            shape = res[node.output[0]].shape
            new_attr = onnx.helper.make_attribute(
                'value',
                onnx.helper.make_tensor(
                    name=node.output[0],
                    data_type=elem_type,
                    dims=shape,
                    vals=np.array(res[node.output[0]]).flatten().astype(get_np_type_from_elem_type(elem_type))
                ))
            del node.input[:]
            del node.attribute[:]
            node.attribute.extend(
                [new_attr])
    return model


def optimize(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    :param model: The onnx model.
    :return: The optimized onnx model.
    Before simplifying, use this method to generate value_info, which is used in `forward_all`
    After simplifying, use this method to fold constants generated in previous step into initializer,
    and eliminate unused constants.
    """
    onnx.helper.strip_doc_string(model)
    model = onnx.optimizer.optimize(model, ['eliminate_deadend', 'eliminate_identity', 'eliminate_nop_dropout',
                                            'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
                                            'extract_constant_to_initializer', 'eliminate_unused_initializer',
                                            'eliminate_nop_transpose', 'fuse_add_bias_into_conv', 'fuse_bn_into_conv',
                                            'fuse_consecutive_concats', 'fuse_consecutive_log_softmax',
                                            'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes',
                                            'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm',
                                            'fuse_pad_into_conv', 'fuse_transpose_into_gemm'],
                                    fixed_point=True)
    return model


def check(model_opt: onnx.ModelProto, model_ori: onnx.ModelProto, n_times: int = 5) -> None:
    """
    Warning: Some models (e.g., MobileNet) may fail this check by a small magnitude.
    Just ignore if it happens.
    :param model_opt: The simplified ONNX model
    :param model_ori: The original ONNX model
    :param n_times: Generate n random inputs
    """
    onnx.checker.check_model(model_opt)
    for _ in range(n_times):
        rand_input = generate_rand_input(model_opt)
        res_opt = forward(model_opt, inputs=rand_input)
        res_ori = forward(model_ori, inputs=rand_input)

        for name in res_opt.keys():
            assert np.allclose(res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5)


def simplify(model_ori: Union[str, onnx.ModelProto], check_n: int = 0, perform_optimization: bool = True) \
        -> onnx.ModelProto:
    if type(model_ori) == str:
        model_ori = onnx.load(model_ori)
    onnx.checker.check_model(model_ori)

    model_opt = onnx.shape_inference.infer_shapes(model_ori)
    if perform_optimization:
        model_opt = optimize(model_opt)

    model_opt = eliminate_const_nodes(model_opt, get_constant_nodes(model_opt), forward_all(model_opt))

    if perform_optimization:
        model_opt = optimize(model_opt)

    check(model_opt, model_ori, check_n)

    return model_opt
