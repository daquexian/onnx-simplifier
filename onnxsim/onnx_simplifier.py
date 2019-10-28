from collections import OrderedDict

from typing import List, Dict, Union, Optional

import onnx                     # type: ignore
import onnx.helper              # type: ignore
import onnx.optimizer           # type: ignore
import onnx.shape_inference     # type: ignore
import onnxruntime as rt        # type: ignore

import numpy as np              # type: ignore

TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]


def add_features_to_output(m: onnx.ModelProto) -> None:
    """
    Add features to output in pb, so that ONNX Runtime will output them.
    :param m: the model that will be run in ONNX Runtime
    """
    for node in m.graph.node:
        for output in node.output:
            m.graph.output.extend([onnx.ValueInfoProto(name=output)])


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


def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
    """
    Note: This method relies on onnx shape inference, which is not reliable. So only use it on input or output tensors
    """
    v = get_value_info_all(m, name)
    if v is not None:
        return get_shape_from_value_info_proto(v)
    raise RuntimeError('Cannot get shape of "{}"'.format(name))


def get_elem_type(m: onnx.ModelProto, name: str) -> int:
    v = get_value_info_all(m, name)
    if v is not None:
        return v.type.tensor_type.elem_type
    raise RuntimeError('Cannot get type of "{}"'.format(name))


def get_np_type_from_elem_type(elem_type: int) -> int:
    sizes = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, None, None,
             np.float16, np.double, np.uint32, np.uint64, None, None, np.float16)
    assert len(sizes) == 17
    size = sizes[elem_type]
    assert size is not None
    return size


def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = list(set([ipt.name for ipt in model.graph.input]) -
                       set([x.name for x in model.graph.initializer]))
    return input_names


def generate_rand_input(model, input_shapes: TensorShapes = {}):
    input_names = get_input_names(model)
    full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
    assert None not in input_shapes
    full_input_shapes.update(input_shapes)  # type: ignore
    for key in full_input_shapes:
        if np.prod(full_input_shapes[key]) <= 0:
            raise RuntimeError(
                'The shape of input "{}" has dynamic size, please determine the input size manually by --input-shape xxx'.format(key))

    inputs = {ipt: np.random.rand(*full_input_shapes[ipt]).astype(np.float32) for ipt in
              input_names}
    return inputs


def get_constant_nodes(m: onnx.ModelProto) -> List[onnx.NodeProto]:
    const_nodes = []
    const_tensors = [x.name for x in m.graph.initializer]
    const_tensors.extend([node.output[0]
                          for node in m.graph.node if node.op_type == 'Constant'])

    for node in m.graph.node:
        if node.op_type == 'Shape':
            const_nodes.append(node)
            const_tensors.extend(node.output)
        elif all([x in const_tensors for x in node.input]):
            const_nodes.append(node)
            const_tensors.extend(node.output)
    return const_nodes


def forward(model, inputs=None, input_shapes: TensorShapes = {}) -> Dict[str, np.ndarray]:
    sess = rt.InferenceSession(model.SerializeToString())
    if inputs is None:
        inputs = generate_rand_input(model, input_shapes=input_shapes)
    outputs = [x.name for x in sess.get_outputs()]
    res = OrderedDict(zip(outputs, sess.run(outputs, inputs)))
    return res


def forward_all(model: onnx.ModelProto, input_shapes: TensorShapes = {}) -> Dict[str, np.ndarray]:
    import copy
    model = copy.deepcopy(model)
    add_features_to_output(model)
    res = forward(model, input_shapes=input_shapes)
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
                    vals=np.array(res[node.output[0]]).flatten().astype(
                        get_np_type_from_elem_type(elem_type))
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


def check(model_opt: onnx.ModelProto, model_ori: onnx.ModelProto, n_times: int = 5, input_shapes: TensorShapes = {}) -> None:
    """
    Warning: Some models (e.g., MobileNet) may fail this check by a small magnitude.
    Just ignore if it happens.
    :param model_opt: The simplified ONNX model
    :param model_ori: The original ONNX model
    :param n_times: Generate n random inputs
    """
    onnx.checker.check_model(model_opt)
    for i in range(n_times):
        print("Checking {}/{}...".format(i, n_times))
        rand_input = generate_rand_input(model_opt, input_shapes=input_shapes)
        res_opt = forward(model_opt, inputs=rand_input)
        res_ori = forward(model_ori, inputs=rand_input)

        for name in res_opt.keys():
            assert np.allclose(
                res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5)


def clean_constant_nodes(const_nodes: List[onnx.NodeProto], res: Dict[str, np.ndarray]):
    """
    It seems not needed since commit 6f2a72, but maybe it still prevents some unknown bug
    :param const_nodes: const nodes detected by `get_constant_nodes`
    :param res: The dict containing all tensors, got by `forward_all`
    :return: The constant nodes which have an output in res
    """
    return [node for node in const_nodes if node.output[0] in res]


def check_and_update_input_shapes(model: onnx.ModelProto, input_shapes: TensorShapes) -> TensorShapes:
    input_names = get_input_names(model)
    if None in input_shapes:
        if len(input_names) == 1:
            input_shapes[input_names[0]] = input_shapes[None]
            del input_shapes[None]
        else:
            raise RuntimeError(
                'The model has more than 1 inputs, please use the format "input_name:dim0,dim1,...,dimN" in --input-shape')
    for x in input_shapes:
        if x not in input_names:
            raise RuntimeError(
                'The model doesn\'t have input named "{}"'.format(x))
    return input_shapes


def simplify(model_ori: Union[str, onnx.ModelProto], check_n: int = 0, perform_optimization: bool = True, input_shapes: TensorShapes = {}) \
        -> onnx.ModelProto:
    if type(model_ori) == str:
        model_ori = onnx.load(model_ori)
    onnx.checker.check_model(model_ori)

    input_shapes = check_and_update_input_shapes(model_ori, input_shapes)

    model_opt = onnx.shape_inference.infer_shapes(model_ori)
    if perform_optimization:
        model_opt = optimize(model_opt)

    const_nodes = get_constant_nodes(model_opt)
    res = forward_all(model_opt, input_shapes=input_shapes)
    const_nodes = clean_constant_nodes(const_nodes, res)
    model_opt = eliminate_const_nodes(model_opt, const_nodes, res)

    if perform_optimization:
        model_opt = optimize(model_opt)

    check(model_opt, model_ori, check_n, input_shapes=input_shapes)

    return model_opt
