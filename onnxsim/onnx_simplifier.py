import argparse
from collections import OrderedDict
import copy
import os
import sys
from typing import Callable, List, Dict, Union, Optional, Tuple, Sequence, TypeVar

import onnx  # type: ignore
import onnx.helper  # type: ignore
import onnx.shape_inference  # type: ignore
import onnx.numpy_helper  # type: ignore
import onnxruntime as rt  # type: ignore
import onnxoptimizer  # type: ignore

import numpy as np  # type: ignore

from . import model_info

Tensors = Dict[str, np.ndarray]
TensorShape = List[int]
TensorShapes = Dict[str, TensorShape]
TensorShapesWithOptionalKey = Dict[Optional[str], TensorShape]


class Config:
    dynamic_input_shape: bool
    include_subgraph: bool


config = Config()


def has_subgraph_in_node(node: onnx.NodeProto):
    for attr in node.attribute:
        if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
            return True
    return False


def get_all_subgraphs(model: onnx.ModelProto):
    graphs = [model.graph]
    for node in model.graph.node:
        if has_subgraph_in_node(node):
            for attr in node.attribute:
                graphs.append(attr.g)
    return graphs


def add_features_to_output(m: onnx.ModelProto, nodes: Sequence[onnx.NodeProto]) -> None:
    """
    Add features to output in pb, so that ONNX Runtime will output them.
    Note: the resulting model is not valid, because
    outputs of main graph should has other fields such as 'type'
    :param m: the model that will be run in ONNX Runtime
    :param nodes: nodes whose outputs will be added into the graph outputs
    """
    for node in nodes:
        for output in node.output:
            m.graph.output.extend([onnx.ValueInfoProto(name=output)])


def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]


def get_value_info_all(m: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
    for v in m.graph.value_info:
        if v.name == name:
            return v

    for v in m.graph.input:
        if v.name == name:
            return v

    for v in m.graph.output:
        if v.name == name:
            return v

    return None


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
    raise RuntimeError('Cannot get shape dtype "{}"'.format(name))


def get_np_type_from_elem_type(elem_type: int):
    sizes = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, str, bool,
             np.float16, np.double, np.uint32, np.uint64, np.complex64, np.complex128, np.float16)
    assert len(sizes) == 17
    size = sizes[elem_type]
    assert size is not None
    return size


def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]


def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = [ipt.name for ipt in get_inputs(model)]
    return input_names


def get_output_names(model: onnx.ModelProto) -> List[str]:
    output_names = [opt.name for opt in model.graph.output]
    return output_names


def remove_unused_output(model: onnx.ModelProto, unused_output: Sequence[str]) -> onnx.ModelProto:
    unused_output_names = unused_output
    output_names = get_output_names(model)
    for unused_output_name in unused_output_names:
        if unused_output_name not in output_names:
            raise RuntimeError(
                f'The model doesn\'t have output named "{unused_output_name}"')
    for graph_output in copy.deepcopy(model.graph.output):
        if graph_output.name in unused_output_names:
            model.graph.output.remove(graph_output)
    model = onnxoptimizer.optimize(model, ['eliminate_deadend'],
                                   fixed_point=True)
    onnx.checker.check_model(model)
    return model


def generate_specific_rand_input(model, input_shapes: TensorShapes):
    """
    Only generate rand inputs whose shape in `input_shapes`
    """

    for key, shape in input_shapes.items():
        shape_np = np.array(shape)
        if not np.all(shape_np > 0):
            # treat batch size as 1 automatically if dynamic_input_shape is True
            if config.dynamic_input_shape and len(shape_np) >= 3 and np.all(shape_np[1:] > 0):
                input_shapes[key] = [1] + shape[1:]
                continue

            raise RuntimeError(
                'The shape of input "{}" has dynamic size "{}", '
                'please try "--dynamic-input-shape" or determine '
                'the input size manually by "--input-shape xxx". '
                'Run "python3 -m onnxsim -h" for details'.format(key, shape))

    inputs = {ipt: np.array(np.random.rand(*input_shapes[ipt]),
                            dtype=get_np_type_from_elem_type(get_elem_type(model, ipt))) for ipt in
              input_shapes}
    return inputs


def generate_all_rand_input(model, input_shapes: Optional[TensorShapes] = None):
    """
    Generate random array for all inputs of a model
    """
    if input_shapes is None:
        input_shapes = {}
    input_names = get_input_names(model)
    full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
    assert None not in input_shapes
    full_input_shapes.update(input_shapes)  # type: ignore
    return generate_specific_rand_input(model, full_input_shapes)


def is_non_deterministic_node(node: onnx.NodeProto) -> bool:
    # TODO: handle node with subgraph
    return node.op_type in ['RandomNormal', 'RandomNormalLike', 'RandomUniform', 'RandomUniformLike']


def is_non_deterministic_model(model: onnx.ModelProto) -> bool:
    return any([is_non_deterministic_node(node) for node in model.graph.node])


def get_constant_nodes(m: onnx.ModelProto, dynamic_input_shape: bool = False) -> List[onnx.NodeProto]:
    const_nodes = []
    const_tensors = [x.name for x in m.graph.initializer]
    const_tensors.extend([node.output[0]
                          for node in m.graph.node if node.op_type == 'Constant'])

    # The output shape of some node types is determined by the input value
    # we consider the output of this node doesn't have constant shape,
    # so we do not simplify a such node even if the node is Shape op
    dynamic_tensors = []
    if dynamic_input_shape:
        dynamic_tensors.extend(get_input_names(m))

    def is_dynamic(node):
        if node.op_type in ['NonMaxSuppression', 'NonZero', 'Unique'] and node.input[0] not in const_tensors:
            return True
        if node.op_type in ['Reshape', 'Expand', 'Upsample', 'ConstantOfShape'] and len(node.input) > 1 and node.input[1] not in const_tensors:
            return True
        if node.op_type in ['Resize'] and ((len(node.input) > 2 and node.input[2] not in const_tensors) or (len(node.input) > 3 and node.input[3] not in const_tensors)):
            return True
        return False

    def check_node(graph):
        for node in graph.node:
            if has_subgraph_in_node(node):
                # Skip this node if this node has subgraph in it
                # "If" node with const cond will be eliminated by onnxoptimizer
                if any(x in dynamic_tensors for x in node.input):
                    dynamic_tensors.extend(node.output)
                if config.include_subgraph:
                    for attr in node.attribute:
                        if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                            check_node(attr.g)
            elif any(x in dynamic_tensors for x in node.input):
                dynamic_tensors.extend(node.output)
            # Note "elif" here, only Shape op with non-dynamic input will be seen as const node
            elif node.op_type == 'Shape':
                const_nodes.append(node)
                const_tensors.extend(node.output)
            elif is_dynamic(node):
                dynamic_tensors.extend(node.output)
            elif node.op_type in ['DequantizeLinear', 'QuantizeLinear']:
                # Skip QuantizeLinear and DequantizeLinear to preserve quantization info
                pass
            elif all([x in const_tensors for x in node.input]) and not is_non_deterministic_node(node):
                # Skip these nodes to avoid bloating the model size
                if node.op_type in ['ConstantOfShape', 'Tile']:
                    continue
                const_nodes.append(node)
                const_tensors.extend(node.output)

    check_node(m.graph)
    return copy.deepcopy(const_nodes)


def forward(model: onnx.ModelProto,
            input_data: Optional[Tensors] = None,
            input_shapes: Optional[TensorShapes] = None,
            outputs: Optional[Sequence[str]] = None,
            custom_lib: Optional[str] = None) -> Tensors:
    if outputs is not None and len(outputs) == 0:
        return {}
    if input_shapes is None:
        input_shapes = {}
    sess_options = rt.SessionOptions()
    if custom_lib is not None:
        if os.path.exists(custom_lib):
            sess_options.register_custom_ops_library(custom_lib)
        else:
            print("No such file '{}'".format(custom_lib), file=sys.stderr)
            exit(1)
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
    sess_options.log_severity_level = 3
    sess = rt.InferenceSession(model.SerializeToString(
    ), sess_options=sess_options, providers=['CPUExecutionProvider'])

    input_names = get_input_names(model)
    inputs = {}
    for name in input_names:
        if input_data is not None and input_data.get(name, None) is not None:
            inputs[name] = input_data[name]
        else:
            if input_shapes is not None and input_shapes.get(name, None) is not None:
                shape = input_shapes[name]
            else:
                shape = get_shape(model, name)
            inputs.update(generate_specific_rand_input(model, {name: shape}))

    if not outputs:
        outputs = [x.name for x in sess.get_outputs()]
    run_options = rt.RunOptions()
    run_options.log_severity_level = 3
    res = OrderedDict(zip(outputs, sess.run(
        outputs, inputs, run_options=run_options)))
    return res


def forward_for_node_outputs(model: onnx.ModelProto,
                             nodes: Sequence[onnx.NodeProto],
                             input_shapes: Optional[TensorShapes] = None,
                             input_data: Optional[Tensors] = None,
                             custom_lib: Optional[str] = None) -> Tensors:
    if input_shapes is None:
        input_shapes = {}
    model = copy.deepcopy(model)

    add_features_to_output(model, nodes)
    output_names = []
    for node in nodes:
        output_names.extend(node.output)

    if config.include_subgraph:
        subgraphs = get_all_subgraphs(model)
        for i in range(1, len(subgraphs)):
            subgraphs[0].node.extend(subgraphs[i].node)
        model = onnx.utils.Extractor(model).extract_model([], output_names)
        onnx.checker.check_model(model)
    res = forward(model,
                  input_data=input_data,
                  input_shapes=input_shapes,
                  outputs=output_names,
                  custom_lib=custom_lib)
    return res


def eliminate_const_nodes(model: onnx.ModelProto, const_nodes: Sequence[onnx.NodeProto],
                          res: Tensors) -> onnx.ModelProto:
    """
    :param model: the original onnx model
    :param const_nodes: const nodes detected by `get_constant_nodes`
    :param res: The dict containing all tensors, got by `forward_all`
    :return: the simplified onnx model. Redundant ops are all removed.
    """
    def recursive_eliminate_const_nodes_in_graph(graph, const_nodes, res):
        new_nodes = []
        for i, node in enumerate(graph.node):
            if node in const_nodes:
                for output in node.output:
                    new_node = copy.deepcopy(node)
                    new_node.name = "node_" + output
                    new_node.op_type = 'Constant'
                    new_attr = onnx.helper.make_attribute(
                        'value',
                        onnx.numpy_helper.from_array(res[output], name=output)
                    )
                    del new_node.input[:]
                    del new_node.attribute[:]
                    del new_node.output[:]
                    new_node.output.extend([output])
                    new_node.attribute.extend([new_attr])
                    new_nodes.append(new_node)
            else:
                new_nodes.append(node)
                if has_subgraph_in_node(node):
                    for attr in node.attribute:
                        if attr.g is None:
                            continue
                        recursive_eliminate_const_nodes_in_graph(
                            attr.g, const_nodes, res)
        del graph.node[:]
        graph.node.extend(new_nodes)
    recursive_eliminate_const_nodes_in_graph(model.graph, const_nodes, res)

    return model


def optimize(model: onnx.ModelProto, skip_fuse_bn: bool, skipped_optimizers: Optional[Sequence[str]]) -> onnx.ModelProto:
    """
    :param model: The onnx model.
    :return: The optimized onnx model.
    Before simplifying, use this method to generate value_info, which is used in `forward_all`
    After simplifying, use this method to fold constants generated in previous step into initializer,
    and eliminate unused constants.
    """

    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    optimizers_list = onnxoptimizer.get_fuse_and_elimination_passes()
    if skip_fuse_bn:
        optimizers_list.remove('fuse_bn_into_conv')
    if skipped_optimizers is not None:
        for opt in skipped_optimizers:
            try:
                optimizers_list.remove(opt)
            except ValueError:
                pass

    model = onnxoptimizer.optimize(model, optimizers_list,
                                   fixed_point=True)
    onnx.checker.check_model(model)
    return model


def check(model_ori: onnx.ModelProto, model_opt: onnx.ModelProto, n_times: int,
          input_shapes: Optional[TensorShapes] = None, custom_lib: Optional[str] = None) -> bool:
    """
    Warning: Some models (e.g., MobileNet) may fail this check by a small magnitude.
    Just ignore if it happens.
    :param input_shapes: Shapes of generated random inputs
    :param model_opt: The simplified ONNX model
    :param model_ori: The original ONNX model
    :param n_times: Generate n random inputs
    """
    if input_shapes is None:
        input_shapes = {}
    onnx.checker.check_model(model_opt)

    if is_non_deterministic_model(model_ori) and n_times > 0:
        print("The model has random ops like RandomNormal. Skip checking..")
        n_times = 0

    for i in range(n_times):
        print("Checking {}/{}...".format(i, n_times))
        rand_input = generate_all_rand_input(
            model_opt, input_shapes=input_shapes)
        res_opt = forward(model_opt, input_data=rand_input,
                          custom_lib=custom_lib)
        res_ori = forward(model_ori, input_data=rand_input,
                          custom_lib=custom_lib)

        for name in res_opt.keys():
            if not np.allclose(res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5):
                print("Tensor {} changes after simplifying. The max diff is {}.".format(
                    name, np.max(np.abs(res_opt[name] - res_ori[name]))))  # type: ignore
                print("Note that the checking is not always correct.")
                print("After simplifying:")
                print(res_opt[name])
                print("Before simplifying:")
                print(res_ori[name])
                print("----------------")
                return False
    return True


def clean_constant_nodes(const_nodes: Sequence[onnx.NodeProto], res: Tensors):
    """
    It seems not needed since commit 6f2a72, but maybe it still prevents some unknown bug
    :param const_nodes: const nodes detected by `get_constant_nodes`
    :param res: The dict containing all tensors, got by `forward_all`
    :return: The constant nodes which have an output in res
    """
    return [node for node in const_nodes if node.output[0] in res]


def check_and_update_input_shapes(model: onnx.ModelProto, input_shapes: TensorShapesWithOptionalKey, dynamic_input_shape: bool) -> TensorShapes:
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

    # Overwrite model input shape
    if not dynamic_input_shape:
        for name, input_shape in input_shapes.items():
            for ipt in model.graph.input:
                if ipt.name == name:
                    for i, dim in enumerate(ipt.type.tensor_type.shape.dim):
                        dim.dim_value = input_shape[i]

    return input_shapes  # type: ignore


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except:
        pass
    return model


T = TypeVar('T')


def fixed_point(x: T, func_a: Callable[[T], T], func_b: Callable[[T], T]) -> T:
    """
    Run `func_a` and `func_b` on `x` until func_b(func_a(x)) == x
    :param x:
    :param func_a: A function satisfying func_a(func_a(x)) == func_a(x)
    :param func_b: A function satisfying func_b(func_b(x)) == func_b(x)
    :return: the x that satisfies func_b(func_a(x)) == x
    """
    x = func_a(x)
    x = func_b(x)
    for _ in range(int(os.getenv('ONNXSIM_FIXED_POINT_MAX_ITER', '5'))):
        y = func_a(x)
        if y == x:
            # Since func_b(func_b(x)) == func_b(x),
            # we are already at the fixed point if
            # `y == x`
            return x
        x = y
        y = func_b(x)
        if y == x:
            return x
        x = y
    return x


def simplify(model: Union[str, onnx.ModelProto],
             check_n: int = 0,
             perform_optimization: bool = True,
             skip_fuse_bn: bool = False,
             input_shapes: Optional[TensorShapesWithOptionalKey] = None,
             skipped_optimizers: Optional[Sequence[str]] = None,
             skip_shape_inference=False,
             input_data: Optional[Tensors] = None,
             dynamic_input_shape: bool = False,
             custom_lib: Optional[str] = None,
             include_subgraph: bool = False,
             unused_output: Optional[Sequence[str]] = None) -> Tuple[onnx.ModelProto, bool]:
    """
    :param model: onnx ModelProto object or file path
    :param check_n: The simplified model will be checked for `check_n` times by random inputs
    :param perform_optimization: Whether to run onnx optimizer on the model
    :param skip_fuse_bn: Skip fuse_bn_into_conv onnx optimizer
    :param input_shapes: If the model has dynamic input shape, user must pass a fixed input shape
            for generating random inputs and checking equality. (Also see "dynamic_input_shape" param)
    :param skipped_optimizers: Skip some specific onnx optimizers
    :param skip_shape_inference: Skip shape inference (sometimes shape inference will crash)
    :param input_data: Feed custom input data for checking if needed
    :param dynamic_input_shape: Indicates whether the input shape should be dynamic. Note that
            input_shapes is also needed even if dynamic_input_shape is True,
            the value of input_shapes will be used when generating random inputs for checking equality.
            If 'dynamic_input_shape' is False, the input shape in simplified model will be overwritten
            by the value of 'input_shapes' param.
    :param custom_lib: onnxruntime custom ops's shared library
    :param include_subgraph: Simplify subgraph (e.g. true graph and false graph of "If" operator) instead of only the main graph
    :param unused_output: name of unused outputs that will be eliminated from the model
    :return: A tuple (simplified model, success(True) or failed(False))
    """
    config.dynamic_input_shape = dynamic_input_shape
    config.include_subgraph = include_subgraph

    if input_shapes is None:
        input_shapes = {}
    if input_data is None:
        input_data = {}

    if isinstance(model, str):
        model = onnx.load(model)
    assert(isinstance(model, onnx.ModelProto))
    onnx.checker.check_model(model)
    model_ori = model
    model = copy.deepcopy(model)

    input_names = get_input_names(model)
    for input_name, data in input_data.items():
        if input_name not in input_names:
            raise RuntimeError(
                'The model doesn\'t have input named "{}"'.format(input_name))

        shape = list(input_data[input_name].shape)

        # special case for single constant variables (with shape [])
        if len(shape) == 0:
            shape = [input_data[input_name].size]
        if input_name in input_shapes and shape != input_shapes[input_name]:
            raise RuntimeError('The shape of input_data[{}] is not the same with input_shape[{}]'.format(
                input_name, input_name))
        elif input_name not in input_shapes:
            input_shapes[input_name] = shape

    if unused_output is not None:
        model = remove_unused_output(model, unused_output)

    updated_input_shapes = check_and_update_input_shapes(
        model, input_shapes, dynamic_input_shape)

    def infer_shapes_and_optimize(model: onnx.ModelProto) -> onnx.ModelProto:
        def infer_shapes_if_applicable(model: onnx.ModelProto) -> onnx.ModelProto:
            if not skip_shape_inference:
                model = infer_shapes(model)
            return model

        def optimize_if_applicable(model: onnx.ModelProto) -> onnx.ModelProto:
            if perform_optimization:
                model = optimize(model, skip_fuse_bn, skipped_optimizers)
            return model

        return fixed_point(model, infer_shapes_if_applicable, optimize_if_applicable)

    def constant_folding(model: onnx.ModelProto) -> onnx.ModelProto:
        const_nodes = get_constant_nodes(
            model, dynamic_input_shape=dynamic_input_shape)
        res = forward_for_node_outputs(model,
                                       const_nodes,
                                       input_shapes=updated_input_shapes,
                                       input_data=input_data,
                                       custom_lib=custom_lib)
        const_nodes = clean_constant_nodes(const_nodes, res)
        model = eliminate_const_nodes(model, const_nodes, res)
        onnx.checker.check_model(model)
        return model

    model = fixed_point(model, constant_folding, infer_shapes_and_optimize)

    check_ok = check(model_ori, model, check_n,
                     input_shapes=updated_input_shapes, custom_lib=custom_lib)

    return model, check_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output ONNX model')
    parser.add_argument('check_n', help='Check whether the output is correct with n random inputs',
                        nargs='?', type=int, default=0)
    parser.add_argument('--enable-fuse-bn', help='This option is deprecated. Fusing bn into conv is enabled by default.',
                        action='store_true')
    parser.add_argument('--skip-fuse-bn', help='Skip fusing batchnorm into conv.',
                        action='store_true')
    parser.add_argument('--skip-optimization', help='Skip optimization of ONNX optimizers.',
                        action='store_true')
    parser.add_argument(
        '--input-shape', help='The manually-set static input shape, useful when the input shape is dynamic. The value should be "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.', type=str, nargs='+')
    parser.add_argument(
        '--skip-optimizer', help='Skip a certain ONNX optimizer', type=str, nargs='+')
    parser.add_argument('--skip-shape-inference',
                        help='Skip shape inference. Shape inference causes segfault on some large models', action='store_true')
    parser.add_argument('--dynamic-input-shape', help='This option enables dynamic input shape support. "Shape" ops will not be eliminated in this case. If `--dynamic_input_shape` is not specified and `--input-shape` is specified, the input shape in simplified model will be overwritten by the value of `--input_shape`. Note that if you want to check the simplication correctness (i.e. `check_n` > 0), "--input-shape" is also needed for generating random inputs and checking equality.', action='store_true')
    parser.add_argument(
        '--input-data-path', help='input data, The value should be "input_name1:xxx1.bin"  "input_name2:xxx2.bin ...", input data should be a binary data file.', type=str, nargs='+')
    parser.add_argument(
        '--custom-lib', help="custom lib path which should be absolute path, if you have custom onnxruntime backend you should use this to register you custom op", type=str)
    parser.add_argument(
        '--include-subgraph', help='Experimental feature. Simplify subgraph (e.g. true graph and false graph of "If" operator) instead of only the main graph', action='store_true')
    parser.add_argument(
        '--unused-output', help='Name of unused outputs that will be eliminated from the model', type=str, nargs='+')

    args = parser.parse_args()

    print("Simplifying...")

    if args.dynamic_input_shape and args.input_shape is None and args.check_n > 0:
        raise RuntimeError(
            'Please pass "--input-shape" argument for generating random input to check correctness. Run "python3 -m onnxsim -h" for details.')
    if args.input_shape is not None and not args.dynamic_input_shape:
        print("Note: The input shape of the simplified model will be overwritten by the value of '--input-shape' argument. Pass '--dynamic-input-shape' if it is not what you want. Run 'python3 -m onnxsim -h' for details.")
    input_shapes = dict()
    if args.input_shape is not None:
        for x in args.input_shape:
            if ':' not in x:
                input_shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                # for the input name like input:0
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                input_shapes.update({name: shape})

    input_data_paths = dict()
    if args.input_data_path is not None:
        for x in args.input_data_path:
            pieces = x.split(':')
            name, data = ':'.join(pieces[:-1]), pieces[-1]
            input_data_paths.update({name: data})

    input_tensors = dict()
    if len(input_data_paths) > 0 and args.input_shape is not None:
        for name in input_shapes.keys():
            input_data = np.fromfile(input_data_paths[name], dtype=np.float32)
            input_data = input_data.reshape(input_shapes[name])
            input_tensors.update({name: input_data})

    model = onnx.load(args.input_model)
    model_opt, check_ok = simplify(
        model,
        check_n=args.check_n,
        perform_optimization=not args.skip_optimization,
        skip_fuse_bn=args.skip_fuse_bn,
        input_shapes=input_shapes,
        skipped_optimizers=args.skip_optimizer,
        skip_shape_inference=args.skip_shape_inference,
        input_data=input_tensors,
        dynamic_input_shape=args.dynamic_input_shape,
        custom_lib=args.custom_lib,
        include_subgraph=args.include_subgraph,
        unused_output=args.unused_output)

    onnx.save(model_opt, args.output_model)

    if check_ok:
        print("Finish! Here is the difference:")
        model_info.print_simplifying_info(model, model_opt)
    else:
        print("Check failed. Please be careful to use the simplified model, or try specifying \"--skip-fuse-bn\" or \"--skip-optimization\" (run \"onnxsim -h\" for details).")
        print("Here is the difference after simplification:")
        model_info.print_simplifying_info(model, model_opt)
        sys.exit(1)
