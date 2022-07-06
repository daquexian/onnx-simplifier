import os
from typing import List, Dict, Optional
from collections import OrderedDict

import onnx
import onnx.checker
import numpy as np
import onnxruntime as rt

Tensors = Dict[str, np.ndarray]
TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]


def compare(
    model_opt: onnx.ModelProto,
    model_ori: onnx.ModelProto,
    n_times: int = 5,
    input_shapes: Optional[TensorShapes] = None,
    input_data: Optional[Tensors] = None,
    custom_lib: Optional[str] = None,
    verbose=True,
) -> bool:
    """
    :param model_opt: The simplified ONNX model
    :param model_ori: The original ONNX model
    :param n_times: Generate n random inputs
    :param input_shapes: Shapes of generated random inputs
    :param input_data: User-given data instead of random generated data
    :param custom_lib: ONNX Runtime custom lib for custom ops
    """

    def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
        return [dim.dim_value for dim in v.type.tensor_type.shape.dim]

    def get_value_info_all(
        m: onnx.ModelProto, name: str
    ) -> Optional[onnx.ValueInfoProto]:
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

    def get_elem_type(m: onnx.ModelProto, name: str) -> Optional[int]:
        v = get_value_info_all(m, name)
        if v is not None:
            return v.type.tensor_type.elem_type
        return None

    def get_np_type_from_elem_type(elem_type: int) -> int:
        sizes = (
            None,
            np.float32,
            np.uint8,
            np.int8,
            np.uint16,
            np.int16,
            np.int32,
            np.int64,
            str,
            np.bool,
            np.float16,
            np.double,
            np.uint32,
            np.uint64,
            np.complex64,
            np.complex128,
            np.float16,
        )
        assert len(sizes) == 17
        size = sizes[elem_type]
        assert size is not None
        return size

    def get_input_names(model: onnx.ModelProto) -> List[str]:
        input_names = list(
            set([ipt.name for ipt in model.graph.input])
            - set([x.name for x in model.graph.initializer])
        )
        return input_names

    def generate_rand_input(model, input_shapes: Optional[TensorShapes] = None):
        if input_shapes is None:
            input_shapes = {}
        input_names = get_input_names(model)
        full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
        assert None not in input_shapes
        full_input_shapes.update(input_shapes)  # type: ignore
        for key in full_input_shapes:
            if np.prod(full_input_shapes[key]) <= 0:
                raise RuntimeError(
                    'The shape of input "{}" has dynamic size, '
                    "please set an input shape manually with --test-input-shape".format(key)
                )

        inputs = {
            ipt: np.array(
                np.random.rand(*full_input_shapes[ipt]),
                dtype=get_np_type_from_elem_type(get_elem_type(model, ipt)),
            )
            for ipt in input_names
        }
        return inputs

    def forward(
            model: onnx.ModelProto, inputs: Tensors, custom_lib: Optional[str]=None
    ) -> Dict[str, np.ndarray]:
        sess_options = rt.SessionOptions()
        if custom_lib is not None:
            if os.path.exists(custom_lib):
                sess_options.register_custom_ops_library(custom_lib)
            else:
                raise ValueError("No such file '{}'".format(custom_lib))
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
        sess_options.log_severity_level = 3
        sess = rt.InferenceSession(
            model.SerializeToString(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        outputs = [x.name for x in sess.get_outputs()]
        run_options = rt.RunOptions()
        run_options.log_severity_level = 3
        res = OrderedDict(
            zip(outputs, sess.run(outputs, inputs, run_options=run_options))
        )
        return res

    if input_shapes is None:
        input_shapes = {}
    onnx.checker.check_model(model_opt)
    for i in range(n_times):
        print(f'Checking {i}/{n_times}...')
        if input_data is None:
            inputs = generate_rand_input(model_opt, input_shapes=input_shapes)
        else:
            inputs = input_data
        res_ori = forward(model_ori, inputs, custom_lib)
        res_opt = forward(model_opt, inputs, custom_lib)

        for name in res_opt.keys():
            if not np.allclose(res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5):
                if verbose:
                    print(
                        "Tensor {} changes after optimization. The max diff is {}.".format(
                            name, np.max(np.abs(res_opt[name] - res_ori[name]))
                        )
                    )
                    print("After optimization:")
                    print(res_opt[name])
                    print("Before optimization:")
                    print(res_ori[name])
                    print("----------------")
                return False
    return True
