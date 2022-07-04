import argparse
from collections import OrderedDict
import copy
import os
import sys
from typing import Callable, List, Dict, Union, Optional, Tuple, Sequence, TypeVar
from rich.text import Text
from rich import print
from rich.prompt import Confirm

import onnx  # type: ignore
import onnx.helper  # type: ignore
import onnx.shape_inference  # type: ignore
import onnx.numpy_helper  # type: ignore
import onnxruntime as rt  # type: ignore
import onnxoptimizer  # type: ignore

import numpy as np  # type: ignore

import onnxsim.onnxsim_cpp2py_export as C
from . import model_info


def get_output_names(model: onnx.ModelProto) -> List[str]:
    output_names = [opt.name for opt in model.graph.output]
    return output_names


def remove_unused_output(
    model: onnx.ModelProto, unused_output: Sequence[str]
) -> onnx.ModelProto:
    unused_output_names = unused_output
    output_names = get_output_names(model)
    for unused_output_name in unused_output_names:
        if unused_output_name not in output_names:
            raise RuntimeError(
                f'The model doesn\'t have output named "{unused_output_name}"'
            )
    for graph_output in copy.deepcopy(model.graph.output):
        if graph_output.name in unused_output_names:
            model.graph.output.remove(graph_output)
    onnx.checker.check_model(model)
    return model


def simplify(
    model: Union[str, onnx.ModelProto],
    check_n: int = 0,
    perform_optimization: bool = True,
    skip_fuse_bn: bool = False,
    input_shapes=None,
    skipped_optimizers: Optional[List[str]] = None,
    skip_shape_inference=False,
    input_data=None,
    dynamic_input_shape: bool = False,
    custom_lib: Optional[str] = None,
    include_subgraph: bool = False,
    unused_output: Optional[Sequence[str]] = None,
    allow_large_tensor: bool = True,
) -> Tuple[onnx.ModelProto, bool]:
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
    if (
        input_shapes is not None
        or input_data is not None
        or dynamic_input_shape is not None
        or custom_lib is not None
        or include_subgraph is not None
    ):
        print(
            Text(
                "WARNING: The argument you used is deprecated, please refer to the latest documentation.",
                style="bold red",
            )
        )
    if check_n > 0:
        print(
            Text(
                "WARNING: checking correctness by random data is not implemented in onnxsim v0.4 for now. Checking will be skipped.",
                style="bold red",
            )
        )

    if not perform_optimization:
        # None means skip all optimizers
        skipped_optimizers = None
    elif skipped_optimizers is None:
        skipped_optimizers = []

    if skip_fuse_bn and skipped_optimizers is not None:
        skipped_optimizers.append("fuse_bn_into_conv")
    if isinstance(model, str):
        model = onnx.load(model)
    if unused_output is not None:
        model = remove_unused_output(model, unused_output)

    model_opt_bytes, check_ok = C.simplify(
        model.SerializeToString(),
        skipped_optimizers,
        True,
        not skip_shape_inference,
        allow_large_tensor,
    )
    model_opt = onnx.load_from_string(model_opt_bytes)
    return model_opt, check_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Input ONNX model")
    parser.add_argument("output_model", help="Output ONNX model")
    parser.add_argument(
        "check_n",
        help="Check whether the output is correct with n random inputs",
        nargs="?",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--enable-fuse-bn",
        help="This option is deprecated. Fusing bn into conv is enabled by default.",
        action="store_true",
    )
    parser.add_argument(
        "--skip-fuse-bn", help="Skip fusing batchnorm into conv.", action="store_true"
    )
    parser.add_argument(
        "--skip-optimization",
        help="Skip all ONNX optimizers or some of them. To skip all optimizers, use `onnxsim a.onnx b.onnx --skip-optimization`. To skip some of optimizers, use something like `onnxsim a.onnx b.onnx --skip-optimization fuse_bn_into_conv fuse_pad_into_pool`.",
        type=str,
        nargs="*",
    )
    parser.add_argument("--skip-constant-folding", action="store_true")
    parser.add_argument(
        "--input-shape",
        help="This argument has been renamed to --overwrite-input-shape, please refer to it",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--overwrite-input-shape",
        help='The manually-set static input shape, useful when the input shape is dynamic. The value should be "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.',
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--skip-optimizer",
        help="Deprecated. Refer to --skip-optimization",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--skip-shape-inference", help="Skip shape inference", action="store_true"
    )
    parser.add_argument(
        "--dynamic-input-shape",
        help="Deprecated. Not needed any more.",
        action="store_true",
    )
    parser.add_argument(
        "--input-data-path",
        help='input data, The value should be "input_name1:xxx1.bin"  "input_name2:xxx2.bin ...", input data should be a binary data file.',
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--custom-lib", help="Deprecated. Not needed any more.", type=str
    )
    parser.add_argument(
        "--include-subgraph",
        help='Experimental feature. Simplify subgraph (e.g. true graph and false graph of "If" operator) instead of only the main graph',
        action="store_true",
    )
    parser.add_argument(
        "--unused-output",
        help="Name of unused outputs that will be eliminated from the model",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--no-large-tensor",
        help="Some ops like Tile and ConstantOfShape can produce large tensor and make the model size much larger. Specifying this flag to skip folding these ops, with loss of some optimization chances.",
        action="store_true",
    )

    args = parser.parse_args()

    if args.check_n > 0:
        print(
            Text(
                "WARNING: checking correctness by random data is not implemented in onnxsim v0.4 for now. Checking will be skipped.",
                style="bold red",
            )
        )
    if args.enable_fuse_bn:
        print(
            Text(
                'WARNING: "--enable-fuse-bn" is not needed any more, because fuse bn is enabled by default. "--enable-fuse-bn" flag is ignored now and will raise an error in the future.',
                style="bold red",
            )
        )
    if args.dynamic_input_shape:
        print(
            Text(
                'WARNING: "--dynamic-input-shape" is not needed any more, onnxsim v0.4 now handles dynamic input shapes automatically. "--dynamic-input-shape" flag is ignored now and will raise an error in the future.',
                style="bold red",
            )
        )
    if args.custom_lib:
        print(
            Text(
                'WARNING: "--custom-lib" is not needed any more, onnxsim v0.4 now handles custom ops automatically. "--custom-lib" flag is ignored now and will raise an error in the future.',
                style="bold red",
            )
        )
    assert not (args.input_shape is not None and args.overwrite_input_shape is not None)
    if args.input_shape:
        print(
            Text(
                'WARNING: "--input-shape" is renamed to "--overwrite-input-shape". Please use it instead.',
                style="bold red",
            )
        )
        args.overwrite_input_shape = args.input_shape
    if args.include_subgraph:
        print(
            Text(
                "WARNING: subgraph optimization is not supported in v0.4 for now.",
                style="bold red",
            )
        )
    assert not (args.skip_optimizer is not None and args.skip_optimization is not None)
    if args.skip_optimizer:
        print(
            Text(
                'WARNING: "--skip-optimizer" is renamed to "--skip-optimization". Please use it instead.',
                style="bold red",
            )
        )
        args.skip_optimization = args.skip_optimizer
    if args.skip_optimization is None:
        # user doesn't specify --skip-optimization
        args.skip_optimization = []
    elif len(args.skip_optimization) == 0:
        # user specify --skip-optimization without any certain optimizer name
        # set it to None means skip all optimizations
        args.skip_optimization = None
    if args.skip_fuse_bn and args.skip_optimization is not None:
        args.skip_optimization.append("fuse_bn_into_conv")

    model = onnx.load(args.input_model)

    if not args.no_large_tensor:
        for node in model.graph.node:
            if node.op_type in ["Tile", "ConstantOfShape"]:
                print(
                    Text(
                        'Your model contains "Tile" ops or/and "ConstantOfShape" ops. Folding these ops can make the simplified model much larger. If it is not expected, please specify "--no-large-tensor" (which will lose some optimization chances)',
                        style="bold magenta",
                    )
                )
                break

    print("Simplifying...")

    if args.unused_output is not None:
        model = remove_unused_output(model, args.unused_output)

    model_opt_bytes, check_ok = C.simplify(
        model.SerializeToString(),
        args.skip_optimization,
        not args.skip_constant_folding,
        not args.skip_shape_inference,
        not args.no_large_tensor,
    )
    model_opt = onnx.load_from_string(model_opt_bytes)

    onnx.save(model_opt, args.output_model)

    if check_ok:
        print("Finish! Here is the difference:")
        model_info.print_simplifying_info(model, model_opt)
    else:
        print(
            'Check failed. Please be careful to use the simplified model, or try specifying "--skip-fuse-bn" or "--skip-optimization" (run "onnxsim -h" for details).'
        )
        print("Here is the difference after simplification:")
        model_info.print_simplifying_info(model, model_opt)
        sys.exit(1)
