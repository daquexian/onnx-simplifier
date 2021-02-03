import argparse
import sys

import onnx     # type: ignore
import onnxsim
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output ONNX model')
    parser.add_argument('check_n', help='Check whether the output is correct with n random inputs',
                        nargs='?', type=int, default=3)
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
    parser.add_argument('--dynamic-input-shape', help='This option enables dynamic input shape support. "Shape" ops will not be eliminated in this case. Note that "--input-shape" is also needed for generating random inputs and checking equality. If "dynamic_input_shape" is False, the input shape in simplified model will be overwritten by the value of "input_shapes" param.', action='store_true')
    parser.add_argument(
        '--input-data-path', help='input data, The value should be "input_name1:xxx1.bin"  "input_name2:xxx2.bin ...", input data should be a binary data file.', type=str, nargs='+')
    parser.add_argument(
        '--custom-lib', help="custom lib path which should be absolute path, if you have custom onnxruntime backend you should use this to register you custom op", type=str)

    args = parser.parse_args()

    print("Simplifying...")

    if args.dynamic_input_shape and args.input_shape is None:
        raise RuntimeError(
            'Please pass "--input-shape" argument for generating random input and checking equality. Run "python3 -m onnxsim -h" for details.')
    if args.input_shape is not None and not args.dynamic_input_shape:
        print("Note: The input shape of the simplified model will be overwritten by the value of '--input--shape' argument. Pass '--dynamic-input-shape' if it is not what you want. Run 'python3 -m onnxsim -h' for details.")
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

    model_opt, check_ok = onnxsim.simplify(
        args.input_model,
        check_n=args.check_n,
        perform_optimization=not args.skip_optimization,
        skip_fuse_bn=args.skip_fuse_bn,
        input_shapes=input_shapes,
        skipped_optimizers=args.skip_optimizer,
        skip_shape_inference=args.skip_shape_inference,
        input_data=input_tensors,
        dynamic_input_shape=args.dynamic_input_shape,
        custom_lib=args.custom_lib)

    onnx.save(model_opt, args.output_model)

    if check_ok:
        print("Ok!")
    else:
        print("Check failed, please be careful to use the simplified model, or try specifying \"--skip-fuse-bn\" or \"--skip-optimization\" (run \"python3 -m onnxsim -h\" for details)")
        sys.exit(1)


if __name__ == '__main__':
    main()
