import argparse

import onnx
import onnxsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output ONNX model')
    parser.add_argument('check_n', help='Check whether the output is correct with n random inputs',
                        nargs='?', type=int, default=0)
    parser.add_argument('--skip-optimization', help='Skip optimization of ONNX optimizers.',
                        action='store_true')
    args = parser.parse_args()
    print("Simplifying...")
    model_opt = onnxsim.simplify(args.input_model, check_n=args.check_n, perform_optimization=not args.skip_optimization)

    onnx.save(model_opt, args.output_model)
    print("Ok!")


if __name__ == '__main__':
    main()
