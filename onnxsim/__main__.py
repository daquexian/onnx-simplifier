import argparse

import onnx
import onnxsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output ONNX model')
    args = parser.parse_args()
    model_opt = onnxsim.simplify(args.input_model)

    onnx.save(model_opt, args.output_model)


if __name__ == '__main__':
    main()
