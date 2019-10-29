# ONNX Simplifier

[![PyPI version](https://img.shields.io/pypi/v/onnx-simplifier.svg)](https://pypi.python.org/pypi/onnx-simplifier/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/onnx-simplifier.svg)](https://pypi.python.org/pypi/onnx-simplifier/)
[![PyPI license](https://img.shields.io/pypi/l/onnx-simplifier.svg)](https://pypi.python.org/pypi/onnx-simplifier/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/daquexian/onnx-simplifier/pulls)

_ONNX is great, but sometimes too complicated._

## Background

One day I wanted to export the following simple reshape operation to ONNX:

```python
import torch


class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()

    def forward(self, x):
        return x.view((x.shape[0], x.shape[1], x.shape[3], x.shape[2]))


net = JustReshape()
model_name = 'just_reshape.onnx'
dummy_input = torch.randn(2, 3, 4, 5)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])
```

The input shape in this model is static, so what I expected is

![simple_reshape](imgs/simple_reshape.png)

However, I got the following complicated model even after
[polishing](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#polishing-the-model):

![complicated_reshape](imgs/complicated_reshape.png)

Moreover, there are also some operations performed on weights (like
[this](https://github.com/JDAI-CV/DNNLibrary/issues/17#issuecomment-455934190)), which
can all be eliminated by offline computation.

## Our solution

ONNX Simplifier is presented to simplify the ONNX model. It infers the whole computation graph
and then replaces the redundant operators with their constant outputs.

Install it via pip (Python >= 3.5)

```
pip3 install onnx-simplifier
```

Then

```
python3 -m onnxsim input_onnx_model output_onnx_model
```

For more functions like skipping optimization and setting input shape manually (when input shape is dynamic itself), try the following command for help message

```
python3 -m onnxsim -h
```

## Demonstration

An overall comparison between
[a complicated model](https://github.com/JDAI-CV/DNNLibrary/issues/17#issuecomment-455934190)
and its simplified version:

![Comparison between old model and new model](imgs/comparison.png)

