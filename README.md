# ONNX Simplifier

[![PyPI version](https://img.shields.io/pypi/v/onnx-simplifier.svg)](https://pypi.python.org/pypi/onnx-simplifier/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/onnx-simplifier.svg)](https://pypi.python.org/pypi/onnx-simplifier/)
[![PyPI license](https://img.shields.io/pypi/l/onnx-simplifier.svg)](https://pypi.python.org/pypi/onnx-simplifier/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

_ONNX is great, but sometimes too complicated._

## Background

One day I wanted to export the following simple reshape operation to ONNX:

```python
import torch


class OnlyReshape(torch.nn.Module):
    def __init__(self):
        super(OnlyReshape, self).__init__()

    def forward(self, x):
        return x.view((x.shape[0], x.shape[1], x.shape[3], x.shape[2]))


net = OnlyReshape()
model_name = 'only_reshape.onnx'
dummy_input = torch.randn(2, 3, 4, 5)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])
```

Input shape in ONNX is [static](https://github.com/onnx/onnx/issues/654), so what I expected is

![simple_reshape](imgs/simple_reshape.png)

However, I got the following complicated model even after
[polishing](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#polishing-the-model) it:

![complicated_reshape](imgs/complicated_reshape.png)

Moreover, there are also operations performed on weights in some ONNX models (e.g.,
[this](https://github.com/JDAI-CV/DNNLibrary/issues/17#issuecomment-455934190)). As pointed out in
https://github.com/onnx/onnx/issues/1758 and https://github.com/JDAI-CV/DNNLibrary/issues/26,
they can all be eliminated by offline computation.

## Our solution

ONNX Simplifier is presented to simplify the ONNX model. It infers the whole computation graph
and then replaces the redundant operators with their constant outputs.

Just install it via pip (Python >= 3.5)

```
pip3 install onnx-simplifier
```

Then

```
python3 -m onnxsim input_model output_model
```

## Results

An overall comparison between
[a complicated model](https://github.com/JDAI-CV/DNNLibrary/issues/17#issuecomment-455934190)
and its simplified version:

![Comparison between old model and new model](imgs/comparison.png)

