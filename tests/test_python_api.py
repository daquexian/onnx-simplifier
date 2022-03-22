import io
from typing import Any
import os

import torch
import onnx
import onnxsim
import torchvision as tv
import pytest


def export_simplify_and_check_by_python_api(m: torch.nn.Module, input: Any, *args, **kwargs):
    with io.BytesIO() as f:
        torch.onnx.export(m, input, f, *args, **kwargs)
        model = onnx.load_model_from_string(f.getvalue())
        sim_model, check_ok = onnxsim.simplify(model, check_n=3)
        assert check_ok
        return sim_model


def export_simplify_and_check_by_cli(m: torch.nn.Module, input: Any, *args, **kwargs):
    with io.BytesIO() as f:
        torch.onnx.export(m, input, f, *args, **kwargs)
        model = onnx.load_model_from_string(f.getvalue())
        sim_model, check_ok = onnxsim.simplify(model, check_n=3)
        assert check_ok
        return sim_model


class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()

    def forward(self, x):
        return x.view((x.shape[0], x.shape[1], x.shape[3] * x.shape[2]))


def test_just_reshape():
    net = JustReshape()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(net, dummy_input, do_constant_folding=False)
    assert len(sim_model.graph.node) == 1


@pytest.mark.skipif("ONNXSIM_CI" in os.environ, reason="memory limited")
def test_torchvision_fasterrcnn_fpn():
    model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(model, x, opset_version=11)


# maskrcnn is only supported in opset 11 and higher
def test_torchvision_maskrcnn_fpn_opset11():
    model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(model, x, opset_version=11)


# keypointrcnn is only supported in opset 11 and higher
@pytest.mark.skipif("ONNXSIM_CI" in os.environ, reason="memory limited")
def test_torchvision_keypointrcnn_fpn():
    model = tv.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(model, x, opset_version=11)


def test_torchvision_shufflenet_v2():
    model = tv.models.shufflenet_v2_x1_0(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


def test_torchvision_mnasnet():
    model = tv.models.mnasnet1_0(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


@pytest.mark.skipif("ONNXSIM_CI" in os.environ, reason="memory limited")
def test_torchvision_deeplabv3():
    model = tv.models.segmentation.deeplabv3_resnet50(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)
