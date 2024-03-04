import io
from typing import Any, Callable, Dict, Optional
import os
import tempfile

import numpy as np
import torch
import onnx
import onnxsim
import torchvision as tv
import pytest


def export_simplify_and_check_by_python_api(
    m: torch.nn.Module,
    input: Any,
    *,
    is_model_valid: Optional[Callable[[Any], bool]] = None,
    export_kwargs: Optional[Dict[str, Any]] = None,
    simplify_kwargs: Optional[Dict[str, Any]] = None,
) -> onnx.ModelProto:
    if is_model_valid is None:
        is_model_valid = lambda _: True
    if export_kwargs is None:
        export_kwargs = {}
    if simplify_kwargs is None:
        simplify_kwargs = {}
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_fn = os.path.join(tmpdirname, "tmp.onnx")
        torch.onnx.export(m, input, model_fn, **export_kwargs)
        model = onnx.load(model_fn)
        if not is_model_valid(model):
            raise AssertionError(f"model is invalid:\n{model}")
        # read the model from filesystem to support >2GB large model
        sim_model, check_ok = onnxsim.simplify(model_fn, check_n=3, **simplify_kwargs)
        assert check_ok
        return sim_model


def str_is_logical_positive(x: str) -> bool:
    return x.lower() in ["1", "on", "true"]


def skip_in_ci():
    return pytest.mark.skipif(
        str_is_logical_positive(os.getenv("ONNXSIM_CI", "")), reason="memory limited"
    )


def test_just_reshape():
    class JustReshape(torch.nn.Module):
        def __init__(self):
            super(JustReshape, self).__init__()

        def forward(self, x):
            return x.view((x.shape[0], x.shape[1], x.shape[3] * x.shape[2]))

    net = JustReshape()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net, dummy_input, export_kwargs={"do_constant_folding": False}
    )
    assert len(sim_model.graph.node) == 1


def test_a_model_not_need_simplification():
    class ModelNotNeedSimplification(torch.nn.Module):
        def __init__(self):
            super(ModelNotNeedSimplification, self).__init__()

        def forward(self, x):
            return x + 1

    net = ModelNotNeedSimplification()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(net, dummy_input)
    assert len(sim_model.graph.node) == 1


def test_exprimental_simplify_subgraph():
    class WithSubGraph(torch.nn.Module):
        def __init__(self):
            super(WithSubGraph, self).__init__()

        def forward(self, x):
            if x.sum() > 1.0:
                # NOTE: even onnxsim cannot simplify it,
                # a canonical pass in onnx-optimizer is needed for it.
                # so this test only tests that include_subgraph doesn't
                # result in invalid model in this case
                return 3 + x + 3
            else:
                return x + 4

    net = torch.jit.script(WithSubGraph())
    dummy_input = torch.randn(2)
    sim_model = export_simplify_and_check_by_python_api(
        net, dummy_input, simplify_kwargs={"include_subgraph": True}
    )
    assert len(sim_model.graph.node) == 3
    assert len(sim_model.graph.node[2].attribute[0].g.node) == 2
    assert len(sim_model.graph.node[2].attribute[1].g.node) == 1


def test_dynamic_batch_size():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()

        def forward(self, x):
            return x + 2

    net = SimpleModel()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        export_kwargs={
            "input_names": ["input"],
            "dynamic_axes": {"input": {0: "batch_size"}},
        },
        simplify_kwargs={"test_input_shapes": {"input": [2, 3, 4, 5]}},
    )
    assert len(sim_model.graph.node) == 1


# NOTE: `include_subgraph` makes this test fail
@skip_in_ci()
def test_torchvision_fasterrcnn_fpn():
    model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(
        model, x, export_kwargs={"opset_version": 11}
    )


# maskrcnn is only supported in opset 11 and higher
@skip_in_ci()
def test_torchvision_maskrcnn_fpn_opset11():
    model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(
        model, x, export_kwargs={"opset_version": 11}
    )


# keypointrcnn is only supported in opset 11 and higher
@skip_in_ci()
def test_torchvision_keypointrcnn_fpn():
    model = tv.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    export_simplify_and_check_by_python_api(
        model, x, export_kwargs={"opset_version": 11}
    )


# shufflenet and mnasnet causes segfault in CI (perhaps because of memory limit)
# but works locally
@skip_in_ci()
def test_torchvision_shufflenet_v2():
    model = tv.models.shufflenet_v2_x1_0(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


@skip_in_ci()
def test_torchvision_mnasnet():
    model = tv.models.mnasnet1_0(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


@skip_in_ci()
def test_torchvision_deeplabv3():
    model = tv.models.segmentation.deeplabv3_resnet50(pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    export_simplify_and_check_by_python_api(model, x)


def test_unused_output():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()

        def forward(self, x):
            x1 = x + 2
            x1 = x1 - 2
            x1 = x1 * 2
            x1 = x1 / 2
            y1 = x1
            x2 = x + 2
            x2 = x2 - 2
            x2 = x2 * 2
            x2 = x2 / 2
            y2 = x2
            x3 = x + 2
            x3 = x3 - 2
            x3 = x3 * 2
            x3 = x3 / 2
            y3 = x3
            return y1, y2, y3

    net = SimpleModel()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        export_kwargs={
            "input_names": ["input"],
            "output_names": ["output0", "output1", "output2"],
        },
        simplify_kwargs={"unused_output": ["output1", "output2"]},
    )
    assert len(sim_model.graph.node) == 4


def test_remove_unused_initializer():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.w = torch.nn.Parameter(torch.ones(5, 4))

        def forward(self, x):
            return x + torch.transpose(self.w, 0, 1)

    net = SimpleModel()
    dummy_input = torch.randn(2, 3, 4, 5)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        is_model_valid=lambda model: any(
            node.op_type == "Transpose" for node in model.graph.node
        ),
        export_kwargs={"do_constant_folding": False},
    )
    assert len(sim_model.graph.node) == 1
    assert len(sim_model.graph.initializer) == 1


@skip_in_ci()
def test_model_larger_than_2gb():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            # a parameter is 500MB
            self.w1 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w2 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w3 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w4 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))
            self.w5 = torch.nn.Parameter(torch.ones(125 * 1024 * 1024))

        def forward(self, x):
            return x + (self.w1 + self.w2 + self.w3 + self.w4 + self.w5)

    net = SimpleModel()
    dummy_input = torch.randn(125 * 1024 * 1024)
    sim_model = export_simplify_and_check_by_python_api(
        net,
        dummy_input,
        is_model_valid=lambda model: sum(
            node.op_type == "Add" for node in model.graph.node
        )
        == 5,
        export_kwargs={"do_constant_folding": False},
    )
    assert len(sim_model.graph.node) == 1
    assert sim_model.graph.node[0].op_type == "Add"


def test_unset_optional_input():
    fmap = []
    nodes = [] 
    initializers = []

    fmap.append(onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, shape=(1,3,4,4)))

    X = np.random.rand(1,3,2,2).astype(np.float32)
    initializers.append(onnx.helper.make_tensor('X', onnx.TensorProto.FLOAT, X.shape, X.copy().tobytes(), raw=True))
    sizes = np.asarray([1,3,4,4]).astype(np.int64)
    initializers.append(onnx.helper.make_tensor('sizes', onnx.TensorProto.INT64, sizes.shape, sizes.copy().tobytes(), raw=True))

    nodes.append(onnx.helper.make_node(
      'Resize',
      inputs=['X', '', '', 'sizes'],
      outputs=['y'],
      mode='linear'))

    graph_def = onnx.helper.make_graph(
      nodes,
      'test_unset_optional_input',
      [],
      [fmap[-1]],
      value_info=fmap,
      initializer=initializers
      )

    opset_imports = [onnx.helper.make_opsetid("", 14)]
    
    model = onnx.helper.make_model(graph_def, opset_imports=opset_imports)
    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok
    assert len(model.graph.node) == 1
    assert len(model.graph.initializer) == 2
    assert len(sim_model.graph.node) == 0
    assert len(sim_model.graph.initializer) == 1
