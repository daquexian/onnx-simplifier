from onnxsim.onnx_simplifier import simplify, main

# register python executor
import onnxsim.onnx_simplifier
import onnxsim.onnxsim_cpp2py_export
x = onnxsim.onnx_simplifier.PyModelExecutor()
onnxsim.onnxsim_cpp2py_export._set_model_executor(x)

from .version import version as __version__  # noqa
