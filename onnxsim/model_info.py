from collections import defaultdict
from typing import Callable, Any, Optional

import onnx
from rich.table import Table
from rich.text import Text
from rich import print


__all__ = ['ModelInfo', 'print_simplifying_info']


def human_readable_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


class ModelInfo:
    """
    Model info contains:
    1. Num of every op
    2. Model size
    TODO: 
    Based on onnx runtime, get
    1、FLOPs
    2、forward memory footprint
    3、memory access
    4、compute density
    """

    def __init__(self, model: onnx.ModelProto):
        self.op_nums = defaultdict(int)
        # TODO: include ops in subgraph
        for node in model.graph.node:
            self.op_nums[node.op_type] += 1
        self.model_size = model.ByteSize()


def print_simplifying_info(model_ori: onnx.ModelProto, model_opt: onnx.ModelProto) -> None:
    """
    --------------------------------------------------------
    |             | original model | simplified model |
    --------------------------------------------------------
    | ****        | ****           | ****             |
    --------------------------------------------------------
    | Model Size  | ****           | ****             |
    --------------------------------------------------------
    """
    ori_info = ModelInfo(model_ori)
    opt_info = ModelInfo(model_opt)
    table = Table()
    table.add_column('')
    table.add_column('Original Model')
    table.add_column('Simplified Model')

    def add_row(table: Table, key, ori_data, opt_data, is_better: Callable[[Any, Any], Any], postprocess: Optional[Callable[[Any], Any]] = None) -> None:
        if postprocess is None:
            postprocess = str
        if is_better(opt_data, ori_data):
            table.add_row(key, postprocess(ori_data), Text(
                postprocess(opt_data), style='bold green1'))
        else:
            table.add_row(key, postprocess(ori_data), postprocess(opt_data))

    for key in sorted(list(set(ori_info.op_nums.keys()) | set(opt_info.op_nums.keys()))):
        add_row(table, key, ori_info.op_nums[key],
                opt_info.op_nums[key], lambda opt, ori: opt < ori)
    add_row(
        table, 'Model Size', ori_info.model_size, opt_info.model_size, lambda opt, ori: opt < ori, postprocess=human_readable_size)
    print(table)
