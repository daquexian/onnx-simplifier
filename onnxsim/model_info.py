from collections import defaultdict
from typing import Callable, Any, Optional, Tuple, Dict

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

    def get_info(self, graph: onnx.GraphProto) -> Tuple[Dict[str, int], int]:
        op_nums = defaultdict(int)
        model_size = 0
        for node in graph.node:
            op_nums[node.op_type] += 1
            for attr in node.attribute:
                sub_graphs = []
                if attr.g is not None:
                    sub_graphs.append(attr.g)
                if attr.graphs is not None:
                    sub_graphs.extend(attr.graphs)
                for sub_graph in sub_graphs:
                    sub_op_nums, sub_model_size = self.get_info(sub_graph)
                    op_nums = defaultdict(int, {k: op_nums[k] + sub_op_nums[k] for k in set(op_nums) | set(sub_op_nums)})
                    model_size += sub_model_size
        op_nums["Constant"] += len(graph.initializer)
        model_size += graph.ByteSize()
        return op_nums, model_size

    def __init__(self, model: onnx.ModelProto):
        self.op_nums, self.model_size = self.get_info(model.graph)


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
