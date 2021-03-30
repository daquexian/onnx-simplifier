from collections import OrderedDict
from functools import reduce

from typing import Callable, List, Dict, Union, Optional, Tuple, Sequence, TypeVar, Any
import copy

import onnx  # type: ignore
import onnx.numpy_helper  # type: ignore


# get nodes that uses output of node
def out_usedby(model: onnx.ModelProto, node) -> List[Tuple[onnx.NodeProto, List[int]]]:
    nodes = []
    for i, nd in enumerate(model.graph.node):
        in_id = []
        for idx, inp in enumerate(nd.input):
            if node.output[0] == inp:
                in_id.append(idx)
        if len(in_id) > 0:
            nodes.append((nd, in_id))
    return nodes


# get node that creates inpt
def in_usedby(model: onnx.ModelProto, inpt: str) -> onnx.NodeProto:
    for nd in model.graph.node:
        if inpt in nd.output:
            return nd


def remove_add_node(model: onnx.ModelProto, node):
    b = in_usedby(model, node.input[0])
    a = out_usedby(model, node)
    for aa in a:
        for in_idx in aa[1]:
            aa[0].input[in_idx] = b.output[0]
    for i, nd in enumerate(model.graph.node):
        if nd == node:
            del model.graph.node[i]


def eliminate_zero_add(model: onnx.ModelProto):
    remove_node = []
    for node in model.graph.node:
        if node.op_type == 'Add':
            for x in node.input:
                b = next(
                    (True for xr in model.graph.initializer if (xr.name == x and not any([z for z in xr.raw_data]))),
                    False)
                if b:
                    remove_node.append(node)
                    break

    for node in remove_node:
        remove_add_node(model, node)
    return model
