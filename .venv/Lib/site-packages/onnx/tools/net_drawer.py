# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# A library and utility for drawing ONNX nets. Most of this implementation has
# been borrowed from the caffe2 implementation
# https://github.com/pytorch/pytorch/blob/master/caffe2/python/net_drawer.py
#
# The script takes two required arguments:
#   -input: a path to a serialized ModelProto .pb file.
#   -output: a path to write a dot file representation of the graph
#
# Given this dot file representation, you can-for example-export this to svg
# with the graphviz `dot` utility, like so:
#
#   $ dot -Tsvg my_output.dot -o my_output.svg

import argparse
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import pydot

from onnx import GraphProto, ModelProto, NodeProto

OP_STYLE = {
    "shape": "box",
    "color": "#0F9D58",
    "style": "filled",
    "fontcolor": "#FFFFFF",
}

BLOB_STYLE = {"shape": "octagon"}

_NodeProducer = Callable[[NodeProto, int], pydot.Node]


def _escape_label(name: str) -> str:
    # json.dumps is poor man's escaping
    return json.dumps(name)


def _form_and_sanitize_docstring(s: str) -> str:
    url = "javascript:alert("
    url += _escape_label(s).replace('"', "'").replace("<", "").replace(">", "")
    url += ")"
    return url


def GetOpNodeProducer(  # noqa: N802
    embed_docstring: bool = False, **kwargs: Any
) -> _NodeProducer:
    def really_get_op_node(op: NodeProto, op_id: int) -> pydot.Node:
        if op.name:
            node_name = f"{op.name}/{op.op_type} (op#{op_id})"
        else:
            node_name = f"{op.op_type} (op#{op_id})"
        for i, input_ in enumerate(op.input):
            node_name += "\n input" + str(i) + " " + input_
        for i, output in enumerate(op.output):
            node_name += "\n output" + str(i) + " " + output
        node = pydot.Node(node_name, **kwargs)
        if embed_docstring:
            url = _form_and_sanitize_docstring(op.doc_string)
            node.set_URL(url)
        return node

    return really_get_op_node


def GetPydotGraph(  # noqa: N802
    graph: GraphProto,
    name: Optional[str] = None,
    rankdir: str = "LR",
    node_producer: Optional[_NodeProducer] = None,
    embed_docstring: bool = False,
) -> pydot.Dot:
    if node_producer is None:
        node_producer = GetOpNodeProducer(embed_docstring=embed_docstring, **OP_STYLE)
    pydot_graph = pydot.Dot(name, rankdir=rankdir)
    pydot_nodes: Dict[str, pydot.Node] = {}
    pydot_node_counts: Dict[str, int] = defaultdict(int)
    for op_id, op in enumerate(graph.node):
        op_node = node_producer(op, op_id)
        pydot_graph.add_node(op_node)
        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = pydot.Node(
                    _escape_label(input_name + str(pydot_node_counts[input_name])),
                    label=_escape_label(input_name),
                    **BLOB_STYLE,
                )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            pydot_graph.add_node(input_node)
            pydot_graph.add_edge(pydot.Edge(input_node, op_node))
        for output_name in op.output:
            if output_name in pydot_nodes:
                pydot_node_counts[output_name] += 1
            output_node = pydot.Node(
                _escape_label(output_name + str(pydot_node_counts[output_name])),
                label=_escape_label(output_name),
                **BLOB_STYLE,
            )
            pydot_nodes[output_name] = output_node
            pydot_graph.add_node(output_node)
            pydot_graph.add_edge(pydot.Edge(op_node, output_node))
    return pydot_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX net drawer")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The input protobuf file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output protobuf file.",
    )
    parser.add_argument(
        "--rankdir",
        type=str,
        default="LR",
        help="The rank direction of the pydot graph.",
    )
    parser.add_argument(
        "--embed_docstring",
        action="store_true",
        help="Embed docstring as javascript alert. Useful for SVG format.",
    )
    args = parser.parse_args()
    model = ModelProto()
    with open(args.input, "rb") as fid:
        content = fid.read()
        model.ParseFromString(content)
    pydot_graph = GetPydotGraph(
        model.graph,
        name=model.graph.name,
        rankdir=args.rankdir,
        node_producer=GetOpNodeProducer(
            embed_docstring=args.embed_docstring, **OP_STYLE
        ),
    )
    pydot_graph.write_dot(args.output)


if __name__ == "__main__":
    main()
