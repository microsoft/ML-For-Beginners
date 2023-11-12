# SPDX-License-Identifier: Apache-2.0

"""
Optimization of :epkg:`ONNX` graphs.
Functions in *onnxconverter-common* do not support
opset < 9.
"""
from logging import getLogger
from onnx.helper import make_graph
from ._onnx_optimisation_common import (
    _rename_node_input,
    _rename_node_output,
    _apply_optimisation_on_graph,
    _apply_remove_node_fct_node,
)


logger = getLogger("skl2onnx")


def onnx_remove_node_identity(onnx_model, recursive=True, debug_info=None):
    """
    Removes as many *Identity* nodes as possible.
    The function looks into every node and subgraphs if
    *recursive* is True for identity node. Unless such a
    node directy connects one input to one output, it will
    be removed and every other node gets its inputs or
    outputs accordingly renamed.

    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :param debug_info: debug information (private)
    :return: new onnx model
    """
    if debug_info is None:
        debug_info = [str(type(onnx_model)).split(".")[-1].strip("'>")]
    else:
        debug_info = debug_info + [str(type(onnx_model)).split(".")[-1].strip("'>")]

    if hasattr(onnx_model, "graph"):
        return _apply_optimisation_on_graph(
            onnx_remove_node_identity,
            onnx_model,
            recursive=recursive,
            debug_info=debug_info,
        )

    graph = onnx_model

    inputs = set(i.name for i in graph.input)
    outputs = set(o.name for o in graph.output)

    def retrieve_idnodes(graph, existing_nodes):
        idnodes = []
        for i, exnode in enumerate(existing_nodes):
            if exnode is None:
                continue
            if exnode.op_type == "Identity":
                input = exnode.input[0]
                output = exnode.output[0]
                idnodes.append((i, exnode, input, output))
        return idnodes

    def retrieve_local_variables_subgraphs(graph):
        local = set()
        existing = set(i.name for i in graph.input)
        for node in graph.node:
            for i in node.input:
                if i not in existing:
                    local.add(i)
            for o in node.output:
                existing.add(o)
            res = retrieve_local_variables_nodes([node])
            for r in res:
                if r not in existing:
                    local.add(r)
        return local

    def retrieve_local_variables_nodes(nodes):
        names = set()
        for node in nodes:
            for att in node.attribute:
                if att.g:
                    names |= retrieve_local_variables_subgraphs(att.g)
        return names

    nodes = list(graph.node)
    local_variables = retrieve_local_variables_nodes(nodes)
    rem = 1
    while rem > 0:
        rem = 0
        idnodes = retrieve_idnodes(graph, nodes)
        restart = False
        for i, _, inp, out in idnodes:
            if restart:
                break
            if nodes[i] is None:
                # Already removed.
                continue
            if inp in inputs and out in outputs:
                # Cannot be removed.
                continue
            if out in local_variables:
                # out is used a local variable, this case is not implemented
                continue
            if not restart and out not in outputs:
                # We cannot change an output name.
                for j in range(len(nodes)):
                    if nodes[j] is None:
                        continue
                    if out in nodes[j].input:
                        nodes[j] = _rename_node_input(nodes[j], out, inp)
                        logger.debug(
                            "[VarId-a] rename node input %r into %r" % (out, inp)
                        )
                        rem += 1
                        if nodes[j].op_type == "Identity":
                            restart = True
                logger.debug("[NodeId-a] remove %r" % nodes[i])
                nodes[i] = None
                rem += 1
                continue
            if (
                not restart
                and inp not in inputs
                and inp not in outputs
                and out not in outputs
            ):
                # We cannot change an input name or an output name.
                for j in range(len(nodes)):
                    if nodes[j] is None:
                        continue
                    if inp in nodes[j].output:
                        nodes[j] = _rename_node_output(nodes[j], inp, out)
                        logger.debug("[Var] rename node output %r into %r" % (out, inp))
                        rem += 1
                        if nodes[j].op_type == "Identity":
                            restart = True
                    if inp in nodes[j].input:
                        nodes[j] = _rename_node_input(nodes[j], inp, out)
                        logger.debug(
                            "[VarId-b] rename node input %r into %r" % (out, inp)
                        )
                        rem += 1
                        if nodes[j].op_type == "Identity":
                            restart = True
                logger.debug("[NodeId-b] remove %r" % nodes[i])
                nodes[i] = None
                rem += 1

    if recursive:
        # Handles subgraphs.
        for i in range(len(nodes)):
            node = nodes[i]
            if node is None or not (node.attribute):
                continue
            nodes[i] = _apply_remove_node_fct_node(
                onnx_remove_node_identity,
                node,
                recursive=True,
                debug_info=debug_info + [node.name],
            )

    # Finally create the new graph.
    nodes = list(filter(lambda n: n is not None, nodes))
    graph = make_graph(
        nodes,
        onnx_model.name,
        onnx_model.input,
        onnx_model.output,
        onnx_model.initializer,
    )

    graph.value_info.extend(onnx_model.value_info)
    return graph
