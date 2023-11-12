# SPDX-License-Identifier: Apache-2.0

"""
Common functions to reduce the number of
nodes of an :epkg:`ONNX` graphs.
"""
from onnx.helper import make_graph, ValueInfoProto, make_model
from onnx import AttributeProto, NodeProto
from onnx.helper import make_attribute


def _apply_optimisation_on_graph(
    fct, onnx_model, recursive=True, debug_info=None, **kwargs
):
    """
    Applies an optimisation function *fct* on a graph
    and not on the model.

    :param fct: function to optimise like
    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :param debug_info: debug information (private)
    :param kwargs: additional parameters
    return: new onnx model
    """
    if hasattr(onnx_model, "graph"):
        graph = fct(onnx_model.graph, debug_info=debug_info + ["GRAPH"], **kwargs)
        new_model = make_model(graph)
        new_model.ir_version = onnx_model.ir_version
        new_model.producer_name = onnx_model.producer_name
        new_model.producer_version = onnx_model.producer_version
        new_model.domain = onnx_model.domain
        new_model.model_version = onnx_model.model_version
        new_model.doc_string = onnx_model.doc_string
        if hasattr(onnx_model, "value_info"):
            graph.value_info.extend(onnx_model.value_info)
        while len(new_model.opset_import) > 0:
            new_model.opset_import.pop()
        for oimp in onnx_model.opset_import:
            op_set = new_model.opset_import.add()
            op_set.domain = oimp.domain
            op_set.version = oimp.version
        return new_model
    raise TypeError(
        "This function only works on 'ModelProto' anod not not on"
        " {}.".format(type(onnx_model))
    )


def _apply_remove_node_fct_node(fct, node, recursive, debug_info):
    """
    Applies an optimizing function on a subgraphs.

    :param node: onnx node
    :param recursive: does it in subgraphs as well
    :return: new node
    """
    if not hasattr(node, "attribute"):
        return node
    modified = 0
    new_atts = []
    for att in node.attribute:
        if att.name == "body":
            new_body = fct(
                att.g, recursive=recursive, debug_info=debug_info + [att.name]
            )
            new_atts.append(_make_att_graph(att.name, new_body))
            modified += 1
        else:
            new_atts.append(att)
    if modified > 0:
        new_node = _make_node(
            node.op_type, node.input, node.output, name=node.name, attributes=new_atts
        )
        return new_node
    return node


def _make_node(
    op_type, inputs, outputs, name=None, doc_string=None, domain=None, attributes=None
):
    """
    Constructs a NodeProto.

    :param op_type: (string): The name of the operator to construct
    :param inputs: list of input names
    :param outputs: list of output names
    :param name: optional unique identifier for NodeProto
    :param doc_string: optional documentation
        string for NodeProto
    :param domain: optional domain for NodeProto.
        If it's None, we will just use default domain (which is empty)
    :param attributes: the attributes of the node.  The acceptable values
        are documented in :func:`make_attribute`.
    :return: node
    """
    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)
    node.output.extend(outputs)
    if name:
        node.name = name
    if doc_string:
        node.doc_string = doc_string
    if domain is not None:
        node.domain = domain
    if isinstance(attributes, dict):
        if len(attributes) > 0:
            node.attribute.extend(
                make_attribute(key, value) for key, value in sorted(attributes.items())
            )
    elif attributes:
        for att in attributes:
            node.attribute.extend([att])
    return node


def _replace(name, old_name, new_name):
    if isinstance(old_name, dict) and new_name is None:
        return old_name.get(name, name)
    if name == old_name:
        return new_name
    return name


def _rename_node_input(onnx_node, old_name, new_name=None):
    """
    Renames an input from a node.

    :param onnx_node: onnx_node
    :param old_name: old name
    :param new_name: new name or None if *old_name* is a dictionary
    :return: new node
    """
    inputs = [_replace(name, old_name, new_name) for name in onnx_node.input]
    outputs = list(onnx_node.output)
    if hasattr(onnx_node, "attribute"):
        new_atts = []
        for att in onnx_node.attribute:
            if att.name == "body":
                new_body = _rename_graph_input(att.g, old_name, new_name)
                attr = AttributeProto()
                attr.name = att.name
                attr.g.CopyFrom(new_body)
                attr.type = AttributeProto.GRAPH
                new_atts.append(attr)
            else:
                new_atts.append(att)
        atts = new_atts
    else:
        atts = onnx_node.attribute
    node = _make_node(
        onnx_node.op_type,
        inputs,
        outputs,
        name=onnx_node.name,
        domain=onnx_node.domain,
        attributes=atts,
    )
    return node


def _rename_graph_output(graph, old_name, new_name):
    """
    Renames an output and adds an *Identity* node
    to connect the dots.

    :param graph: ONNX graph
    :return: modified graph
    """
    outputs = []
    for o in graph.output:
        if old_name != o.name:
            outputs.append(o)
        else:
            value_info = ValueInfoProto()
            value_info.name = new_name
            value_info.type.CopyFrom(o.type)
            if o.type.doc_string:
                value_info.doc_string = o.type.doc_string
            outputs.append(value_info)
    nodes = list(graph.node)
    nodes.append(_make_node("Identity", [old_name], [new_name]))
    new_graph = make_graph(nodes, graph.name, graph.input, outputs, graph.initializer)
    new_graph.value_info.extend(graph.value_info)
    return new_graph


def _rename_graph_input(graph, old_name, new_name):
    """
    Renames an input and adds an *Identity* node
    to connect the dots.

    :param graph: ONNX graph
    :return: modified graph
    """
    inputs = []
    for i in graph.input:
        if old_name != i.name:
            inputs.append(i)
        else:
            value_info = ValueInfoProto()
            value_info.name = new_name
            value_info.type.CopyFrom(i.type)
            if i.type.doc_string:
                value_info.doc_string = i.type.doc_string
            inputs.append(value_info)
    nodes = list(graph.node)
    nodes.append(_make_node("Identity", [new_name], [old_name]))
    new_graph = make_graph(nodes, graph.name, inputs, graph.output, graph.initializer)
    new_graph.value_info.extend(graph.value_info)
    return new_graph


def _make_att_graph(name, new_body):
    attr = AttributeProto()
    attr.name = name
    attr.g.CopyFrom(new_body)
    attr.type = AttributeProto.GRAPH
    return attr


def _rename_node_output(onnx_node, old_name, new_name):
    """
    Renames an output from a node.

    :param onnx_node: onnx_node
    :param old_name: old name
    :param new_name: new name
    :return: new node
    """
    inputs = list(onnx_node.input)
    outputs = [_replace(name, old_name, new_name) for name in onnx_node.output]
    if hasattr(onnx_node, "attribute"):
        new_atts = []
        for att in onnx_node.attribute:
            if att.name == "body":
                new_body = _rename_graph_output(att.g, old_name, new_name)
                new_atts.append(_make_att_graph(att.name, new_body))
            else:
                new_atts.append(att)
        atts = new_atts
    else:
        atts = onnx_node.attribute
    node = _make_node(
        onnx_node.op_type,
        inputs,
        outputs,
        name=onnx_node.name,
        domain=onnx_node.domain,
        attributes=atts,
    )
    return node
