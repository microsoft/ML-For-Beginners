# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from io import BytesIO
import numpy as np
import onnx  # noqa
from onnx import shape_inference, TensorProto, ValueInfoProto
from onnx.numpy_helper import from_array, to_array
from onnx.helper import (
    make_tensor,
    make_node,
    make_tensor_value_info,
    make_graph,
    make_model,
)
from ..proto import get_latest_tested_opset_version
from onnx import onnx_pb as onnx_proto
from ..common._topology import Variable


def load_onnx_model(onnx_file_or_bytes):
    """
    Loads an *ONNX* file.

    :param onnx_file_or_bytes: *ONNX* file or bytes
    :return: *ONNX* model
    """
    if isinstance(onnx_file_or_bytes, str):
        with open(onnx_file_or_bytes, "rb") as f:
            return onnx.load(f)
    elif hasattr(onnx_file_or_bytes, "read"):
        return onnx.load(onnx_file_or_bytes)
    else:
        b = BytesIO(onnx_file_or_bytes)
        return onnx.load(b)


def save_onnx_model(model, filename=None):
    """
    Saves a model as a file or bytes.

    :param model: *ONNX* model
    :param filename: filename or None to return bytes
    :return: bytes
    """
    content = model.SerializeToString()
    if filename is not None:
        if hasattr(filename, "write"):
            filename.write(content)
        else:
            with open(filename, "wb") as f:
                f.write(content)
    return content


def enumerate_model_node_outputs(model, add_node=False):
    """
    Enumerates all the nodes of a model.

    :param model: ONNX graph
    :param add_node: if False, the function enumerates
        all output names from every node, otherwise, it
        enumerates tuple (output name, node)
    :return: enumerator
    """
    if not hasattr(model, "graph"):
        raise TypeError(
            "Parameter model is not an ONNX model but " "{}".format(type(model))
        )
    for node in model.graph.node:
        for out in node.output:
            yield (out, node) if add_node else out


def enumerate_model_initializers(model, add_node=False):
    """
    Enumerates all the initializers of a model.

    :param model: ONNX graph
    :param add_node: if False, the function enumerates
        all output names from every node, otherwise, it
        enumerates tuple (output name, node)
    :return: enumerator
    """
    for node in model.graph.initializer:
        yield (node.name, node) if add_node else node.name


def select_model_inputs_outputs(model, outputs=None, inputs=None):
    """
    Takes a model and changes its outputs.

    :param model: *ONNX* model
    :param inputs: new inputs
    :param outputs: new outputs
    :return: modified model

    The function removes unneeded files.
    """
    if inputs is not None:
        raise NotImplementedError("Parameter inputs cannot be empty.")
    if outputs is None:
        raise RuntimeError("Parameter outputs cannot be None.")
    if not isinstance(outputs, list):
        outputs = [outputs]

    mark_var = {}
    for out in enumerate_model_node_outputs(model):
        mark_var[out] = 0
    for inp in model.graph.input:
        mark_var[inp.name] = 0
    for out in outputs:
        if out not in mark_var:
            raise ValueError("Output '{}' not found in model.".format(out))
        mark_var[out] = 1

    nodes = model.graph.node[::-1]
    mark_op = {}
    for node in nodes:
        mark_op[node.name] = 0

    # We mark all the nodes we need to keep.
    nb = 1
    while nb > 0:
        nb = 0
        for node in nodes:
            if mark_op[node.name] == 1:
                continue
            mod = False
            for out in node.output:
                if mark_var[out] == 1:
                    mark_op[node.name] = 1
                    mod = True
                    break
            if not mod:
                continue

            nb += 1
            for inp in node.input:
                if mark_var.get(inp, 0) == 1:
                    continue
                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes[::-1] if mark_op[node.name] == 1]

    var_out = []
    for out in outputs:
        value_info = ValueInfoProto()
        value_info.name = out
        var_out.append(value_info)
    graph = make_graph(
        keep_nodes,
        model.graph.name,
        model.graph.input,
        var_out,
        model.graph.initializer,
    )
    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    if len(onnx_model.graph.input) != len(model.graph.input):
        raise RuntimeError(
            "Input mismatch {} != {}".format(len(onnx_model.input), len(model.input))
        )

    # fix opset import
    del onnx_model.opset_import[:]
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    return onnx_model


def infer_outputs(
    op_type, inputs, outputs=None, initializer=None, target_opset=None, **atts
):
    """
    Infers outputs type and shapes given an ONNX operator.
    """
    logger = getLogger("skl2onnx")
    logger.debug(
        "[infer_outputs] op_type=%r inputs=%r outputs=%r",
        op_type,
        [x.name for x in inputs],
        outputs,
    )
    if isinstance(op_type, str):
        required_outputs = []
        if outputs:
            for o in outputs:
                if hasattr(o, "onnx_name"):
                    required_outputs.append(o.onnx_name)
                elif isinstance(o, str):
                    required_outputs.append(o)
                else:
                    raise TypeError("Unable to require output {}.".format(o))
        node = make_node(
            op_type, [i.onnx_name for i in inputs], required_outputs, **atts
        )
        node = [node]
    elif hasattr(op_type, "nodes"):
        node = op_type.nodes
    else:
        raise RuntimeError(
            "Unable to build ONNX nodes from type {}.".format(type(op_type))
        )

    input_init = inputs.copy()
    if initializer:
        input_init.extend(initializer)
    onnx_inputs = []
    for input in input_init:
        if isinstance(input, Variable):
            onnx_type = input.type.to_onnx_type()
            tensor_type = onnx_type.tensor_type
            shape = [
                tensor_type.shape.dim[i].dim_value
                for i in range(len(tensor_type.shape.dim))
            ]
            inp = make_tensor_value_info(
                input.onnx_name, tensor_type.elem_type, tuple(shape)
            )
            onnx_inputs.append(inp)
        elif isinstance(input, onnx.TensorProto):
            v = make_tensor_value_info(
                input.name, input.data_type.real, list(d for d in input.dims)
            )
            onnx_inputs.append(v)
        elif isinstance(input, onnx.AttributeProto):
            value_info = ValueInfoProto()
            value_info.name = input.name
            onnx_type = onnx_proto.TypeProto()
            onnx_type.tensor_type.elem_type = input.type
            value_info.type.CopyFrom(onnx_type)
            onnx_inputs.append(value_info)
        else:
            onnx_inputs.append(input)

    graph = make_graph(node, "infer_shapes", onnx_inputs, [])
    original_model = make_model(graph, producer_name="skl2onnx")
    domains = {}
    for n in node:
        domains[n.domain] = max(domains.get(n.domain, 1), getattr(n, "op_version", 1))
    for i, (k, v) in enumerate(domains.items()):
        if i == 0 and len(original_model.opset_import) == 1:
            op_set = original_model.opset_import[0]
        else:
            op_set = original_model.opset_import.add()
        op_set.domain = k
        if target_opset:
            if isinstance(target_opset, dict):
                op_set.version = target_opset.get(k, get_latest_tested_opset_version())
            else:
                op_set.version = target_opset
        else:
            op_set.version = get_latest_tested_opset_version()

    try:
        inferred_model = shape_inference.infer_shapes(original_model)
    except RuntimeError as e:
        raise RuntimeError(
            "Unable to infer shape of node '{}'\n{}".format(op_type, original_model)
        ) from e
    all_shapes = Variable.from_pb(inferred_model.graph.value_info)
    used = set()
    for node in graph.node:
        for name in node.input:
            used.add(name)
    shapes = [shape for shape in all_shapes if shape.onnx_name not in used]
    if len(shapes) == 0:
        raise RuntimeError(
            f"Shape inference fails.\n*Inputs*\n{onnx_inputs}\n"
            f"*all_shapes*\n{all_shapes}'\n"
            f"*Model*\n{original_model}'"
        )
    logger.debug("[infer_outputs] shapes=%r", shapes)
    return shapes


def change_onnx_domain(model, ops):
    """
    Takes a model and changes its outputs.

    :param model: *ONNX* model
    :param ops: dictionary { optype: ('optype', 'new domain') }
    :return: modified model

    The function removes unneeded files.
    """
    nodes = model.graph.node
    for node in nodes:
        rep = ops.get(node.op_type, None)
        if rep is None:
            continue
        node.op_type = rep[0]
        node.domain = rep[1]

    graph = make_graph(
        nodes,
        model.graph.name,
        model.graph.input,
        model.graph.output,
        model.graph.initializer,
    )
    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    if len(onnx_model.graph.input) != len(model.graph.input):
        raise RuntimeError(
            "Input mismatch {} != {}".format(len(onnx_model.input), len(model.input))
        )

    # fix opset import
    domain_set = set()
    has_domain = False
    del onnx_model.opset_import[:]
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version
        domain_set.add(oimp.domain)
        if not has_domain:
            has_domain = oimp.domain in domain_set
    for v in ops.values():
        if v[1] not in domain_set:
            op_set = onnx_model.opset_import.add()
            op_set.domain = v[1]
            op_set.version = 1
    return onnx_model


def add_output_initializer(model_onnx, name, value, suffix="_init"):
    """
    Add a constant and link it to one output.
    It allows the user to store arrays into the graph
    and retrieve them when using it.
    The initializer is named `name + suffix`, the output
    is named `name`.

    :param model_onnx: ONNX graph
    :param name: initializer name (initializer name, output name)
    :param value: array to store
    :param suffix: name of the initializer
    :return: new model

    It is possible to add multiple constant by using list:
    ``add_output_initializer(model_onnx, ['name1', 'name2'], [v1, v2])``.
    """
    if isinstance(name, str):
        name_list = [name]
        value_list = [value]
    else:
        name_list = name
        value_list = value

    if len(name_list) != len(value_list):
        raise ValueError(
            "Mismatched names and values. There are %d names and %d values."
            "" % (len(name_list), len(value_list))
        )

    nodes = list(model_onnx.graph.node)
    inits = list(model_onnx.graph.initializer)
    outputs = list(model_onnx.graph.output)

    for name, value in zip(name_list, value_list):
        name_output = name
        name_init = name + suffix
        names = set(i.name for i in model_onnx.graph.initializer)
        if name_output in names or name_init in names:
            raise ValueError(
                "Names %r or %r is already taken by an initializer: %r."
                % (name_output, name_init, ", ".join(sorted(names)))
            )
        names = set(i.name for i in model_onnx.graph.output)
        if name_output in names or name_init in names:
            raise ValueError(
                "Names %r or %r is already taken by an output: %r."
                % (name_output, name_init, ", ".join(sorted(names)))
            )
        names = set(i.name for i in model_onnx.graph.input)
        if name_output in names or name_init in names:
            raise ValueError(
                "Names %r or %r is already taken by an output: %r."
                % (name_output, name_init, ", ".join(sorted(names)))
            )

        try:
            cst = from_array(value, name=name_init)
        except RuntimeError as e:
            st = str(value.dtype).lower()
            if st.startswith("u") or st.startswith("<u"):
                cst_value = np.array([s.encode("utf-8") for s in value])
                cst = make_tensor(
                    name_init,
                    data_type=TensorProto.STRING,
                    dims=value.shape,
                    vals=list(cst_value),
                )
            else:
                raise e

        inits.append(cst)

        outputs.append(make_tensor_value_info(name_output, cst.data_type, cst.dims))

        nodes.append(make_node("Identity", [name_init], [name_output]))

    graph = make_graph(
        nodes, model_onnx.graph.name, model_onnx.graph.input, outputs, inits
    )

    onnx_model = make_model(graph)
    onnx_model.ir_version = model_onnx.ir_version
    onnx_model.producer_name = model_onnx.producer_name
    onnx_model.producer_version = model_onnx.producer_version
    onnx_model.domain = model_onnx.domain
    onnx_model.model_version = model_onnx.model_version
    onnx_model.doc_string = model_onnx.doc_string
    if len(model_onnx.metadata_props) > 0:
        values = {p.key: p.value for p in model_onnx.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    if len(onnx_model.graph.input) != len(model_onnx.graph.input):
        raise RuntimeError(
            "Input mismatch {} != {}".format(
                len(onnx_model.input), len(model_onnx.input)
            )
        )

    # fix opset import
    del onnx_model.opset_import[:]
    for oimp in model_onnx.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version
    return onnx_model


def get_initializers(model_onnx):
    """
    Retrieves the list of initializers in a model in a
    dictionary `{ name: value }`.
    """
    res = {}
    for init in model_onnx.graph.initializer:
        res[init.name] = to_array(init)
    return res


def update_onnx_initializers(model_onnx, new_inits):
    """
    Updates initializer in a ONNX model.

    :param model_onnx: ONNX model
    :param new_inits: new initializers
    :return: list of updated initializers
    """
    updated = []
    replace_weights = []
    replace_indices = []
    for i, w in enumerate(model_onnx.graph.initializer):
        if w.name in new_inits:
            replace_weights.append(from_array(new_inits[w.name], w.name))
            replace_indices.append(i)
            updated.append(w.name)
    replace_indices.sort(reverse=True)
    for w_i in replace_indices:
        del model_onnx.graph.initializer[w_i]
    model_onnx.graph.initializer.extend(replace_weights)
    return updated
