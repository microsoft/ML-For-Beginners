# SPDX-License-Identifier: Apache-2.0


import onnx
from onnx.helper import make_graph, make_model
from onnx.defs import get_all_schemas_with_history


def _build_op_version():
    res = {}
    for schema in get_all_schemas_with_history():
        dom = schema.domain
        name = schema.name
        vers = schema.since_version
        if (dom, name) not in res:
            res[dom, name] = set()
        res[dom, name].add(vers)
    op_versions = {}
    for k, v in res.items():
        op_versions[k] = list(sorted(v))
    return op_versions


def _check_possible_opset(versions, nodes, domain, old_opset, new_opset):
    if old_opset >= new_opset:
        raise RuntimeError("Condition new_opset > old_opset is not true.")
    for node in nodes:
        name = node.op_type
        vers = versions.get((domain, name), None)
        if vers is None:
            # custom operator
            continue
        for v in vers:
            if v > old_opset and v <= new_opset:
                raise RuntimeError(
                    "Operator '{}' (domain: '{}') was updated "
                    "in opset {} in ]{}, {}]."
                    "".format(name, domain, v, old_opset, new_opset)
                )


def upgrade_opset_number(model, new_opsets):
    """
    Upgrades the domain opsets. It checks if that's
    possible. It means ONNX specifications do not
    propose a new version of every operator between
    the current opset and the new one.

    :param model: *ONNX* model
    :param new_opdets: integer or dictionary
        ``{ domain: opset }``
    :return: modified model
    """
    if isinstance(new_opsets, int):
        new_opsets = {"": new_opsets}
    graph = make_graph(
        model.graph.node,
        model.graph.name,
        model.graph.input,
        model.graph.input,
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

    # fix opset import
    versions = _build_op_version()
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        if oimp.domain in new_opsets:
            _check_possible_opset(
                versions,
                model.graph.node,
                oimp.domain,
                oimp.version,
                new_opsets[oimp.domain],
            )
            op_set.version = new_opsets[oimp.domain]
        else:
            op_set.version = oimp.version
    return onnx_model
