# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

"""
This tool converts converts a model to mixed precision (float32->float16) while excluding nodes as needed to maintain
a certain accuracy.

Example usage:

    from onnxconverter_common import auto_mixed_precision
    import onnx

    model = onnx.load(model_path)

    # Could also use rtol/atol attributes directly instead of this
    def validate(res1, res2):
        for r1, r2 in zip(res1, res2):
            if not np.allclose(r1, r2, rtol=0.01, atol=0.001):
                return False
        return True

    model_fp16 = auto_convert_mixed_precision(model, test_data, validate, keep_io_types=True)
    onnx.save(model_fp16, "ouptut_path")

"""

import onnx
import numpy as np
from onnxconverter_common import float16
from onnx import helper, mapping
import copy


def auto_convert_mixed_precision(model, feed_dict, validate_fn=None, rtol=None, atol=None, keep_io_types=False):
    """
    Automatically converts a model to mixed precision, excluding the minimum number of nodes required to
    ensure valudate_fn returns True and/or results are equal according to rtol/atol
    """
    if rtol is None and atol is not None:
        rtol = 1e-5

    if atol is None and rtol is not None:
        atol = 1e-8

    if rtol is None and validate_fn is None:
        raise ValueError("Argument `validate_fn` and `rtol` cannot both be `None`.")

    def validate(res1, res2):
        if validate_fn is not None and not validate_fn(res1, res2):
            return False
        if rtol is not None:
            for r1, r2 in zip(res1, res2):
                if not np.allclose(r1, r2, rtol, atol):
                    return False
        return True

    model0 = onnx.shape_inference.infer_shapes(model)
    model0 = add_missing_dtypes_using_ort(model0, feed_dict)
    res0 = get_tensor_values_using_ort(model0, feed_dict)
    if not keep_io_types:
        feed_dict = {k: v.astype(np.float16) if v.dtype == np.float32 else v for k, v in feed_dict.items()}
    if not validate(res0, res0):
        raise ValueError("validation failed for original fp32 model")
    node_names = [n.name for n in model0.graph.node if n.op_type not in ["Loop", "If", "Scan"]]

    def run_attempt(node_block_list, return_model=False):
        print(node_block_list)
        model = float16.convert_float_to_float16(copy.deepcopy(model0), node_block_list=node_block_list,
                                                 keep_io_types=keep_io_types, disable_shape_infer=True)
        res1 = get_tensor_values_using_ort(model, feed_dict)
        if return_model:
            return validate(res0, res1), model
        else:
            valid = validate(res0, res1)
            print(valid)
            return valid

    if not run_attempt(node_names):
        raise ValueError("validation failed for model with all nodes in node_block_list")
    print("Sanity checks passed. Starting autoconvert.")
    segments = SegmentList(node_names)
    i = 0
    while segments.get_largest() is not None:
        seg = segments.get_largest()
        nodes_to_try = segments.get_nodes(seg)
        i += 1
        print("Running attempt %d excluding conversion of %s nodes" % (i, len(nodes_to_try)))
        if run_attempt(nodes_to_try):
            seg.good = True
            print("Attempt succeeded.")
        else:
            print("Attempt failed.")
            if seg.size == 1:
                seg.bad = True
            else:
                seg.split()
        print(segments)
    print("Done:", segments.get_nodes())
    valid, model = run_attempt(segments.get_nodes(), return_model=True)
    if not valid:
        raise ValueError("validation failed for final fp16 model")
    print("Final model validated successfully.")
    return model


def add_missing_dtypes_using_ort(model, feed_dict, outputs_per_iter=100):
    outputs = [out for node in model.graph.node for out in node.output]
    graph_io = [inp.name for inp in model.graph.input] + [out.name for out in model.graph.output]
    value_info_names = [info.name for info in model.graph.value_info]
    skip = set(graph_io + value_info_names)
    outputs = [out for out in outputs if out not in skip]
    print("Adding missing dtypes for %s outputs" % len(outputs))
    out_to_dtype = {}
    i = 0
    while i < len(outputs):
        outs = outputs[i:i + outputs_per_iter]
        vals = get_tensor_values_using_ort(model, feed_dict, outs)
        for out, val in zip(outs, vals):
            out_to_dtype[out] = mapping.NP_TYPE_TO_TENSOR_TYPE[val.dtype]
        i += outputs_per_iter
    for out, dtype in out_to_dtype.items():
        model.graph.value_info.append(helper.make_tensor_value_info(out, dtype, shape=None))
    return model


def get_tensor_values_using_ort(model, input_feed, output_names=None, sess_options=None):
    # delayed import to avoid taking a strong dependancy on onnxruntime
    import onnxruntime as ort
    if output_names is None:
        sess = ort.InferenceSession(model.SerializeToString(), sess_options, providers=['CUDAExecutionProvider'])
        return sess.run(None, input_feed)
    original_outputs = list(model.graph.output)
    while len(model.graph.output) > 0:
        model.graph.output.pop()
    for n in output_names:
        out = model.graph.output.add()
        out.name = n
    sess = ort.InferenceSession(model.SerializeToString(), sess_options, providers=['CUDAExecutionProvider'])
    try:
        return sess.run(output_names, input_feed)
    finally:
        while len(model.graph.output) > 0:
            model.graph.output.pop()
        for orig_out in original_outputs:
            out = model.graph.output.add()
            out.CopyFrom(orig_out)


class SegmentList:
    def __init__(self, node_names):
        self.node_names = node_names
        self.first = NodeSegment(len(node_names))

    def get_largest(self, adjacent_to_good=False):
        adjacent_to_good = False
        largest = None
        current = self.first
        prev_good = False
        while current is not None:
            can_use = not current.good and not current.bad
            if adjacent_to_good:
                next_good = current.next is not None and current.next.good
                can_use = can_use and (prev_good or next_good)
            if can_use and (largest is None or current.size > largest.size):
                largest = current
            prev_good = current.good
            current = current.next
        return largest

    def get_nodes(self, node_segment=None):
        i = 0
        current = self.first
        nodes = []
        while current is not None:
            if current is not node_segment and not current.good:
                nodes.extend(self.node_names[i:i + current.size])
            i += current.size
            current = current.next
        return nodes

    def __repr__(self):
        res = []
        current = self.first
        while current is not None:
            res.append(current)
            current = current.next
        return repr(res)


class NodeSegment:
    def __init__(self, size):
        self.size = size
        self.next = None
        self.good = False
        self.bad = False

    def split(self):
        new_size = self.size // 2
        new_segment = NodeSegment(self.size - new_size)
        new_segment.next = self.next
        self.next = new_segment
        self.size = new_size

    def __repr__(self):
        if self.good:
            return "*" + str(self.size) + "*"
        if self.bad:
            return "(" + str(self.size) + ")"
        return str(self.size)
