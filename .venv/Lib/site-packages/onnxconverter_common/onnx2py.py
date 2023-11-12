# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

"""
Converts onnx model into model.py file for easy editing. Resulting model.py file uses onnx.helper library to
recreate the original onnx model. Constant tensors with more than 10 elements are saved into .npy
files in location model/const#_tensor_name.npy

Example usage:
python -m onnxconverter_common.onnx2py my_model.onnx my_model.py
"""

import sys
import onnx
import collections
import inspect
from collections import OrderedDict
from onnx import helper, numpy_helper, TensorProto, external_data_helper
import numpy as np
import os

from .pytracing import TracingObject

needed_types = set()
const_dir = None
const_counter = None

np_traced = TracingObject("np", np)
helper_traced = TracingObject("helper", helper)
numpy_helper_traced = TracingObject("numpy_helper", numpy_helper)
TensorProtoTraced = TracingObject("TensorProto", TensorProto)
os_traced = TracingObject("os", os)


# <Helpers> These can be inlined into the output script #

def clear_field(proto, field):
    proto.ClearField(field)
    return proto


def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))


def make_external_tensor(name, data_type, dims, raw_data=None, **kwargs):
    tensor = TensorProto()
    tensor.data_type = data_type
    tensor.name = name
    tensor.dims.extend(dims)
    tensor.raw_data = raw_data if raw_data is not None else b''
    external_data_helper.set_external_data(tensor, **kwargs)
    if raw_data is None:
        tensor.ClearField("raw_data")
    order_repeated_field(tensor.external_data, 'key', kwargs.keys())
    return tensor


def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == '':
        node.doc_string = ''
    order_repeated_field(node.attribute, 'name', kwargs.keys())
    return node


def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == '':
        graph.doc_string = ''
    return graph

# </Helpers> #


clear_field_traced = TracingObject("clear_field", clear_field)
make_external_tensor_traced = TracingObject("make_external_tensor", make_external_tensor)
make_node_traced = TracingObject("make_node", make_node)
make_graph_traced = TracingObject("make_graph", make_graph)
DATA_DIR_TRACED = None


def convert_tensor_type(i):
    return getattr(TensorProtoTraced, TensorProto.DataType.Name(i))


def convert_field(field):
    global needed_types
    if isinstance(field, (int, str, float, bytes)):
        return field
    elif isinstance(field, onnx.GraphProto):
        converted = convert_graph(field)
    elif isinstance(field, onnx.ModelProto):
        converted = convert_model(field)
    elif isinstance(field, onnx.NodeProto):
        converted = convert_node(field)
    elif isinstance(field, onnx.TensorProto):
        converted = convert_tensor(field)
    elif isinstance(field, onnx.ValueInfoProto):
        converted = convert_value_info(field)
    elif isinstance(field, onnx.OperatorSetIdProto):
        converted = convert_operatorsetid(field)
    elif isinstance(field, collections.abc.Iterable):
        return list(convert_field(x) for x in field)
    else:
        # Missing handler needs to be added
        t = str(type(field))
        needed_types.add(t)
        return field
    # Verify that resulting protobuf is identical to original
    # assert TracingObject.get_py_obj(converted) == field
    return converted


def convert_value_info(val_info):
    name = val_info.name
    is_sequence_type = val_info.type.HasField('sequence_type')
    if is_sequence_type:
        tensor_type = val_info.type.sequence_type.elem_type.tensor_type
    else:
        tensor_type = val_info.type.tensor_type
    elem_type = convert_tensor_type(tensor_type.elem_type)
    kwargs = OrderedDict()

    def convert_shape_dim(d):
        if d.HasField("dim_value"):
            return d.dim_value
        if d.HasField("dim_param"):
            return d.dim_param
        return None

    def convert_shape_denotation(d):
        if d.HasField("denotation"):
            return d.denotation
        return None

    if tensor_type.HasField("shape"):
        kwargs["shape"] = [convert_shape_dim(d) for d in tensor_type.shape.dim]
    else:
        kwargs["shape"] = None
    if any(d.HasField("denotation") for d in tensor_type.shape.dim):
        kwargs["shape_denotation"] = [convert_shape_denotation(d) for d in tensor_type.shape.dim]

    if val_info.HasField("doc_string"):
        kwargs["doc_string"].doc_string

    if is_sequence_type:
        return helper_traced.make_sequence_value_info(name, elem_type, **kwargs)
    else:
        return helper_traced.make_tensor_value_info(name, elem_type, **kwargs)


def convert_operatorsetid(opsetid):
    version = opsetid.version
    if opsetid.HasField("domain"):
        domain = opsetid.domain
        return helper_traced.make_operatorsetid(domain, version)
    else:
        return clear_field_traced(helper_traced.make_operatorsetid('', version), 'domain')


def convert_external_tensor(tensor):
    kwargs = OrderedDict()
    if tensor.HasField("raw_data"):
        kwargs["raw_data"] = tensor.raw_data
    if tensor.external_data:
        for d in tensor.external_data:
            kwargs[d.key] = d.value
    return make_external_tensor_traced(tensor.name, tensor.data_type, tensor.dims, **kwargs)


def convert_tensor(tensor):
    global const_dir, const_counter
    if tensor.data_location == TensorProto.EXTERNAL:
        return convert_external_tensor(tensor)
    np_data = numpy_helper.to_array(tensor)
    if np.product(np_data.shape) <= 10:
        return numpy_helper_traced.from_array(np_data, name=tensor.name)
    dtype = np_data.dtype
    if dtype == object:
        np_data = np_data.astype(str)
    os.makedirs(const_dir, exist_ok=True)
    name = "const" + str(const_counter)
    if tensor.name and len(tensor.name) < 100:
        # Avoid path length limit on windows
        name = name + "_" + tensor.name
    for c in '~"#%&*:<>?/\\{|}':
        name = name.replace(c, '_')
    const_path = "%s/%s.npy" % (const_dir, name)
    np.save(const_path, np_data)
    data_path = os_traced.path.join(DATA_DIR_TRACED, name + '.npy')
    const_counter += 1
    np_dtype = str(dtype)
    np_shape = list(np_data.shape)
    np_array = np_traced.load(data_path).astype(np_dtype).reshape(np_shape)
    return numpy_helper_traced.from_array(np_array, name=tensor.name)


def convert_node(node):
    fields = OrderedDict((f[0].name, f[1]) for f in node.ListFields())
    attributes = fields.pop("attribute", [])
    attrs = OrderedDict((a.name, convert_field(helper.get_attribute_value(a))) for a in attributes)
    fields = OrderedDict((f, convert_field(v)) for f, v in fields.items())
    op_type = fields.pop("op_type")
    if op_type == "Cast" and "to" in attrs:
        attrs["to"] = convert_tensor_type(attrs["to"])
    inputs = fields.pop("input", [])
    outputs = fields.pop("output", [])
    return make_node_traced(op_type, inputs=inputs, outputs=outputs, **fields, **attrs)


def convert_graph(graph):
    fields = OrderedDict((f[0].name, convert_field(f[1])) for f in graph.ListFields())
    nodes = fields.pop("node", [])
    name = fields.pop("name")
    inputs = fields.pop("input", [])
    outputs = fields.pop("output", [])
    return make_graph_traced(name=name, inputs=inputs, outputs=outputs, **fields, nodes=nodes)


def convert_model(model):
    fields = OrderedDict((f[0].name, convert_field(f[1])) for f in model.ListFields())
    graph = fields.pop("graph")
    opset_imports = fields.pop("opset_import", [])
    return helper_traced.make_model(opset_imports=opset_imports, **fields, graph=graph)


def clear_directory(path):
    for f in os.listdir(path):
        if f.endswith(".npy"):
            os.remove(os.path.join(path, f))
    try:
        # Delete if empty
        os.rmdir(path)
    except OSError:
        pass


class MissingHandlerException(Exception):
    pass


FILE_HEADER = '''"""
Run this script to recreate the original onnx model.
Example usage:
python %s.py out_model_path.onnx
"""'''


def convert(model, out_path):
    global needed_types, const_dir, const_counter, DATA_DIR_TRACED
    needed_types = set()
    if out_path.endswith(".py"):
        out_path = out_path[:-3]
    if os.path.exists(out_path):
        clear_directory(out_path)
    const_dir = out_path
    const_dir_name = os.path.basename(out_path)
    const_counter = 0
    TracingObject.reset_cnt(clear_field_traced)
    TracingObject.reset_cnt(make_external_tensor_traced)
    DATA_DIR_TRACED = TracingObject("DATA_DIR", const_dir)

    model_trace = convert_field(model)

    code = FILE_HEADER % os.path.basename(out_path) + "\n"
    code += "\nfrom onnx import helper, numpy_helper, TensorProto\n"
    if TracingObject.get_cnt(make_external_tensor_traced):
        code += ", external_data_helper"
    code += "\n"
    code += "import onnx\n"
    code += "import numpy as np\n"
    code += "import sys\n"
    if os.path.exists(const_dir):
        code += "import os\n"
        code += "\nDATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), %r)\n" % const_dir_name
    if TracingObject.get_cnt(clear_field_traced):
        code += "\n" + inspect.getsource(clear_field)
    code += "\n" + inspect.getsource(order_repeated_field)
    if TracingObject.get_cnt(make_external_tensor_traced):
        code += "\n" + inspect.getsource(make_external_tensor)
    code += "\n" + inspect.getsource(make_node)
    code += "\n" + inspect.getsource(make_graph)
    code += "\n" + "model = " + repr(model_trace) + "\n"
    code += "\nif __name__ == '__main__' and len(sys.argv) == 2:\n"
    code += "    _, out_path = sys.argv\n"
    if TracingObject.get_cnt(make_external_tensor_traced):
        code += "    with open(out_path, 'wb') as f:\n"
        code += "        f.write(model.SerializeToString())\n"
    else:
        code += "    onnx.save(model, out_path)\n"
    with open(out_path + ".py", "wt", encoding='utf8') as file:
        file.write(code)
    if needed_types:
        raise MissingHandlerException("Missing handler for types: %s" % list(needed_types))
    return model_trace


def main():
    _, in_path, out_path = sys.argv
    if not out_path.endswith(".py"):
        out_path = out_path + ".py"

    model = onnx.load(in_path, load_external_data=False)
    try:
        model_trace = convert(model, out_path)
        if TracingObject.get_py_obj(model_trace).SerializeToString() == model.SerializeToString():
            print("\nConversion successful. Converted model is identical.\n")
        else:
            print("\nWARNING: Conversion succeeded but converted model is not identical. "
                  "Difference might be trivial.\n")
    except MissingHandlerException as e:
        print("ERROR:", e)

    print("Model saved to", out_path)
    print("Run 'python %s output.onnx' to generate ONNX file" % out_path)
    print("Import the model with 'from %s import model'" % os.path.basename(out_path[:-3]))


if __name__ == '__main__':
    main()
