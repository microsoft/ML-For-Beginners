# SPDX-License-Identifier: Apache-2.0

"""
Place holder for all ONNX operators.
"""
import sys
import os
import numpy as np

try:
    from scipy.sparse import coo_matrix
except ImportError:
    from scipy.sparse.coo import coo_matrix
import onnx
from ..common.data_types import DataType
from ..common._topology import Variable
from .automation import get_rst_doc
from ._cache import cache_folder


def ClassFactory(
    class_name,
    op_name,
    inputs,
    outputs,
    input_range,
    output_range,
    domain,
    attr_names,
    doc,
    deprecated,
    since_version,
    past_version,
):
    from .onnx_operator import OnnxOperator, OnnxOperatorItem

    def __init__(self, *args, **kwargs):
        op_version = kwargs.pop("op_version", None)
        if isinstance(op_version, dict):
            op_version = op_version.get(domain, None)

        if op_version is None:
            if len(args) == 0 and input_range[0] == input_range[1]:
                args = [_[0] for _ in self.__class__.expected_inputs]
            if not (input_range[0] <= len(args) <= input_range[1]):
                raise RuntimeError(
                    "Unexpected number of inputs, "
                    "got {}, expecting {} for operator "
                    "'{}'.".format(len(args), len(inputs), op_name)
                )

        attr_names = self.attr_names
        if "_" in self.__class__.__name__:
            op_version_class = int(self.__class__.__name__.split("_")[-1])
            if op_version is None:
                op_version = op_version_class
            try:
                op_version = min(op_version, op_version_class)
            except TypeError:
                raise TypeError(
                    "Could not compare versions {} ? {} for "
                    "class '{}' since_version {}. Parameter 'op_version' "
                    "is probably missing when the class "
                    "is instantiated.".format(
                        op_version, op_version_class, class_name, since_version
                    )
                )
        else:
            op_version_class = None

        # By default, the op_version is None.
        # None means the latest available.
        if op_version is None:
            op_version = since_version

        found = None
        if op_version is not None:
            # attr_names refers to the most recent version of
            # this operator. We may need an older one.
            for op in range(op_version, 0, -1):
                name = "{}_{}".format(self.__class__.__name__, op)
                if name in self.past_version:
                    found = (name, op)
                    attr_names = self.past_version[name].attr_names
                    break
        if (
            op_version_class is not None
            and found is not None
            and found[-1] != op_version_class
        ):
            raise RuntimeError(
                "op_version={} does not refer to the same opset as the class "
                "name ('{}').".format(op_version, self.__class__.__name__)
            )
        for key in kwargs:
            if key in {
                "output_names",
                "op_version",
                "domain",
                "ir_version",
                "global_context",
                "clear_subgraph_inputs",
            }:
                continue
            if key not in attr_names:
                raise TypeError(
                    "Argument '%s' not valid for '%s' opset=%s."
                    % (key, op_name, op_version)
                )

        if op_version is not None:
            kwargs["op_version"] = op_version
        # This class can only be created by a user. Let's check
        # types are either a variable, an operator or an array.
        for i, a in enumerate(args):
            if isinstance(a, tuple):
                if len(a) != 2:
                    raise TypeError(
                        "Input %r is a tuple or class %r, it must have two "
                        "elements (name, type) not %r." % (i, class_name, a)
                    )
                if not isinstance(a[0], str) or not isinstance(a[1], DataType):
                    raise TypeError(
                        "Input %r is a tuple or class %r, it must be a tuple "
                        "(name, type) not %r." % (i, class_name, a)
                    )
                continue
            if not isinstance(
                a,
                (Variable, OnnxOperator, np.ndarray, str, OnnxOperatorItem, coo_matrix),
            ):
                raise TypeError(
                    "Unexpected type %r for input %r of operator %r. "
                    "It must be an instance of Variable (or a string), "
                    "OnnxOperator, OnnxOperatorItem, numpy.ndarray, "
                    "coo_matrix)." % (type(a), i, class_name)
                )
        OnnxOperator.__init__(self, *args, **kwargs)

    newclass = type(
        class_name,
        (OnnxOperator,),
        {
            "__init__": __init__,
            "__doc__": doc,
            "expected_inputs": inputs,
            "expected_outputs": outputs,
            "operator_name": op_name,
            "input_range": input_range,
            "output_range": output_range,
            "domain": domain,
            "is_deprecated": deprecated,
            "since_version": since_version,
            "past_version": past_version,
            "attr_names": attr_names,
            "__module__": __name__,
        },
    )
    return newclass


def dynamic_class_creation(cache=False):
    """
    Automatically generates classes for each of the operators
    module *onnx* defines and described at
    `Operators
    <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
    and `Operators
    <https://github.com/onnx/onnx/blob/master/docs/
    Operators-ml.md>`_.
    """
    cache_dir = cache_folder()
    res = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.support_level == schema.SupportType.EXPERIMENTAL:
            # Skips experimental operators.
            continue
        # Multiple version can coexist. The last one is kept.
        if schema.name in res:
            if schema.since_version > res[schema.name].since_version:
                # We keep the most recent one.
                res[schema.name] = schema
        else:
            res[schema.name] = schema
        res[schema.name + "_" + str(schema.since_version)] = schema
    cls = {}

    def _c(obj, label, i):
        name = "%s%d" % (obj.name or label, i)
        try:
            tys = obj.type_str or ""
        except AttributeError:
            tys = obj.typeStr or ""
        return (name, tys)

    for name in sorted(res):
        schema = res[name]
        inputs = [_c(o, "I", i) for i, o in enumerate(schema.inputs)]
        outputs = [_c(o, "O", i) for i, o in enumerate(schema.outputs)]
        args = [p for p in schema.attributes]

        if "_" in name:
            class_name = "Onnx" + name
        else:
            class_name = "Onnx" + schema.name

        filename = os.path.join(
            cache_dir, schema.name + "_" + str(schema.since_version) + ".rst"
        )
        if not cache and os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                doc = f.read()
        else:
            doc = get_rst_doc(schema)
            if cache:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(doc)

        cl = ClassFactory(
            class_name,
            schema.name,
            inputs,
            outputs,
            [schema.min_input, schema.max_input],
            [schema.min_output, schema.max_output],
            schema.domain,
            args,
            "**Version**" + doc.split("**Version**")[-1],
            getattr(schema, "deprecated", False),
            schema.since_version,
            {},
        )
        cls[class_name] = cl

    # Retrieves past classes.
    for name in cls:
        if "_" not in name:
            continue
        main, version = name.split("_")
        last = cls[main]
        last.past_version[name] = cls[name]

    return cls


def _update_module():
    """
    Dynamically updates the module with operators defined
    by *ONNX*.
    """
    res = dynamic_class_creation()
    this = sys.modules[__name__]
    for k, v in res.items():
        setattr(this, k, v)


_update_module()


def OnnxReduceSumApi11(*x, axes=None, keepdims=1, op_version=None, output_names=None):
    """
    Adds operator ReduceSum with opset>=13 following API from opset 12.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 13:
        if axes is None:
            return OnnxReduceSum(  # noqa
                *x, keepdims=keepdims, op_version=op_version, output_names=output_names
            )
        return OnnxReduceSum(  # noqa
            *x,
            np.array(axes, dtype=np.int64),
            keepdims=keepdims,
            op_version=op_version,
            output_names=output_names,
        )
    if op_version >= 11:
        if axes is None:
            return OnnxReduceSum_11(  # noqa
                *x, keepdims=keepdims, op_version=op_version, output_names=output_names
            )
        return OnnxReduceSum_11(  # noqa
            *x,
            axes=axes,
            keepdims=keepdims,
            op_version=op_version,
            output_names=output_names,
        )
    if axes is None:
        return OnnxReduceSum_1(
            *x,
            keepdims=keepdims,  # noqa
            op_version=op_version,
            output_names=output_names,
        )
    return OnnxReduceSum_1(
        *x,
        axes=axes,
        keepdims=keepdims,  # noqa
        op_version=op_version,
        output_names=output_names,
    )


def OnnxReduceAnyApi18(
    cl18, cl13, cl11, cl1, *x, axes=None, keepdims=1, op_version=None, output_names=None
):
    """
    Adds operator Reduce* with opset>=18 following API from opset 17.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 18:
        if axes is None:
            return cl18(  # noqa
                *x, keepdims=keepdims, op_version=op_version, output_names=output_names
            )
        return cl18(  # noqa
            *x,
            np.array(axes, dtype=np.int64),
            keepdims=keepdims,
            op_version=op_version,
            output_names=output_names,
        )
    if op_version >= 13:
        if axes is None:
            return cl13(
                *x,
                keepdims=keepdims,  # noqa
                op_version=op_version,
                output_names=output_names,
            )
        return cl13(
            *x,
            axes=axes,
            keepdims=keepdims,  # noqa
            op_version=op_version,
            output_names=output_names,
        )
    if op_version >= 11:
        if axes is None:
            return cl11(
                *x,
                keepdims=keepdims,  # noqa
                op_version=op_version,
                output_names=output_names,
            )
        return cl11(
            *x,
            axes=axes,
            keepdims=keepdims,  # noqa
            op_version=op_version,
            output_names=output_names,
        )
    if axes is None:
        return cl1(
            *x,
            keepdims=keepdims,  # noqa
            op_version=op_version,
            output_names=output_names,
        )
    return cl1(
        *x,
        axes=axes,
        keepdims=keepdims,  # noqa
        op_version=op_version,
        output_names=output_names,
    )


def OnnxReduceSumSquareApi18(
    *x, axes=None, keepdims=1, op_version=None, output_names=None
):
    """
    Adds operator ReduceSumSquare with opset>=18 following API from opset 17.
    """
    if axes is None or not isinstance(axes, (list, np.ndarray)):
        raise TypeError(f"axes must be a list or an array not {type(axes)}.")
    return OnnxReduceAnyApi18(
        OnnxReduceSumSquare,
        OnnxReduceSumSquare_13,  # noqa
        OnnxReduceSumSquare_11,
        OnnxReduceSumSquare_1,  # noqa
        *x,
        axes=axes,
        keepdims=keepdims,
        op_version=op_version,
        output_names=output_names,
    )


def OnnxReduceMeanApi18(*x, axes=None, keepdims=1, op_version=None, output_names=None):
    """
    Adds operator ReduceMean with opset>=18 following API from opset 17.
    """
    return OnnxReduceAnyApi18(
        OnnxReduceMean,
        OnnxReduceMean_13,  # noqa
        OnnxReduceMean_11,
        OnnxReduceMean_1,  # noqa
        *x,
        axes=axes,
        keepdims=keepdims,
        op_version=op_version,
        output_names=output_names,
    )


def OnnxReduceMaxApi18(*x, axes=None, keepdims=1, op_version=None, output_names=None):
    """
    Adds operator ReduceMean with opset>=18 following API from opset 17.
    """
    return OnnxReduceAnyApi18(
        OnnxReduceMax,
        OnnxReduceMax_13,  # noqa
        OnnxReduceMax_11,
        OnnxReduceMax_1,  # noqa
        *x,
        axes=axes,
        keepdims=keepdims,
        op_version=op_version,
        output_names=output_names,
    )


def OnnxReduceLogSumExpApi18(
    *x, axes=None, keepdims=1, op_version=None, output_names=None
):
    """
    Adds operator ReduceMean with opset>=18 following API from opset 17.
    """
    return OnnxReduceAnyApi18(
        OnnxReduceLogSumExp,
        OnnxReduceLogSumExp_13,  # noqa
        OnnxReduceLogSumExp_11,
        OnnxReduceLogSumExp_1,  # noqa
        *x,
        axes=axes,
        keepdims=keepdims,
        op_version=op_version,
        output_names=output_names,
    )


def OnnxReduceL2Api18(*x, axes=None, keepdims=1, op_version=None, output_names=None):
    """
    Adds operator ReduceMean with opset>=18 following API from opset 17.
    """
    return OnnxReduceAnyApi18(
        OnnxReduceL2,
        OnnxReduceL2_13,  # noqa
        OnnxReduceL2_11,
        OnnxReduceL2_1,  # noqa
        *x,
        axes=axes,
        keepdims=keepdims,
        op_version=op_version,
        output_names=output_names,
    )


def OnnxSplitApi18(
    *x, axis=0, split=None, num_outputs=None, op_version=None, output_names=None
):
    """
    Adds operator Split with opset>=13 following API from opset 11.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 18:
        if split is None:
            if num_outputs is None:
                if output_names is None:
                    raise RuntimeError(
                        "split or num_outputs or output_names "
                        "must be specified since opset 18."
                    )
                num_outputs = len(output_names)
            if num_outputs is None:
                raise AttributeError("num_outputs cannot be None for Split-18.")
            return OnnxSplit_18(  # noqa
                *x,
                axis=axis,
                op_version=op_version,
                num_outputs=num_outputs,
                output_names=output_names,
            )
        if num_outputs is None:
            return OnnxSplit_18(  # noqa
                *x,
                np.array(split, dtype=np.int64),
                axis=axis,
                op_version=op_version,
                output_names=output_names,
            )
        return OnnxSplit_18(  # noqa
            *x,
            np.array(split, dtype=np.int64),
            axis=axis,
            num_outputs=num_outputs,
            op_version=op_version,
            output_names=output_names,
        )
    if op_version >= 13:
        if split is None:
            return OnnxSplit_13(  # noqa
                *x, axis=axis, op_version=op_version, output_names=output_names
            )
        return OnnxSplit_13(  # noqa
            *x,
            np.array(split, dtype=np.int64),
            axis=axis,
            op_version=op_version,
            output_names=output_names,
        )
    if op_version >= 11:
        if split is None:
            return OnnxSplit_11(  # noqa
                *x, axis=axis, op_version=op_version, output_names=output_names
            )
        return OnnxSplit_11(  # noqa
            *x, split=split, axis=axis, op_version=op_version, output_names=output_names
        )
    if split is None:
        return OnnxSplit_2(  # noqa
            *x, axis=axis, op_version=op_version, output_names=output_names
        )
    return OnnxSplit_2(
        *x,
        split=split,
        axis=axis,  # noqa
        op_version=op_version,
        output_names=output_names,
    )


def OnnxSqueezeApi11(*x, axes=None, op_version=None, output_names=None):
    """
    Adds operator Squeeze with opset>=13 following API from opset 11.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 13:
        return OnnxSqueeze(  # noqa
            *x,
            np.array(axes, dtype=np.int64),
            op_version=op_version,
            output_names=output_names,
        )
    if op_version >= 11:
        return OnnxSqueeze_11(  # noqa
            *x, axes=axes, op_version=op_version, output_names=output_names
        )
    return OnnxSqueeze_1(
        *x, axes=axes, op_version=op_version, output_names=output_names  # noqa
    )


def OnnxUnsqueezeApi11(*x, axes=None, op_version=None, output_names=None):
    """
    Adds operator Unsqueeze with opset>=13 following API from opset 11.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 13:
        return OnnxUnsqueeze(  # noqa
            *x,
            np.array(axes, dtype=np.int64),
            op_version=op_version,
            output_names=output_names,
        )
    if op_version >= 11:
        return OnnxUnsqueeze_11(  # noqa
            *x, axes=axes, op_version=op_version, output_names=output_names
        )
    return OnnxUnsqueeze_1(
        *x, axes=axes, op_version=op_version, output_names=output_names  # noqa
    )


def OnnxReduceL2_typed(
    dtype, x, axes=None, keepdims=1, op_version=None, output_names=None
):
    """
    Adds operator ReduceL2 for float or double.
    """
    if dtype == np.float32:
        return OnnxReduceL2Api18(  # noqa
            x,
            axes=axes,
            keepdims=keepdims,
            op_version=op_version,
            output_names=output_names,
        )
    x2 = OnnxMul(x, x, op_version=op_version)  # noqa
    red = OnnxReduceSumApi11(x2, axes=[1], keepdims=1, op_version=op_version)
    return OnnxSqrt(red, op_version=op_version, output_names=output_names)  # noqa


def OnnxReshapeApi13(*x, allowzero=0, op_version=None, output_names=None):
    """
    Adds operator Reshape with opset>=14 following API from opset 13.
    """
    if op_version is None:
        raise RuntimeError("op_version must be specified.")
    if op_version is None or op_version >= 14:
        return OnnxReshape(  # noqa
            *x, allowzero=allowzero, op_version=op_version, output_names=output_names
        )
    if op_version >= 13:
        return OnnxReshape_13(  # noqa
            *x, op_version=op_version, output_names=output_names
        )
    return OnnxReshape_5(*x, op_version=op_version, output_names=output_names)  # noqa
