# SPDX-License-Identifier: Apache-2.0

"""
Place holder for all ONNX operators.
"""
import sys
import textwrap
from sklearn.pipeline import Pipeline, FeatureUnion

try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # ColumnTransformer was introduced in 0.20.
    ColumnTransformer = None
from .onnx_subgraph_operator_mixin import OnnxSubGraphOperatorMixin


def ClassFactorySklearn(skl_obj, class_name, doc, conv, shape_calc, alias):
    from .onnx_subgraph_operator_mixin import OnnxSubGraphOperatorMixin

    newclass = type(
        class_name,
        (OnnxSubGraphOperatorMixin, skl_obj),
        {
            "__doc__": doc,
            "operator_name": skl_obj.__name__,
            "_fct_converter": conv,
            "_fct_shape_calc": shape_calc,
            "input_range": [1, 1e9],
            "output_range": [1, 1e9],
            "op_version": None,
            "alias": alias,
            "__module__": __name__,
        },
    )
    return newclass


def dynamic_class_creation_sklearn():
    """
    Automatically generates classes for each of the converter.
    """
    from ..common._registration import _shape_calculator_pool, _converter_pool
    from .._supported_operators import sklearn_operator_name_map

    cls = {}

    for skl_obj, name in sklearn_operator_name_map.items():
        if skl_obj is None:
            continue
        conv = _converter_pool[name]
        shape_calc = _shape_calculator_pool[name]
        skl_name = skl_obj.__name__
        doc = ["OnnxOperatorMixin for **{}**".format(skl_name), ""]
        if conv.__doc__:
            doc.append(textwrap.dedent(conv.__doc__))
        doc = "\n".join(doc)
        prefix = "Sklearn" if "sklearn" in str(skl_obj) else ""
        class_name = "Onnx" + prefix + skl_name
        try:
            cl = ClassFactorySklearn(skl_obj, class_name, doc, conv, shape_calc, name)
        except TypeError:
            continue
        cls[class_name] = cl
    return cls


def _update_module():
    """
    Dynamically updates the module with operators defined
    by *ONNX*.
    """
    res = dynamic_class_creation_sklearn()
    this = sys.modules[__name__]
    for k, v in res.items():
        setattr(this, k, v)


def find_class(skl_cl):
    """
    Finds the corresponding :class:`OnnxSubGraphOperatorMixin`
    class to *skl_cl*.
    """
    name = skl_cl.__name__
    prefix = "OnnxSklearn"
    full_name = prefix + name
    this = sys.modules[__name__]
    if not hasattr(this, full_name):
        available = sorted(filter(lambda n: prefix in n, sys.modules))
        raise RuntimeError(
            "Unable to find a class for '{}' in\n{}".format(
                skl_cl.__name__, "\n".join(available)
            )
        )
    cl = getattr(this, full_name)
    if "automation" in str(cl):
        raise RuntimeError(
            "Dynamic operation issue with class "
            "name '{}' from '{}'.".format(cl, __name__)
        )
    return cl


class OnnxSklearnPipeline(Pipeline, OnnxSubGraphOperatorMixin):
    """
    Combines `Pipeline
    <https://scikit-learn.org/stable/modules/generated/
    sklearn.pipeline.Pipeline.html>`_ and
    :class:`OnnxSubGraphOperatorMixin`.
    """

    def __init__(self, steps, memory=None, verbose=False, op_version=None):
        Pipeline.__init__(self, steps=steps, memory=memory, verbose=verbose)
        OnnxSubGraphOperatorMixin.__init__(self)
        self.op_version = op_version


if ColumnTransformer is not None:

    class OnnxSklearnColumnTransformer(ColumnTransformer, OnnxSubGraphOperatorMixin):
        """
        Combines `ColumnTransformer
        <https://scikit-learn.org/stable/modules/generated/
        sklearn.compose.ColumnTransformer.html>`_ and
        :class:`OnnxSubGraphOperatorMixin`.
        """

        def __init__(self, op_version=None):
            self.op_version = op_version


class OnnxSklearnFeatureUnion(FeatureUnion, OnnxSubGraphOperatorMixin):
    """
    Combines `FeatureUnion
    <https://scikit-learn.org/stable/modules/generated/
    sklearn.pipeline.FeatureUnion.html>`_ and
    :class:`OnnxSubGraphOperatorMixin`.
    """

    def __init__(self, op_version=None):
        self.op_version = op_version


_update_module()
