# SPDX-License-Identifier: Apache-2.0


import numbers
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer


def convert_sklearn_dict_vectorizer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    When a *DictVectorizer* converts numbers into strings,
    scikit-learn adds a separator to disambiguate strings
    and still outputs floats. The method *predict*
    contains the following lines:

    ::

        if isinstance(v, str):
            f = "%s%s%s" % (f, self.separator, v)
            v = 1

    This cannot be implemented in ONNX. The converter
    raises an exception in that case.
    """
    op_type = "DictVectorizer"
    op = operator.raw_operator
    attrs = {"name": scope.get_unique_operator_name(op_type)}
    if all(isinstance(feature_name, str) for feature_name in op.feature_names_):
        # all strings, scikit-learn does the following:
        new_cats = []
        unique_cats = set()
        nbsep = 0
        for i in op.feature_names_:
            if op.separator in i:
                nbsep += 1
            if i in unique_cats:
                raise RuntimeError("Duplicated category '{}'.".format(i))
            unique_cats.add(i)
            new_cats.append(i)
        if nbsep >= len(new_cats):
            raise RuntimeError(
                "All categories contain a separator '{}'. "
                "This case is not supported by the converter. "
                "The mapping must map to numbers not string.".format(op.separator)
            )
        attrs["string_vocabulary"] = new_cats
    elif all(
        isinstance(feature_name, numbers.Integral) for feature_name in op.feature_names_
    ):
        attrs["int64_vocabulary"] = list(int(i) for i in op.feature_names_)
    else:
        raise ValueError("Keys must be all integers or all strings.")

    container.add_node(
        op_type,
        operator.input_full_names,
        operator.output_full_names,
        op_domain="ai.onnx.ml",
        **attrs
    )


register_converter("SklearnDictVectorizer", convert_sklearn_dict_vectorizer)
