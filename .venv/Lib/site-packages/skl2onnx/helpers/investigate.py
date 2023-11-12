# SPDX-License-Identifier: Apache-2.0

import textwrap
import warnings
from types import MethodType
import numpy
from numpy.testing import assert_almost_equal

try:
    from scipy.sparse import csr_matrix
except ImportError:
    from scipy.sparse.csr import csr_matrix
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion

try:
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
except ImportError:
    # not avaiable in 0.19
    ColumnTransformer = None
    TransformedTargetRegressor = None


def enumerate_pipeline_models(pipe, coor=None, vs=None):
    """
    Enumerates all the models within a pipeline.
    """
    if coor is None:
        coor = (0,)
    yield coor, pipe, vs
    if hasattr(pipe, "transformer_and_mapper_list") and len(
        pipe.transformer_and_mapper_list
    ):
        # azureml DataTransformer
        raise NotImplementedError("Unable to handle this specific case.")
    elif hasattr(pipe, "mapper") and pipe.mapper:
        # azureml DataTransformer
        for couple in enumerate_pipeline_models(pipe.mapper, coor + (0,)):
            yield couple
    elif hasattr(pipe, "built_features"):
        # sklearn_pandas.dataframe_mapper.DataFrameMapper
        for i, (columns, transformers, _) in enumerate(pipe.built_features):
            if isinstance(columns, str):
                columns = (columns,)
            if transformers is None:
                yield (coor + (i,)), None, columns
            else:
                for couple in enumerate_pipeline_models(
                    transformers, coor + (i,), columns
                ):
                    yield couple
    elif isinstance(pipe, Pipeline):
        for i, (_, model) in enumerate(pipe.steps):
            for couple in enumerate_pipeline_models(model, coor + (i,)):
                yield couple
    elif ColumnTransformer is not None and isinstance(pipe, ColumnTransformer):
        for i, (_, fitted_transformer, column) in enumerate(pipe.transformers):
            for couple in enumerate_pipeline_models(
                fitted_transformer, coor + (i,), column
            ):
                yield couple
    elif isinstance(pipe, FeatureUnion):
        for i, (_, model) in enumerate(pipe.transformer_list):
            for couple in enumerate_pipeline_models(model, coor + (i,)):
                yield couple
    elif TransformedTargetRegressor is not None and isinstance(
        pipe, TransformedTargetRegressor
    ):
        raise NotImplementedError("Not yet implemented for TransformedTargetRegressor.")
    elif isinstance(pipe, (TransformerMixin, ClassifierMixin, RegressorMixin)):
        pass
    elif isinstance(pipe, BaseEstimator):
        pass
    else:
        raise TypeError(
            "Parameter pipe is not a scikit-learn object: {}\n{}".format(
                type(pipe), pipe
            )
        )


class BaseEstimatorDebugInformation:
    """
    Stores information when the outputs of a pipeline
    is computed. It as added by function
    :func:`_alter_model_for_debugging`.
    """

    def __init__(self, model):
        self.model = model
        self.inputs = {}
        self.outputs = {}
        self.methods = {}
        if hasattr(model, "transform") and callable(model.transform):
            model._debug_transform = model.transform
            self.methods["transform"] = lambda model, X: model._debug_transform(X)
        if hasattr(model, "predict") and callable(model.predict):
            model._debug_predict = model.predict
            self.methods["predict"] = lambda model, X: model._debug_predict(X)
        if hasattr(model, "predict_proba") and callable(model.predict_proba):
            model._debug_predict_proba = model.predict_proba
            self.methods["predict_proba"] = lambda model, X: model._debug_predict_proba(
                X
            )
        if hasattr(model, "decision_function") and callable(
            model.decision_function
        ):  # noqa
            model._debug_decision_function = model.decision_function  # noqa
            self.methods[
                "decision_function"
            ] = lambda model, X: model._debug_decision_function(X)

    def __repr__(self):
        """
        usual
        """
        return self.to_str()

    def to_str(self, nrows=5):
        """
        Tries to produce a readable message.
        """
        rows = [
            "BaseEstimatorDebugInformation({})".format(self.model.__class__.__name__)
        ]
        for k in sorted(self.inputs):
            if k in self.outputs:
                rows.append("  " + k + "(")
                self.display(self.inputs[k], nrows)
                rows.append(textwrap.indent(self.display(self.inputs[k], nrows), "   "))
                rows.append("  ) -> (")
                rows.append(
                    textwrap.indent(self.display(self.outputs[k], nrows), "   ")
                )
                rows.append("  )")
            else:
                raise KeyError("Unable to find output for method '{}'.".format(k))
        return "\n".join(rows)

    def display(self, data, nrows):
        """
        Displays the first
        """
        text = str(data)
        rows = text.split("\n")
        if len(rows) > nrows:
            rows = rows[:nrows]
            rows.append("...")
        if hasattr(data, "shape"):
            rows.insert(0, "shape={}".format(data.shape))
        return "\n".join(rows)


def _alter_model_for_debugging(skl_model, recursive=False):
    """
    Overwrite methods transform, predict or predict_proba
    to collect the last inputs and outputs
    seen in these methods.

    :param skl_model: *scikit-learn* pipeline or model
    :param recursive: alter the current model (False) or git into
        contained models
    """

    def transform(self, X, *args, **kwargs):
        self._debug.inputs["transform"] = X
        y = self._debug.methods["transform"](self, X, *args, **kwargs)
        self._debug.outputs["transform"] = y
        return y

    def predict(self, X, *args, **kwargs):
        self._debug.inputs["predict"] = X
        y = self._debug.methods["predict"](self, X, *args, **kwargs)
        self._debug.outputs["predict"] = y
        return y

    def predict_proba(self, X, *args, **kwargs):
        self._debug.inputs["predict_proba"] = X
        y = self._debug.methods["predict_proba"](self, X, *args, **kwargs)
        self._debug.outputs["predict_proba"] = y
        return y

    def decision_function(self, X, *args, **kwargs):
        self._debug.inputs["decision_function"] = X
        y = self._debug.methods["decision_function"](self, X, *args, **kwargs)
        self._debug.outputs["decision_function"] = y
        return y

    new_methods = {
        "decision_function": decision_function,
        "transform": transform,
        "predict": predict,
        "predict_proba": predict_proba,
    }

    if hasattr(skl_model, "_debug"):
        raise RuntimeError(
            "The same operator cannot be used twice in "
            "the same pipeline or this method was called "
            "a second time."
        )

    if recursive:
        for model_ in enumerate_pipeline_models(skl_model):
            model = model_[1]
            model._debug = BaseEstimatorDebugInformation(model)
            for k in model._debug.methods:
                try:
                    setattr(model, k, MethodType(new_methods[k], model))
                except AttributeError:
                    warnings.warn(
                        "Unable to overwrite method '{}' for class "
                        "{}.".format(k, type(model))
                    )
    else:
        skl_model._debug = BaseEstimatorDebugInformation(skl_model)
        for k in skl_model._debug.methods:
            try:
                setattr(skl_model, k, MethodType(new_methods[k], skl_model))
            except AttributeError:
                warnings.warn(
                    "Unable to overwrite method '{}' for class "
                    "{}.".format(k, type(skl_model))
                )


def collect_intermediate_steps(model, *args, **kwargs):
    """
    Converts a scikit-learn model into ONNX with :func:`convert_sklearn`
    and returns intermediate results for each included operator.

    :param model: model or pipeline to convert
    :param args: arguments for :func:`convert_sklearn`
    :param kwargs: optional arguments for :func:`convert_sklearn`

    The model *model* is modified by the function,
    it should be pickled first to be retrieved unaltered.
    This function is used to check every intermediate model in
    a pipeline.
    """
    if "intermediate" in kwargs:
        if not kwargs["intermediate"]:
            raise ValueError("Parameter intermediate must be true.")
        del kwargs["intermediate"]

    from .. import convert_sklearn
    from ..helpers.onnx_helper import select_model_inputs_outputs
    from ..common import MissingShapeCalculator, MissingConverter

    try:
        model_onnx, topology = convert_sklearn(
            model, *args, intermediate=True, **kwargs
        )
    except (MissingShapeCalculator, MissingConverter):
        # The model cannot be converted.
        raise

    steps = []
    for operator in topology.unordered_operator_iterator():
        if operator.raw_operator is None:
            continue
        _alter_model_for_debugging(operator.raw_operator)
        inputs = [i.full_name for i in operator.inputs]
        outputs = [o.full_name for o in operator.outputs]
        steps.append(
            {
                "model": operator.raw_operator,
                "model_onnx": model_onnx,
                "inputs": inputs,
                "outputs": outputs,
                "onnx_step": select_model_inputs_outputs(model_onnx, outputs=outputs),
            }
        )
    return steps


def compare_objects(o1, o2, decimal=4):
    """
    Compares two objects assuming they are vectors or matrices.
    *o1* and *o2* can be a numpy array, a sparse matrix,
    a dataframe. The function raises an exception if it cannot
    convert both object into the same type or the comparison
    fails.

    :param o1: a dataframe, a series, an array a sparse matrix
    :param o2: a dataframe, a series, an array a sparse matrix
    :param decimal: parameter decimal for assert_almost_equal
    """

    def convert(o):
        if isinstance(o, list) and len(o) == 1:
            if isinstance(o[0], numpy.ndarray):
                if o[0].dtype in (numpy.str_, object, str):
                    o = list(o[0])
                else:
                    o = o[0]
        # Following line avoid importing pandas and taking
        # dependency on pandas.
        if o.__class__.__name__ == "Series":
            c = list(o)
        elif isinstance(o, numpy.ndarray):
            c = o
        elif isinstance(o, csr_matrix):
            c = o.todense()
        elif isinstance(o, list):
            c = o.copy()
        elif isinstance(o, tuple):
            c = list(o)
        else:
            raise TypeError("Unexpected type {}.".format(type(o)))
        return c

    def to_string(c):
        s = str(c)
        if len(s) > 200:
            s = s[:200] + "..."
        return s

    c1 = convert(o1)
    c2 = convert(o2)
    reason = None
    if isinstance(c2, list) and isinstance(c2[0], dict):
        res = numpy.zeros((len(c2), max(len(c) for c in c2)))
        for i, row in enumerate(c2):
            for k, v in row.items():
                res[i, k] = v
        c2 = res
    if isinstance(c1, numpy.ndarray) and isinstance(c2, list):
        c1 = list(c1.ravel())
    if isinstance(c1, list) and isinstance(c2, list):
        try:
            res = c1 == c2
            reason = "list-equal"
        except ValueError:  # lgtm [py/unreachable-statement]
            res = False
            reason = "list"
    elif isinstance(c1, numpy.ndarray) and isinstance(c2, numpy.ndarray):
        try:
            assert_almost_equal(c1, c2, decimal=decimal)
            res = True
        except (AssertionError, TypeError):
            reason = "array"
            cc1 = c1.ravel()
            cc2 = c2.ravel()
            try:
                assert_almost_equal(cc1, cc2, decimal=decimal)
                res = True
            except (AssertionError, TypeError) as e:
                res = False
                reason = "array-ravel" + str(e)
    else:
        raise TypeError("Types {} and {}".format(type(c1), type(c2)))
    if not res:
        msg = "o1 and o2 are different ({})\n---o1---\n{}\n---o2---\n{}"
        raise ValueError(msg.format(reason, to_string(c1), to_string(c2)))
