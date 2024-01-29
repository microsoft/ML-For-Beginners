# SPDX-License-Identifier: Apache-2.0

import warnings
import numpy as np

from sklearn import pipeline
from sklearn.base import ClassifierMixin, ClusterMixin, is_classifier

try:
    from sklearn.base import OutlierMixin
except ImportError:
    # scikit-learn <= 0.19
    class OutlierMixin:
        pass


from sklearn.ensemble import (
    IsolationForest,
    RandomTreesEmbedding,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, NuSVC, SVC

try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # ColumnTransformer was introduced in 0.20.
    ColumnTransformer = None
try:
    from sklearn.preprocessing import Imputer
except ImportError:
    Imputer = None
try:
    from sklearn.impute import SimpleImputer
except ImportError:
    # changed in 0.20
    SimpleImputer = None

from ._supported_operators import _get_sklearn_operator_name, cluster_list, outlier_list
from ._supported_operators import sklearn_classifier_list, sklearn_operator_name_map
from .common._container import SklearnModelContainerNode
from .common._registration import _converter_pool, _shape_calculator_pool
from .common._topology import Topology, Variable
from .common.data_types import (
    DictionaryType,
    Int64TensorType,
    SequenceType,
    StringTensorType,
    TensorType,
    FloatTensorType,
    guess_tensor_type,
)
from .common.utils import get_column_indices
from .common.utils_checking import check_signature
from .common.utils_classifier import get_label_classes
from .common.utils_sklearn import _process_options


do_not_merge_columns = tuple(
    filter(
        lambda op: op is not None, [OrdinalEncoder, OneHotEncoder, ColumnTransformer]
    )
)


def _fetch_input_slice(scope, inputs, column_indices):
    if not isinstance(inputs, list):
        raise TypeError("Parameter inputs must be a list.")
    if len(inputs) == 0:
        raise RuntimeError(
            "Operator ArrayFeatureExtractor requires at " "least one inputs."
        )
    if len(inputs) != 1:
        raise RuntimeError(
            "Operator ArrayFeatureExtractor does not support " "multiple input tensors."
        )
    if (
        isinstance(inputs[0].type, TensorType)
        and len(inputs[0].type.shape) == 2
        and inputs[0].type.shape[1] == len(column_indices)
    ):
        # No need to extract.
        return inputs
    array_feature_extractor_operator = scope.declare_local_operator(
        "SklearnArrayFeatureExtractor"
    )
    array_feature_extractor_operator.inputs = inputs
    array_feature_extractor_operator.column_indices = column_indices
    output_variable_name = scope.declare_local_variable(
        "extracted_feature_columns", inputs[0].type
    )
    array_feature_extractor_operator.outputs.append(output_variable_name)
    return array_feature_extractor_operator.outputs


def _parse_sklearn_simple_model(scope, model, inputs, custom_parsers=None, alias=None):
    """
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., *OneHotEncoder*
        or *LogisticRegression*)
    :param inputs: A list of variables
    :param custom_parsers: dictionary of custom parsers
    :param alias: use this alias instead of the one based on the class name
    :return: A list of output variables which will be passed to next
        stage
    """
    # alias can be None
    if isinstance(model, str):
        raise RuntimeError(
            "Parameter model must be an object not a " "string '{0}'.".format(model)
        )
    if any(not isinstance(i, Variable) for i in inputs):
        raise TypeError(
            "One input is not a Variable for model %r - %r." "" % (model, inputs)
        )
    if alias is None:
        alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs

    if hasattr(model, "onnx_parser"):
        parser_names = model.onnx_parser()
        if parser_names is not None:
            try:
                names = parser_names(scope=scope, inputs=inputs)
            except TypeError as e:
                warnings.warn(
                    "Calling parser %r for model type %r failed due to %r. "
                    "This warnings will become an exception in version 1.11. "
                    "The parser signature should parser(scope=None, "
                    "inputs=None)." % (parser_names, e, type(model)),
                    DeprecationWarning,
                )
                names = parser_names()
            if names is not None:
                for name in names:
                    if isinstance(name, Variable):
                        this_operator.outputs.append(name)
                    elif isinstance(name, str):
                        var = scope.declare_local_variable(
                            name, guess_tensor_type(inputs[0].type)
                        )
                        this_operator.outputs.append(var)
                    elif isinstance(name, tuple) and len(name) == 2:
                        var = scope.declare_local_variable(
                            name[0], guess_tensor_type(name[1])
                        )
                        this_operator.outputs.append(var)
                    else:
                        raise RuntimeError(
                            "Unexpected output type %r (value=%r) for "
                            "operator %r." % (type(name), name, type(model))
                        )
                return this_operator.outputs

    if (
        type(model) in sklearn_classifier_list
        or isinstance(model, ClassifierMixin)
        or (isinstance(model, GridSearchCV) and is_classifier(model))
    ):
        # For classifiers, we may have two outputs, one for label and
        # the other one for probabilities of all classes. Notice that
        # their types here are not necessarily correct and they will
        # be fixed in shape inference phase.
        label_variable = scope.declare_local_variable("label", Int64TensorType())
        if type(model) in [RandomForestClassifier]:
            prob_dtype = FloatTensorType()
        else:
            prob_dtype = guess_tensor_type(inputs[0].type)
        probability_tensor_variable = scope.declare_local_variable(
            "probabilities", prob_dtype
        )
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(probability_tensor_variable)

    elif type(model) in cluster_list or isinstance(model, ClusterMixin):
        # For clustering, we may have two outputs, one for label and
        # the other one for scores of all classes. Notice that their
        # types here are not necessarily correct and they will be fixed
        # in shape inference phase
        label_variable = scope.declare_local_variable("label", Int64TensorType())
        score_tensor_variable = scope.declare_local_variable(
            "scores", guess_tensor_type(inputs[0].type)
        )
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(score_tensor_variable)

    elif type(model) in {IsolationForest, LocalOutlierFactor}:
        label_variable = scope.declare_local_variable("label", Int64TensorType())
        score_tensor_variable = scope.declare_local_variable(
            "scores", guess_tensor_type(inputs[0].type)
        )
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(score_tensor_variable)
        options = scope.get_options(model, dict(score_samples=False))
        if options["score_samples"]:
            scores_var = scope.declare_local_variable(
                "score_samples", guess_tensor_type(inputs[0].type)
            )
            this_operator.outputs.append(scores_var)

    elif type(model) in outlier_list or isinstance(model, OutlierMixin):
        # For outliers, we may have two outputs, one for label and
        # the other one for scores.
        label_variable = scope.declare_local_variable("label", Int64TensorType())
        score_tensor_variable = scope.declare_local_variable(
            "scores", guess_tensor_type(inputs[0].type)
        )
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(score_tensor_variable)

    elif isinstance(model, NearestNeighbors):
        # For Nearest Neighbours, we have two outputs, one for nearest
        # neighbours' indices and the other one for distances
        index_variable = scope.declare_local_variable("index", Int64TensorType())
        distance_variable = scope.declare_local_variable(
            "distance", guess_tensor_type(inputs[0].type)
        )
        this_operator.outputs.append(index_variable)
        this_operator.outputs.append(distance_variable)

    elif type(model) in {GaussianMixture, BayesianGaussianMixture}:
        label_variable = scope.declare_local_variable("label", Int64TensorType())
        prob_variable = scope.declare_local_variable(
            "probabilities", guess_tensor_type(inputs[0].type)
        )
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(prob_variable)
        options = scope.get_options(model, dict(score_samples=False))
        if options["score_samples"]:
            scores_var = scope.declare_local_variable(
                "score_samples", guess_tensor_type(inputs[0].type)
            )
            this_operator.outputs.append(scores_var)
    elif type(model) in {SimpleImputer, Imputer}:
        if isinstance(inputs[0].type, (Int64TensorType, StringTensorType)):
            otype = inputs[0].type.__class__()
        else:
            otype = guess_tensor_type(inputs[0].type)
        variable = scope.declare_local_variable("variable", otype)
        this_operator.outputs.append(variable)
    else:
        if hasattr(model, "get_feature_names_out"):
            try:
                out_names = model.get_feature_names_out()
            except (AttributeError, ValueError):
                # Catch a bug in scikit-learn.
                out_names = None
            this_operator.feature_names_out_ = out_names
        input_type = guess_tensor_type(inputs[0].type)
        variable = scope.declare_local_variable("variable", input_type)
        this_operator.outputs.append(variable)

    options = scope.get_options(model, dict(decision_path=False), fail=False)
    if options is not None and options["decision_path"]:
        dec_path = scope.declare_local_variable("decision_path", StringTensorType())
        this_operator.outputs.append(dec_path)

    options = scope.get_options(model, dict(decision_leaf=False), fail=False)
    if options is not None and options["decision_leaf"]:
        dec_path = scope.declare_local_variable("decision_leaf", Int64TensorType())
        this_operator.outputs.append(dec_path)

    return this_operator.outputs


def _parse_sklearn_pipeline(scope, model, inputs, custom_parsers=None):
    """
    The basic ideas of scikit-learn parsing:
        1. Sequentially go though all stages defined in the considered
           scikit-learn pipeline
        2. The output variables of one stage will be fed into its next
           stage as the inputs.

    :param scope: Scope object defined in _topology.py
    :param model: scikit-learn pipeline object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by the input pipeline
    """
    for step in model.steps:
        inputs = _parse_sklearn(scope, step[1], inputs, custom_parsers=custom_parsers)
    return inputs


def _parse_sklearn_feature_union(scope, model, inputs, custom_parsers=None):
    """
    :param scope: Scope object
    :param model: A scikit-learn FeatureUnion object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by feature union
    """
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    # Encode each transform as our IR object
    for name, transform in model.transformer_list:
        transformed_result_names.append(
            _parse_sklearn(scope, transform, inputs, custom_parsers=custom_parsers)[0]
        )
        if model.transformer_weights is not None and name in model.transformer_weights:
            transform_result = [transformed_result_names.pop()]
            # Create a Multiply ONNX node
            multiply_operator = scope.declare_local_operator("SklearnMultiply")
            multiply_operator.inputs = transform_result
            multiply_operator.operand = model.transformer_weights[name]
            multiply_output = scope.declare_local_variable(
                "multiply_output", guess_tensor_type(inputs[0].type)
            )
            multiply_operator.outputs.append(multiply_output)
            transformed_result_names.append(multiply_operator.outputs[0])

    # Create a Concat ONNX node
    concat_operator = scope.declare_local_operator("SklearnConcat")
    concat_operator.inputs = transformed_result_names

    # Declare output name of scikit-learn FeatureUnion
    union_name = scope.declare_local_variable(
        "union", guess_tensor_type(inputs[0].type)
    )
    concat_operator.outputs.append(union_name)

    return concat_operator.outputs


def _parse_sklearn_column_transformer(scope, model, inputs, custom_parsers=None):
    """
    :param scope: Scope object
    :param model: A *scikit-learn* *ColumnTransformer* object
    :param inputs: A list of Variable objects
    :return: A list of output variables produced by column transformer
    """
    # Output variable name of each transform. It's a list of string.
    transformed_result_names = []
    # Encode each transform as our IR object
    for name, op, column_indices in model.transformers_:
        if op == "drop":
            continue
        if isinstance(column_indices, slice):
            column_indices = list(
                range(
                    column_indices.start if column_indices.start is not None else 0,
                    column_indices.stop,
                    column_indices.step if column_indices.step is not None else 1,
                )
            )
        elif isinstance(column_indices, (int, str)):
            column_indices = [column_indices]
        names = get_column_indices(column_indices, inputs, multiple=True)
        transform_inputs = []
        for onnx_var, onnx_is in names.items():
            tr_inputs = _fetch_input_slice(scope, [inputs[onnx_var]], onnx_is)
            transform_inputs.extend(tr_inputs)

        merged_cols = False
        if len(transform_inputs) > 1:
            if isinstance(op, Pipeline):
                if not isinstance(op.steps[0][1], do_not_merge_columns):
                    merged_cols = True
            elif not isinstance(op, do_not_merge_columns):
                merged_cols = True

        if merged_cols:
            # Many ONNX operators expect one input vector,
            # the default behaviour is to merge columns.
            ty = transform_inputs[0].type.__class__([None, None])

            conc_op = scope.declare_local_operator("SklearnConcat")
            conc_op.inputs = transform_inputs
            conc_names = scope.declare_local_variable("merged_columns", ty)
            conc_op.outputs.append(conc_names)
            transform_inputs = [conc_names]

        model_obj = model.named_transformers_[name]
        if isinstance(model_obj, str):
            if model_obj == "passthrough":
                var_out = transform_inputs[0]
            elif model_obj == "drop":
                var_out = None
            else:
                raise RuntimeError(
                    "Unknown operator alias "
                    "'{0}'. These are specified in "
                    "_supported_operators.py."
                    "".format(model_obj)
                )
        else:
            var_out = _parse_sklearn(
                scope, model_obj, transform_inputs, custom_parsers=custom_parsers
            )[0]
            if (
                model.transformer_weights is not None
                and name in model.transformer_weights
            ):
                # Create a Multiply ONNX node
                multiply_operator = scope.declare_local_operator("SklearnMultiply")
                multiply_operator.inputs.append(var_out)
                multiply_operator.operand = model.transformer_weights[name]
                var_out = scope.declare_local_variable(
                    "multiply_output", guess_tensor_type(inputs[0].type)
                )
                multiply_operator.outputs.append(var_out)
        if var_out:
            transformed_result_names.append(var_out)

    # Create a Concat ONNX node
    if len(transformed_result_names) > 1:
        ty = transformed_result_names[0].type.__class__([None, None])
        concat_operator = scope.declare_local_operator("SklearnConcat")
        concat_operator.inputs = transformed_result_names

        # Declare output name of scikit-learn ColumnTransformer
        transformed_column_name = scope.declare_local_variable("transformed_column", ty)
        concat_operator.outputs.append(transformed_column_name)
        return concat_operator.outputs
    return transformed_result_names


def _parse_sklearn_grid_search_cv(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model)
    if options:
        scope.add_options(id(model.best_estimator_), options)
    res = parse_sklearn(
        scope, model.best_estimator_, inputs, custom_parsers=custom_parsers
    )
    scope.replace_raw_operator(model.best_estimator_, model, "SklearnGridSearchCV")
    return res


def _parse_sklearn_random_trees_embedding(scope, model, inputs, custom_parsers=None):
    res = parse_sklearn(
        scope, model.base_estimator_, inputs, custom_parsers=custom_parsers
    )
    if len(res) != 1:
        raise RuntimeError("A regressor only produces one output not %r." % res)
    scope.replace_raw_operator(
        model.base_estimator_, model, "SklearnRandomTreesEmbedding"
    )
    return res


def _apply_zipmap(zipmap_options, scope, model, input_type, probability_tensor):
    if zipmap_options == "columns":
        zipmap_operator = scope.declare_local_operator("SklearnZipMapColumns")
        classes = get_label_classes(scope, model)
        classes_names = get_label_classes(scope, model, node_names=True)
    else:
        zipmap_operator = scope.declare_local_operator("SklearnZipMap")
        classes = get_label_classes(scope, model)

    zipmap_operator.inputs = probability_tensor
    label_type = Int64TensorType([None])

    if (
        hasattr(model, "classes_")
        and isinstance(model.classes_, list)
        and isinstance(model.classes_[0], np.ndarray)
    ):
        # multi-label problem
        pass
    elif np.issubdtype(classes.dtype, np.floating):
        classes = np.array(list(map(lambda x: int(x), classes)))
        if set(map(lambda x: float(x), classes)) != set(model.classes_):
            raise RuntimeError(
                "skl2onnx implicitly converts float class "
                "labels into integers but at least one label "
                "is not an integer. Class labels should "
                "be integers or strings."
            )
        zipmap_operator.classlabels_int64s = classes
    elif np.issubdtype(classes.dtype, np.signedinteger):
        zipmap_operator.classlabels_int64s = [int(i) for i in classes]
    elif np.issubdtype(classes.dtype, np.unsignedinteger) or classes.dtype == np.bool_:
        zipmap_operator.classlabels_int64s = [int(i) for i in classes]
    else:
        classes = np.array([s.encode("utf-8") for s in classes])
        zipmap_operator.classlabels_strings = classes
        label_type = StringTensorType([None])

    zip_label = scope.declare_local_variable("output_label", label_type)
    if len(probability_tensor) == 2:
        zipmap_operator.outputs.append(zip_label)

    if zipmap_options == "columns":
        prob_type = probability_tensor[-1].type
        for cl in classes_names:
            output_cl = scope.declare_local_variable(cl, prob_type.__class__())
            zipmap_operator.outputs.append(output_cl)
    else:
        zip_probability = scope.declare_local_variable(
            "output_probability",
            SequenceType(DictionaryType(label_type, guess_tensor_type(input_type))),
        )
        zipmap_operator.outputs.append(zip_probability)

    zipmap_operator.init_status(is_evaluated=True)
    return zipmap_operator.outputs


def _parse_sklearn_classifier(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model, dict(zipmap=True))
    no_zipmap = (isinstance(options["zipmap"], bool) and not options["zipmap"]) or (
        model.__class__ in [NuSVC, SVC] and not model.probability
    )
    probability_tensor = _parse_sklearn_simple_model(
        scope, model, inputs, custom_parsers=custom_parsers
    )

    if no_zipmap:
        if options.get("output_class_labels", False):
            if not hasattr(model, "classes_"):
                raise RuntimeError(
                    "Model type %r has no attribute 'classes_'. "
                    "Option 'output_class_labels' is invalid or a new parser "
                    "must be used." % model.__class__.__name__
                )

            clout = scope.declare_local_operator("SklearnClassLabels")
            clout.classes = get_label_classes(scope, model)
            if model.classes_.dtype in (np.int32, np.int64, np.bool_):
                ctype = Int64TensorType
            else:
                ctype = StringTensorType
            label_type = ctype(clout.classes.shape)
            class_labels = scope.declare_local_variable("class_labels", label_type)
            clout.outputs.append(class_labels)
            outputs = list(probability_tensor)
            outputs.append(class_labels)
            return outputs
        return probability_tensor

    if options.get("output_class_labels", False):
        raise RuntimeError(
            "Option 'output_class_labels' is not compatible with option " "'zipmap'."
        )

    return _apply_zipmap(
        options["zipmap"], scope, model, inputs[0].type, probability_tensor
    )


def _parse_sklearn_multi_output_classifier(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model, dict(zipmap=True))
    if options["zipmap"]:
        warnings.warn(
            "Option zipmap is ignored for model %r. "
            "Set option zipmap to False to "
            "remove this message." % type(model),
            UserWarning,
        )
    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs

    if hasattr(model, "classes_"):
        classes = model.classes_
    else:
        classes = [get_label_classes(scope, m) for m in model.estimators_]
    if len(set(cl.dtype for cl in classes)) != 1:
        raise RuntimeError(
            "Class labels may have only one type %r."
            "" % set(cl.dtype for cl in classes)
        )
    if classes[0].dtype in (np.int32, np.int64, np.bool_):
        ctype = Int64TensorType
    else:
        ctype = StringTensorType

    label = scope.declare_local_variable("label", ctype())
    proba = scope.declare_local_variable(
        "probabilities", SequenceType(guess_tensor_type(inputs[0].type))
    )
    this_operator.outputs.append(label)
    this_operator.outputs.append(proba)

    options = scope.get_options(model)
    if options.get("output_class_labels", False):
        clout = scope.declare_local_operator("SklearnClassLabels")
        clout.is_multi_output = True
        clout.classes = classes
        class_labels = scope.declare_local_variable(
            "class_labels", SequenceType(ctype())
        )
        clout.outputs.append(class_labels)
        return list(this_operator.outputs) + [class_labels]

    return this_operator.outputs


def _parse_sklearn_gaussian_process(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model, dict(return_cov=False, return_std=False))
    if options["return_std"] and options["return_cov"]:
        raise RuntimeError(
            "Not returning standard deviation of predictions when "
            "returning full covariance."
        )

    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    mean_tensor = scope.declare_local_variable(
        "GPmean", guess_tensor_type(inputs[0].type)
    )
    this_operator.inputs = inputs
    this_operator.outputs.append(mean_tensor)

    if options["return_std"] or options["return_cov"]:
        # covariance or standard deviation
        covstd_tensor = scope.declare_local_variable(
            "GPcovstd", guess_tensor_type(inputs[0].type)
        )
        this_operator.outputs.append(covstd_tensor)
    return this_operator.outputs


def _parse_sklearn_bayesian_ridge(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model, dict(return_std=False))
    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    mean_tensor = scope.declare_local_variable(
        "variable", guess_tensor_type(inputs[0].type)
    )
    this_operator.inputs = inputs
    this_operator.outputs.append(mean_tensor)

    if options["return_std"]:
        # covariance or standard deviation
        covstd_tensor = scope.declare_local_variable(
            "std", guess_tensor_type(inputs[0].type)
        )
        this_operator.outputs.append(covstd_tensor)
    return this_operator.outputs


def _parse_sklearn(scope, model, inputs, custom_parsers=None, alias=None):
    """
    This is a delegate function. It does nothing but invokes the
    correct parsing function according to the input model's type.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder
        and LogisticRegression)
    :param inputs: A list of variables
    :param custom_parsers: parsers determines which outputs is expected
        for which particular task, default parsers are defined for
        classifiers, regressors, pipeline but they can be rewritten,
        *custom_parsers* is a dictionary ``{ type: fct_parser(scope,
        model, inputs, custom_parsers=None) }``
    :param alias: alias of the model (None if based on the model class)
    :return: The output variables produced by the input model
    """
    for i, inp in enumerate(inputs):
        if not isinstance(inp, Variable):
            raise TypeError(
                "Unexpected input type %r for input %r: %r." % (type(inp), i, inp)
            )

    if alias is not None:
        outputs = _parse_sklearn_simple_model(
            scope, model, inputs, custom_parsers=custom_parsers, alias=alias
        )
        return outputs

    tmodel = type(model)
    if custom_parsers is not None and tmodel in custom_parsers:
        outputs = custom_parsers[tmodel](
            scope, model, inputs, custom_parsers=custom_parsers
        )
    elif tmodel in sklearn_parsers_map:
        outputs = sklearn_parsers_map[tmodel](
            scope, model, inputs, custom_parsers=custom_parsers
        )
    elif isinstance(model, pipeline.Pipeline):
        parser = sklearn_parsers_map[pipeline.Pipeline]
        outputs = parser(scope, model, inputs, custom_parsers=custom_parsers)
    else:
        outputs = _parse_sklearn_simple_model(
            scope, model, inputs, custom_parsers=custom_parsers
        )
    return outputs


def parse_sklearn(scope, model, inputs, custom_parsers=None, final_types=None):
    """
    This is a delegate function. It does nothing but invokes the
    correct parsing function according to the input model's type.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder
        and LogisticRegression)
    :param inputs: A list of variables
    :param custom_parsers: parsers determines which outputs is expected
        for which particular task, default parsers are defined for
        classifiers, regressors, pipeline but they can be rewritten,
        *custom_parsers* is a dictionary ``{ type: fct_parser(scope,
        model, inputs, custom_parsers=None) }``
    :param final_types: a python list. Works the same way as initial_types
        but not mandatory, it is used to overwrites the type
        (if type is not None) and the name of every output.
    :return: The output variables produced by the input model
    """
    if final_types is not None:
        outputs = []
        for name, ty in final_types:
            var = scope.declare_local_output(name, ty, missing_type=True)
            if var.onnx_name != name:
                raise RuntimeError(
                    "Unable to add duplicated output '{}', '{}'. "
                    "Output and input must have different names."
                    "".format(var.onnx_name, name)
                )
            outputs.append(var)

        hidden_outputs = _parse_sklearn(
            scope, model, inputs, custom_parsers=custom_parsers
        )

        if len(hidden_outputs) != len(outputs):
            raise RuntimeError(
                "Number of declared outputs is unexpected, declared '{}' "
                "found '{}'.".format(
                    ", ".join(_.onnx_name for _ in outputs),
                    ", ".join(_.onnx_name for _ in hidden_outputs),
                )
            )
        for h, o in zip(hidden_outputs, outputs):
            if o.type is None:
                iop = scope.declare_local_operator("SklearnIdentity")
            else:
                iop = scope.declare_local_operator("SklearnCast")
            iop.inputs = [h]
            iop.outputs = [o]
            h.init_status(is_leaf=False)
            o.init_status(is_leaf=True)
            if o.type is None and h.type is not None:
                o.type = h.type
        return outputs

    res = _parse_sklearn(scope, model, inputs, custom_parsers=custom_parsers)
    for r in res:
        r.init_status(is_leaf=True)
    return res


def parse_sklearn_model(
    model,
    initial_types=None,
    target_opset=None,
    custom_conversion_functions=None,
    custom_shape_calculators=None,
    custom_parsers=None,
    options=None,
    white_op=None,
    black_op=None,
    final_types=None,
    naming=None,
):
    """
    Puts *scikit-learn* object into an abstract container so that
    our framework can work seamlessly on models created
    with different machine learning tools.

    :param model: A scikit-learn model
    :param initial_types: a python list. Each element is a tuple of a
        variable name and a type defined in data_types.py
    :param target_opset: number, for example, 7 for ONNX 1.2,
        and 8 for ONNX 1.3.
    :param custom_conversion_functions: a dictionary for specifying
        the user customized conversion function if not registered
    :param custom_shape_calculators: a dictionary for specifying the
        user customized shape calculator if not registered
    :param custom_parsers: parsers determines which outputs is expected
        for which particular task, default parsers are defined for
        classifiers, regressors, pipeline but they can be rewritten,
        *custom_parsers* is a dictionary
        ``{ type: fct_parser(scope, model, inputs, custom_parsers=None) }``
    :param options: specific options given to converters
        (see :ref:`l-conv-options`)
    :param white_op: white list of ONNX nodes allowed
        while converting a pipeline, if empty, all are allowed
    :param black_op: black list of ONNX nodes allowed
        while converting a pipeline, if empty, none are blacklisted
    :param final_types: a python list. Works the same way as initial_types
        but not mandatory, it is used to overwrites the type
        (if type is not None) and the name of every output.
    :param naming: the user may want to change the way intermediate
        are named, this parameter can be a string (a prefix) or a
        function, which signature is the following:
        `get_name(name, existing_names)`, the library will then
        check this name is unique and modify it if not
    :return: :class:`Topology <skl2onnx.common._topology.Topology>`

    .. versionchanged:: 1.10.0
        Parameter *naming* was added.
    """
    options = _process_options(model, options)

    raw_model_container = SklearnModelContainerNode(
        model, white_op=white_op, black_op=black_op
    )

    # Declare a computational graph. It will become a representation of
    # the input scikit-learn model after parsing.
    topology = Topology(
        raw_model_container,
        initial_types=initial_types,
        target_opset=target_opset,
        custom_conversion_functions=custom_conversion_functions,
        custom_shape_calculators=custom_shape_calculators,
        registered_models=dict(
            conv=_converter_pool,
            shape=_shape_calculator_pool,
            aliases=sklearn_operator_name_map,
        ),
    )

    # Declare an object to provide variables' and operators' naming mechanism.
    scope = topology.declare_scope("__root__", options=options, naming=naming)
    inputs = scope.input_variables

    # The object raw_model_container is a part of the topology
    # we're going to return. We use it to store the inputs of
    # the scikit-learn's computational graph.
    for variable in inputs:
        variable.init_status(is_root=True)
        raw_model_container.add_input(variable)

    # Parse the input scikit-learn model as a Topology object.
    outputs = parse_sklearn(
        scope, model, inputs, custom_parsers=custom_parsers, final_types=final_types
    )

    # The object raw_model_container is a part of the topology we're
    # going to return. We use it to store the outputs of the
    # scikit-learn's computational graph.
    if final_types is not None and len(final_types) != len(outputs):
        raise RuntimeError(
            "Unexpected number of outputs, expected %d, got %d "
            "after parsing." % (len(final_types), len(outputs))
        )
    return topology


def build_sklearn_parsers_map():
    map_parser = {
        pipeline.Pipeline: _parse_sklearn_pipeline,
        pipeline.FeatureUnion: _parse_sklearn_feature_union,
        BayesianRidge: _parse_sklearn_bayesian_ridge,
        GaussianProcessRegressor: _parse_sklearn_gaussian_process,
        GridSearchCV: _parse_sklearn_grid_search_cv,
        MultiOutputClassifier: _parse_sklearn_multi_output_classifier,
        RandomTreesEmbedding: _parse_sklearn_random_trees_embedding,
    }
    if ColumnTransformer is not None:
        map_parser[ColumnTransformer] = _parse_sklearn_column_transformer

    for tmodel in sklearn_classifier_list:
        if tmodel not in [LinearSVC]:
            map_parser[tmodel] = _parse_sklearn_classifier
    return map_parser


def update_registered_parser(model, parser_fct):
    """
    Registers or updates a parser for a new model.
    A parser returns the expected output of a model.

    :param model: model class
    :param parser_fct: parser, signature is the same as
        :func:`parse_sklearn <skl2onnx._parse.parse_sklearn>`
    """
    check_signature(parser_fct, _parse_sklearn_classifier)
    sklearn_parsers_map[model] = parser_fct


# registered parsers
sklearn_parsers_map = build_sklearn_parsers_map()
