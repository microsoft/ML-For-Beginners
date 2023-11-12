# SPDX-License-Identifier: Apache-2.0

import warnings
from uuid import uuid4
from .proto import get_latest_tested_opset_version
from .common._topology import convert_topology
from .common.utils_sklearn import _process_options
from ._parse import parse_sklearn_model

# Invoke the registration of all our converters and shape calculators.
from . import shape_calculators  # noqa
from . import operator_converters  # noqa


def convert_sklearn(
    model,
    name=None,
    initial_types=None,
    doc_string="",
    target_opset=None,
    custom_conversion_functions=None,
    custom_shape_calculators=None,
    custom_parsers=None,
    options=None,
    intermediate=False,
    white_op=None,
    black_op=None,
    final_types=None,
    dtype=None,
    naming=None,
    model_optim=True,
    verbose=0,
):
    """
    This function produces an equivalent
    ONNX model of the given scikit-learn model.
    The supported converters is returned by function
    :func:`supported_converters <skl2onnx.supported_converters>`.

    For pipeline conversion, user needs to make sure each component
    is one of our supported items.
    This function converts the specified *scikit-learn* model
    into its *ONNX* counterpart.
    Note that for all conversions, initial types are required.
    *ONNX* model name can also be specified.

    :param model: A scikit-learn model
    :param initial_types: a python list.
        Each element is a tuple of a variable name
        and a type defined in `data_types.py`
    :param name: The name of the graph (type: GraphProto)
        in the produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :param target_opset: number, for example, 7 for
        ONNX 1.2, and 8 for ONNX 1.3,
        if value is not specified, the function will
        choose the latest tested opset
        (see :py:func:`skl2onnx.get_latest_tested_opset_version`)
    :param custom_conversion_functions: a dictionary for
        specifying the user customized conversion function,
        it takes precedence over registered converters
    :param custom_shape_calculators: a dictionary for
        specifying the user customized shape calculator
        it takes precedence over registered shape calculators.
    :param custom_parsers: parsers determines which outputs
        is expected for which particular task,
        default parsers are defined for classifiers,
        regressors, pipeline but they can be rewritten,
        *custom_parsers* is a dictionary
        ``{ type: fct_parser(scope, model, inputs, custom_parsers=None) }``
    :param options: specific options given to converters
        (see :ref:`l-conv-options`)
    :param intermediate: if True, the function returns the
        converted model and the instance of :class:`Topology` used,
        it returns the converted model otherwise
    :param white_op: white list of ONNX nodes allowed
        while converting a pipeline,
        if empty, all are allowed
    :param black_op: black list of ONNX nodes
        allowed while converting a pipeline,
        if empty, none are blacklisted
    :param final_types: a python list. Works the same way as initial_types
        but not mandatory, it is used to overwrites the type
        (if type is not None) and the name of every output.
    :param dtype: removed in version 1.7.5, dtype is
        now inferred from input types,
        converters may add operators Cast to switch
        to double when it is necessary
    :param naming: the user may want to change the way intermediate
        are named, this parameter can be a string (a prefix) or a
        function, which signature is the following:
        `get_name(name, existing_names)`, the library will then
        check this name is unique and modify it if not
    :param model_optim: enable or disable model optimisation
        after the model was converted into onnx, it reduces the number
        of identity nodes
    :param verbose: display progress while converting a model
    :return: An ONNX model (type: ModelProto) which is
        equivalent to the input scikit-learn model

    Example of *initial_types*:
    Assume that the specified *scikit-learn* model takes
    a heterogeneous list as its input.
    If the first 5 elements are floats and the last 10 elements are integers,
    we need to specify initial types as below. The [None] in
    [None, 5] indicates the batch size here is unknown.

    ::

        from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
        initial_type = [('float_input', FloatTensorType([None, 5])),
                        ('int64_input', Int64TensorType([None, 10]))]

    .. note::

        If a pipeline includes an instance of
        `ColumnTransformer <https://scikit-learn.org/stable/modules/
        generated/sklearn.compose.ColumnTransformer.html>`_,
        *scikit-learn* allow the user to specify columns by names.
        This option is not supported
        by *sklearn-onnx* as features names could be different
        in input data and the ONNX graph
        (defined by parameter *initial_types*), only integers are supported.

    .. _l-conv-options:

    Converters options
    ++++++++++++++++++

    Some ONNX operators exposes parameters *sklearn-onnx* cannot
    guess from the raw model. Some default values are usually suggested
    but the users may have to manually overwrite them. This need
    is not obvious to do when a model is included in a pipeline.
    That's why these options can be given to function *convert_sklearn*
    as a dictionary ``{model_type: parameters in a dictionary}`` or
    ``{model_id: parameters in a dictionary}``.
    Option *sep* is used to specify the delimiters between two words
    when the ONNX graph needs to tokenize a string.
    The default value is short and may not include all
    the necessary values. It can be overwritten as:

    ::

        extra = {TfidfVectorizer: {"separators": [' ', '[.]', '\\\\?',
                    ',', ';', ':', '\\\\!', '\\\\(', '\\\\)']}}
        model_onnx = convert_sklearn(
            model, "tfidf",
            initial_types=[("input", StringTensorType([None, 1]))],
            options=extra)

    But if a pipeline contains two model of the same class,
    it is possible to distinguish between the two with function *id*:

    ::

        extra = {id(model): {"separators": [' ', '.', '\\\\?', ',', ';',
                    ':', '\\\\!', '\\\\(', '\\\\)']}}
        model_onnx = convert_sklearn(
            pipeline, "pipeline-with-2-tfidf",
            initial_types=[("input", StringTensorType([None, 1]))],
            options=extra)

    It is used in example :ref:`l-example-tfidfvectorizer`.

    .. versionchanged:: 1.10.0
        Parameter *naming* was added.
    """
    if initial_types is None:
        if hasattr(model, "infer_initial_types"):
            initial_types = model.infer_initial_types()
        else:
            raise ValueError(
                "Initial types are required. See usage of "
                "convert(...) in skl2onnx.convert for details"
            )

    if name is None:
        name = str(uuid4().hex)
    if dtype is not None:
        warnings.warn(
            "Parameter dtype is no longer supported. " "It will be removed in 1.9.0.",
            DeprecationWarning,
        )

    target_opset = target_opset if target_opset else get_latest_tested_opset_version()
    # Parse scikit-learn model as our internal data structure
    # (i.e., Topology)
    if verbose >= 1:
        print("[convert_sklearn] parse_sklearn_model")
    topology = parse_sklearn_model(
        model,
        initial_types,
        target_opset,
        custom_conversion_functions,
        custom_shape_calculators,
        custom_parsers,
        options=options,
        white_op=white_op,
        black_op=black_op,
        final_types=final_types,
        naming=naming,
    )

    # Convert our Topology object into ONNX. The outcome is an ONNX model.
    options = _process_options(model, options)
    if verbose >= 1:
        print("[convert_sklearn] convert_topology")
    onnx_model = convert_topology(
        topology,
        name,
        doc_string,
        target_opset,
        options=options,
        remove_identity=model_optim and not intermediate,
        verbose=verbose,
    )
    if verbose >= 1:
        print("[convert_sklearn] end")
        if verbose >= 2:
            scope = topology.scopes[0]
            print("---INPUTS---")
            for inp in scope.input_variables:
                print("  %r" % inp)
            print("---OUTPUTS---")
            for inp in scope.output_variables:
                print("  %r" % inp)
            print("---VARIABLES---")
            for k, v in sorted(scope.variables.items()):
                print("  %r: is.fed=%r is_leaf=%r - %r" % (k, v.is_fed, v.is_leaf, v))
            print("---OPERATORS---")
            for k, v in sorted(scope.operators.items()):
                print("  %r: is.evaluated=%r - %r" % (k, v.is_evaluated, v))

    return (onnx_model, topology) if intermediate else onnx_model


def to_onnx(
    model,
    X=None,
    name=None,
    initial_types=None,
    target_opset=None,
    options=None,
    white_op=None,
    black_op=None,
    final_types=None,
    dtype=None,
    naming=None,
    model_optim=True,
    verbose=0,
):
    """
    Calls :func:`convert_sklearn` with simplified parameters.

    :param model: model to convert
    :param X: training set, can be None, it is used to infered the
        input types (*initial_types*)
    :param initial_types: if X is None, then *initial_types* must be
        defined
    :param target_opset: conversion with a specific target opset
    :param options: specific options given to converters
        (see :ref:`l-conv-options`)
    :param name: name of the model
    :param white_op: white list of ONNX nodes allowed
        while converting a pipeline, if empty, all are allowed
    :param black_op: black list of ONNX nodes allowed
        while converting a pipeline, if empty, none are blacklisted
    :param final_types: a python list. Works the same way as initial_types
        but not mandatory, it is used to overwrites the type
        (if type is not None) and the name of every output.
    :param dtype: removed in version 1.7.5, dtype is now inferred from
        input types, converters may add operators Cast to switch to
        double when it is necessary
    :param naming: the user may want to change the way intermediate
        are named, this parameter can be a string (a prefix) or a
        function, which signature is the following:
        `get_name(name, existing_names)`, the library will then
        check this name is unique and modify it if not
    :param model_optim: enable or disable model optimisation
        after the model was converted into onnx, it reduces the number
        of identity nodes
    :param verbose: display progress while converting a model
    :return: converted model

    This function checks if the model inherits from class
    :class:`OnnxOperatorMixin`, it calls method *to_onnx*
    in that case otherwise it calls :func:`convert_sklearn`.

    .. versionchanged:: 1.10.0
        Parameter *naming* was added.
    """
    from .algebra.onnx_operator_mixin import OnnxOperatorMixin
    from .algebra.type_helper import guess_initial_types

    if isinstance(model, OnnxOperatorMixin):
        if options is not None:
            raise NotImplementedError(
                "options not yet implemented for OnnxOperatorMixin."
            )
        return model.to_onnx(X=X, name=name, target_opset=target_opset)
    if name is None:
        name = "ONNX(%s)" % model.__class__.__name__
    initial_types = guess_initial_types(X, initial_types)
    if verbose >= 1:
        print("[to_onnx] initial_types=%r" % initial_types)
    return convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=target_opset,
        name=name,
        options=options,
        white_op=white_op,
        black_op=black_op,
        final_types=final_types,
        dtype=dtype,
        verbose=verbose,
        naming=naming,
        model_optim=model_optim,
    )


def wrap_as_onnx_mixin(model, target_opset=None):
    """
    Combines a *scikit-learn* class with :class:`OnnxOperatorMixin`
    which produces a new object which combines *scikit-learn* API
    and *OnnxOperatorMixin* API.
    """
    from .algebra.sklearn_ops import find_class

    cl = find_class(model.__class__)
    if "automation" in str(cl):
        raise RuntimeError("Wrong class name '{}'.".format(cl))
    state = model.__getstate__()
    obj = object.__new__(cl)
    obj.__setstate__(state)
    obj.op_version = target_opset
    return obj
