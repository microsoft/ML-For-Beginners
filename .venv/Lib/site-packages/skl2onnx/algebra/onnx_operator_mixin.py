# SPDX-License-Identifier: Apache-2.0

import warnings
from sklearn.base import BaseEstimator
from onnx import shape_inference
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common._registration import get_converter, get_shape_calculator
from ..common._topology import Variable
from .._supported_operators import sklearn_operator_name_map
from .onnx_operator import OnnxOperator
from .type_helper import guess_initial_types


class OnnxOperatorMixin:
    """
    Base class for *scikit-learn* operators
    sharing an API to convert object to *ONNX*.
    """

    def to_onnx(
        self,
        X=None,
        name=None,
        options=None,
        white_op=None,
        black_op=None,
        final_types=None,
        target_opset=None,
        verbose=0,
    ):
        """
        Converts the model in *ONNX* format.
        It calls method *_to_onnx* which must be
        overloaded.

        :param X: training data, at least one sample,
            it is used to guess the type of the input data.
        :param name: name of the model, if None,
            it is replaced by the the class name.
        :param options: specific options given to converters
            (see :ref:`l-conv-options`)
        :param white_op: white list of ONNX nodes allowed
            while converting a pipeline, if empty, all are allowed
        :param black_op: black list of ONNX nodes allowed
            while converting a pipeline, if empty, none are blacklisted
        :param final_types: a python list. Works the same way as initial_types
            but not mandatory, it is used to overwrites the type
            (if type is not None) and the name of every output.
        :param target_opset: to overwrite `self.op_version`
        :param verbose: displays information while converting
        """
        from .. import convert_sklearn

        if X is None:
            initial_types = self.infer_initial_types()
        else:
            initial_types = guess_initial_types(X, None)
        if not hasattr(self, "op_version"):
            if name is None:
                name = self.__class__.__name__
            raise AttributeError(
                "Attribute 'op_version' is missing for '{}' "
                "(model: '{}').".format(self.__class__.__name__, name)
            )
        return convert_sklearn(
            self,
            initial_types=initial_types,
            target_opset=target_opset or self.op_version,
            options=options,
            white_op=white_op,
            black_op=black_op,
            final_types=final_types,
            verbose=verbose,
        )

    def infer_initial_types(self):
        """
        Infers initial types.
        """
        if hasattr(self, "enumerate_initial_types"):
            return list(self.enumerate_initial_types())
        raise RuntimeError(
            "Method enumerate_initial_types is missing "
            "and initial_types are not defined."
        )

    def _find_sklearn_parent(self):
        for cl in self.__class__.__bases__:
            if issubclass(cl, BaseEstimator):
                return cl
        raise RuntimeError(
            "Unable to find any parent inherited from "
            "BaseEstimator: {}.".format(", ".join(map(str, self.__class__.__bases__)))
        )

    def to_onnx_operator(
        self, inputs=None, outputs=None, target_opset=None, options=None
    ):
        """
        This function must be overloaded.
        """
        raise NotImplementedError()

    def onnx_parser(self):
        """
        Returns a parser for this model.
        If not overloaded, it calls the converter to guess the number
        of outputs. If it still fails, it fetches the parser
        mapped to the first *scikit-learn* parent
        it can find.
        """

        def parser(scope=None, inputs=None):
            try:
                op = self.to_onnx_operator(inputs=inputs, outputs=None)
            except NotImplementedError:
                self._find_sklearn_parent()
                return None

            names = []
            while True:
                try:
                    name = op.get_output_name(len(names), scope=scope)
                    if name is None:
                        break
                    names.append(name)
                except AttributeError:
                    return None
                except IndexError:
                    break
            return names

        return parser

    def get_inputs(self, inputs, i):
        if i >= len(inputs):
            return OnnxOperator.OnnxOperatorVariable(i)
        else:
            input = inputs[i]
            if isinstance(input, (str, OnnxOperator.UnscopedVariable)):
                return OnnxOperator.OnnxOperatorVariable(i, input)
            else:
                return input

    def onnx_shape_calculator(self):
        """
        Returns a shape calculator for this model.
        If not overloaded, it fetches the parser
        mapped to the first *scikit-learn* parent
        it can find.
        """
        if not hasattr(self, "op_version"):
            raise AttributeError(
                "Class '{}' should have an attribute 'op_version'.".format(
                    self.__class__.__name__
                )
            )

        try:
            op = self.to_onnx_operator()
        except NotImplementedError:
            parent = self._find_sklearn_parent()
            name = sklearn_operator_name_map.get(parent, "Sklearn" + parent.__name__)
            return get_shape_calculator(name)

        def shape_calculator(operator):
            onx = op.to_onnx(
                operator.inputs, operator.outputs, target_opset=self.op_version
            )
            inferred_model = shape_inference.infer_shapes(onx)
            shapes = Variable.from_pb(inferred_model.graph.value_info)
            shapes = {shape.onnx_name: shape for shape in shapes}
            for o in operator.outputs:
                name = o.onnx_name
                if name not in shapes:
                    raise RuntimeError(
                        "Shape of output '{}' cannot be "
                        "infered. onnx_shape_calculator "
                        "must be overriden and return "
                        "a shape calculator.".format(name)
                    )
                o.set_type(shapes[name].type)

        return shape_calculator

    def onnx_converter(self):
        """
        Returns a converter for this model.
        If not overloaded, it fetches the converter
        mapped to the first *scikit-learn* parent
        it can find.
        """

        def converter(
            scope: Scope, operator: Operator, container: ModelComponentContainer
        ):
            inputs = operator.inputs  # getattr(self, "parsed_inputs_", None)
            outputs = operator.outputs  # kwargs.get('outputs', None)
            op_version = container.target_opset
            options = scope.get_options(operator.raw_operator, fail=False)
            try:
                if inputs:
                    op = self.to_onnx_operator(
                        inputs=inputs,
                        outputs=outputs,
                        target_opset=op_version,
                        options=options,
                    )
                else:
                    op = self.to_onnx_operator(
                        target_opset=op_version, outputs=outputs, options=options
                    )
            except TypeError:
                warnings.warn(
                    "Signature should be to_onnx_operator(self, inputs=None, "
                    "outputs=None, target_opset=None, **kwargs). "
                    "This will be the case in version 1.11, class=%r."
                    "" % type(self),
                    DeprecationWarning,
                )
                try:
                    if inputs:
                        op = self.to_onnx_operator(inputs=inputs, outputs=outputs)
                    else:
                        op = self.to_onnx_operator()
                except NotImplementedError:
                    parent = self._find_sklearn_parent()
                    name = sklearn_operator_name_map[parent]
                    conv = get_converter(name)
                    return conv(scope, operator, container)
            except NotImplementedError:
                parent = self._find_sklearn_parent()
                name = sklearn_operator_name_map[parent]
                conv = get_converter(name)
                return conv(scope, operator, container)

            op.add_to(scope, container, operator=operator)

        return converter
