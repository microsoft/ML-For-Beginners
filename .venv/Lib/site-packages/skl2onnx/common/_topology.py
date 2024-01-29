# SPDX-License-Identifier: Apache-2.0


import re
import warnings
import pprint
from logging import getLogger
from collections import OrderedDict
import numpy as np
from onnx import onnx_pb as onnx_proto
from onnx.helper import make_graph, make_model, make_tensor_value_info
from onnxconverter_common.data_types import (  # noqa
    DataType,
    TensorType,
    FloatType,
    Int64Type,
    StringType,
    DictionaryType,
    FloatTensorType,  # noqa
    Int64TensorType,
    SequenceType,  # noqa
    StringTensorType,
    DoubleTensorType,
    Int32TensorType,
    BooleanTensorType,
    DoubleTensorType,
)

try:
    from onnxconverter_common.data_types import Int8TensorType, UInt8TensorType
except ImportError:
    Int8TensorType = None
    UInt8TensorType = None
from ..proto import get_opset_number_from_onnx, get_latest_tested_opset_version
from . import _registration
from . import utils
from .exceptions import MissingShapeCalculator, MissingConverter
from ._container import ModelComponentContainer, _build_options
from .onnx_optimisation_identity import onnx_remove_node_identity

type_fct = type


def _default_OPSET_TO_IR_VERSION():
    return {
        1: 3,
        2: 3,
        3: 3,
        4: 3,
        5: 3,
        6: 3,
        7: 3,
        8: 4,
        9: 4,
        10: 5,
        11: 6,
        12: 7,
        13: 7,
        14: 7,
        15: 8,
        16: 8,
        17: 8,
        18: 8,
        19: 9,
        20: 9,
    }


try:
    from onnxconverter_common.topology import OPSET_TO_IR_VERSION

    assert OPSET_TO_IR_VERSION[18] is not None
except (ImportError, KeyError):
    OPSET_TO_IR_VERSION = _default_OPSET_TO_IR_VERSION()

OPSET_ML_TO_OPSET = {1: 11, 2: 15, 3: 18}

logger = getLogger("skl2onnx")


def get_default_opset_for_domain(domain):
    """
    Returns the associated for a domain given the main opset.
    """
    from .. import __max_supported_opset__ as main_opset

    if domain == "":
        return main_opset
    if domain == "ai.onnx.ml":
        if main_opset >= 16:
            return 3
        if main_opset < 6:
            return 1
        return 2
    if domain == "ai.onnx.training":
        return 1
    return None


class Variable:
    """
    Defines a variable which holds any data defined
    from *ONNX* types.
    """

    _UNIQUE_NUMBER_ = 0

    def __init__(self, raw_name, onnx_name, scope, type=None):
        """
        :param raw_name: A string indicating the variable's name in the
                         original model. Usually, it's the seed string
                         used to created its ONNX name (i.e., the
                         field *onnx_name* below).
        :param onnx_name: A string indicating the variable's name in
                          the converted model
        :param scope: A string. It's the name of the scope where this
                      variable is declared
        :param type: A type object defined in .common.data_types.py;
                     e.g., FloatTensorType
        """
        if not isinstance(raw_name, str):
            raise TypeError("raw_name must be a string not '%s'." % raw_name.__class__)
        if type is not None and not hasattr(type, "shape"):
            raise TypeError(
                "Unexpected type for variable raw_name=%r, type=%r." % (raw_name, type)
            )
        if not isinstance(onnx_name, str) or "(" in onnx_name:
            if onnx_name.startswith("u(") and onnx_name[-1] == ")":
                onnx_name0 = onnx_name
                if scope is None:
                    onnx_name = "UU%03dUU" % Variable._UNIQUE_NUMBER_
                    Variable._UNIQUE_NUMBER_ += 1
                else:
                    onnx_name = scope.get_unique_variable_name("U")
                logger.debug(
                    "[Var] rename raw_name=%r, onnx_name=%r into %r",
                    raw_name,
                    onnx_name0,
                    onnx_name,
                )
            else:
                raise TypeError("onnx_name must be a string not %r." % onnx_name)

        if type is not None:
            shape = type.shape
            if shape is not None:
                not_none = [v for v in shape if v is not None]
                if len(not_none) and min(not_none) == 0:
                    raise RuntimeError(
                        "A variable cannot be empty, raw_name=%r, "
                        "onnx_name=%r, shape=%r, type=%r."
                        % (raw_name, onnx_name, shape, type)
                    )

        self._raw_name = raw_name
        self._onnx_name = onnx_name
        self._scope = scope
        self._type = type
        self._parent = None

        # The following fields are bool variables used in parsing and
        # compiling stages
        self._is_fed = None
        self._is_root = None
        self._is_leaf = None
        if self.type is not None and not isinstance(self.type, DataType):
            raise TypeError("shape must be a DataType not {}.".format(self.type))
        if isinstance(self.type, TensorType):
            shape = self.type.shape
            if not isinstance(shape, (list, tuple)):
                try:
                    shape = list(shape)
                except TypeError:
                    raise TypeError(
                        "shape must be a tuple or a list not "
                        "{}.".format(type_fct(shape))
                    )
            for dim in shape:
                if dim is None:
                    continue
                if not isinstance(dim, (int, np.int32, np.int64, np.intc)):
                    raise TypeError(
                        "shape must contains integers not %r (type=%r)."
                        "" % (dim, dim.__class__)
                    )
        logger.debug("[Var] +%s", self)

        # links to operators using those variables
        self.operators_outputs_ = []
        self.operators_inputs_ = []
        self._check()

    def _check(self):
        if self.type is not None and self.type.shape is not None:
            for k in self.type.shape:
                if k is None:
                    continue
                if not isinstance(k, (int, np.integer)):
                    raise ValueError(
                        "Unexpected type %r for shape %r." "" % (type(k), self)
                    )

    @property
    def raw_name(self):
        return self._raw_name

    @property
    def onnx_name(self):
        return self._onnx_name

    @property
    def scope(self):
        return self._scope

    @property
    def type(self):
        return self._type

    @property
    def is_fed(self):
        return self._is_fed

    @property
    def is_root(self):
        return self._is_root

    @property
    def is_leaf(self):
        return self._is_leaf

    def init_status(self, is_fed=None, is_root=None, is_leaf=None):
        if is_fed is not None and is_fed != self.is_fed:
            logger.debug(
                "[Var] update is_fed=%r for %r, parent=%r", is_fed, self, self._parent
            )
            self._is_fed = is_fed
        if is_root is not None and is_root != self.is_root:
            logger.debug("[Var] update is_root=%r for %r", is_root, self)
            self._is_root = is_root
        if is_leaf is not None and is_leaf != self.is_leaf:
            logger.debug("[Var] update is_leaf=%r for %r", is_leaf, self)
            self._is_leaf = is_leaf

    def __setattr__(self, name, value):
        if name == "type":
            self.set_type(value)
        elif name == "onnx_name":
            raise AttributeError("You must use method set_onnx_name.")
        elif name in {"is_fed", "is_root", "is_leaf"}:
            raise AttributeError("You must use method init_status.")
        elif name in {"scope", "raw_name"}:
            raise AttributeError("scope or raw_name cannot be changed.")
        self.__dict__[name] = value

    def set_type(self, new_type):
        if (
            new_type is None
            or isinstance(new_type, (str, Variable))
            or not hasattr(new_type, "shape")
        ):
            raise TypeError(
                "Unexpected new type for variable %r, new_type=%r." % (self, new_type)
            )
        logger.debug("[Var] update type for %r", self)
        self._type = new_type
        self._check()

    def set_onnx_name(self, onnx_name):
        if onnx_name != self._onnx_name:
            logger.debug(
                "[Var] update onnx_name, from %r to %r in %r",
                self.onnx_name,
                onnx_name,
                self,
            )
            if self.scope is not None and not isinstance(self.scope, str):
                self.scope.rename_onnx_name(self._onnx_name, onnx_name)
            self._onnx_name = onnx_name

    def set_parent(self, operator):
        if self._parent is not None:
            raise RuntimeError(
                "This variable is already the output of operator %r. "
                "It cannot be the output of %r." % (self._parent, operator)
            )
        logger.debug("[Var] set parent for %r, parent=%r", self, operator)
        self._parent = operator

    def get_first_dimension(self):
        """
        Returns the first dimension (batch dimension) or
        None if not specified (shape is empty).
        """
        if self.type is None or self.type.shape is None or len(self.type.shape) == 0:
            return None
        return self.type.shape[0]

    def get_second_dimension(self):
        if self.type is None or self.type.shape is None or len(self.type.shape) < 2:
            return None
        return self.type.shape[1]

    @property
    def full_name(self):
        """
        Return a globally unique variable ID
        """
        return self.onnx_name

    def __repr__(self):
        return "Variable('{0}', '{1}', type={2})".format(
            self.raw_name, self.onnx_name, self.type
        )

    @staticmethod
    def from_pb(obj):
        """
        Creates a data type from a protobuf object.
        """

        def get_dim(d):
            r = d.dim_value
            if "dim_param" in str(d):
                return None
            if r == 0:
                # dim_value is 0 when it is 0 or undefined
                return 0 if "0" in str(d) else None
            return r

        def get_shape(tt):
            return [get_dim(tt.shape.dim[i]) for i in range(len(tt.shape.dim))]

        if hasattr(obj, "extend"):
            return [Variable.from_pb(o) for o in obj]

        name = obj.name
        if obj.type.tensor_type:
            tt = obj.type.tensor_type
            elem = tt.elem_type
            shape = get_shape(tt)
            if elem == onnx_proto.TensorProto.FLOAT:
                ty = FloatTensorType(shape)
            elif elem == onnx_proto.TensorProto.BOOL:
                ty = BooleanTensorType(shape)
            elif elem == onnx_proto.TensorProto.DOUBLE:
                ty = DoubleTensorType(shape)
            elif elem == onnx_proto.TensorProto.STRING:
                ty = StringTensorType(shape)
            elif elem == onnx_proto.TensorProto.INT64:
                ty = Int64TensorType(shape)
            elif elem == onnx_proto.TensorProto.INT32:
                ty = Int32TensorType(shape)
            elif UInt8TensorType is not None and elem == onnx_proto.TensorProto.UINT8:
                ty = UInt8TensorType(shape)
            elif Int8TensorType is not None and elem == onnx_proto.TensorProto.INT8:
                ty = Int8TensorType(shape)
            elif elem == 0:
                ty = FloatTensorType(shape)
            else:
                raise NotImplementedError(
                    "Unsupported type '{}' (elem_type={}).".format(
                        type(obj.type.tensor_type), elem
                    )
                )
        else:
            raise NotImplementedError(
                "Unsupported type '{}' as " "a string ({}).".format(type(obj), obj)
            )

        return Variable(name, name, None, ty)

    def __iter__(self):
        "Enables expression such as `a,b = self`."
        yield self.onnx_name
        yield self.type

    def __getitem__(self, index):
        if index == 0:
            return self.onnx_name
        if index == 1:
            return self.type
        raise IndexError("Unreachable element at index %d." % index)

    def add_operator(self, op, in_or_out):
        "Add a link to an operator, True for output, False for input."
        if in_or_out:
            self.operators_outputs_.append(op)
        else:
            self.operators_inputs_.append(op)

    def check_compatible_type(self, other_type):
        def empty_shape(shape):
            return shape is None or len(shape) == 0

        if self.type is None:
            if other_type is None:
                return
        elif other_type is not None:
            if isinstance(self.type, type(other_type)):
                if self.type.shape == other_type.shape:
                    return
                if empty_shape(other_type.shape):
                    return
        raise TypeError(
            "Incompatible type for variable %r and type %r." % (self, other_type)
        )


class VariableStr(Variable):
    """
    Defines a variable a string. This should be avoided.
    """

    def __init__(self, name, scope=None, type=None):
        Variable.__init__(self, name, name, scope=scope, type=type)

    @property
    def raw_name(self):
        return self._raw_name

    @property
    def onnx_name(self):
        if self._onnx_name.startswith("u("):
            raise RuntimeError(
                "Variable should be renamed as onnx_name=%r." "" % self._onnx_name
            )
        return self._onnx_name


class Operator:
    """
    Defines an operator available in *ONNX*.
    """

    class OperatorList(list):
        def __init__(self, parent, kind):
            super(Operator.OperatorList, self).__init__()
            self.parent = parent
            self.kind = kind

        def __eq__(self, second):
            raise NotImplementedError("Operator equal not implemented and not needed.")

        def append(self, v):
            if not isinstance(v, Variable):
                raise TypeError(
                    "Input and output must be of type Variable not %r." "" % type(v)
                )
            if self.kind == "Out":
                v.set_parent(self.parent)
            super(Operator.OperatorList, self).append(v)
            logger.debug("[Op] add %s %r to %r", self.kind, v, self.parent)
            if self.kind == "In":
                v.add_operator(self.parent, False)
            elif self.kind == "Out":
                v.add_operator(self.parent, True)
            else:
                raise RuntimeError("Unexpected value for kind=%r." % self.kind)

        def extend(self, vs):
            for v in vs:
                self.append(v)

        def __getitem__(self, i):
            v = list.__getitem__(self, i)
            if isinstance(i, int) and not isinstance(v, Variable):
                raise TypeError("Element %d must be a Variable not %r." % (i, type(v)))
            return v

        def __setitem__(self, i, v):
            raise LookupError("Setter should not be used to modify an element.")

        def set_element(self, i, v):
            "Updates element i."
            if not isinstance(v, Variable):
                raise TypeError("Value v must be a Variable not %r." % type(v))
            logger.debug(
                "[Op] %s-change element %d from %r to %r in %r",
                self.kind,
                i,
                self[i],
                v,
                self.parent,
            )
            list.__setitem__(self, i, v)

        def to_string(self):
            names = []
            for o in self:
                if hasattr(o, "onnx_name"):
                    names.append(o.onnx_name)
                else:
                    names.append('"%s"' % str(o))
            return ",".join(names)

    def __init__(self, onnx_name, scope, type, raw_operator, target_opset, scope_inst):
        """
        :param onnx_name: A unique ID, which is a string
        :param scope: The name of the scope where this operator is
                      declared. It's a string.
        :param type: A object which uniquely characterizes the type of
                     this operator. For example, it can be a string,
                     pooling, if this operator is associated with a
                     CoreML pooling layer.
        :param raw_operator: The original operator which defines this operator;
                             for example, a scikit-learn Imputer and
                             a CoreML Normalizer.
        :param target_opset: The target opset number for the converted model.
        :param scope_inst: :class:`Scope` instance the operator belongs to
        """
        if isinstance(raw_operator, str):
            raise RuntimeError(
                "Parameter raw_operator must be an object not "
                "a string '{0}'.".format(raw_operator)
            )
        # operator name in the converted model, if raw_operator
        # is not None, output_shapes can be guessed
        # from the raw model. Otherwise, it can be guessed
        # from the input shapes.
        self.onnx_name = onnx_name
        self.scope = scope
        self.type = type
        self.raw_operator = raw_operator
        self.inputs = Operator.OperatorList(self, "In")
        self.outputs = Operator.OperatorList(self, "Out")
        self._is_evaluated = None
        self.target_opset = target_opset
        self.scope_inst = scope_inst
        logger.debug("[Op] +%r", self)

    def new_raw_operator(self, raw_operator, alias):
        """
        Returns a shallow copy of this operator,
        changes the raw_operator but keeps the same inputs
        and outputs.
        """
        op = Operator(
            self.onnx_name,
            self.scope,
            alias,
            raw_operator,
            self.target_opset,
            self.scope_inst,
        )
        op.inputs = self.inputs
        op.outputs = self.outputs
        return op

    def __repr__(self):
        try:
            textop = repr(self.raw_operator)
        except AttributeError:
            textop = "MISSING OP"
        except KeyError:
            # The line above fails for python 3.7
            textop = type(self.raw_operator)
        if isinstance(textop, str) and "\n" in textop:
            textop = textop.replace("\n", "").replace(" ", "")
        return (
            "Operator(type='{0}', onnx_name='{1}', inputs='{2}', "
            "outputs='{3}', raw_operator={4})".format(
                self.type,
                self.onnx_name,
                self.inputs.to_string(),
                self.outputs.to_string(),
                textop,
            )
        )

    def __setattr__(self, name, value):
        if name in ("inputs", "outputs"):
            if isinstance(value, list) and not isinstance(value, Operator.OperatorList):
                if name == "inputs":
                    self.inputs = Operator.OperatorList(self, "In")
                    self.inputs.extend(value)
                    return
                if name == "outputs":
                    self.outputs = Operator.OperatorList(self, "Out")
                    self.outputs.extend(value)
                    return
            if not isinstance(value, Operator.OperatorList):
                raise TypeError(
                    "inputs or outputs must be of type Operator.OperatorList."
                )
            ioo = name == "outputs"
            for v in value:
                v.add_operator(self, ioo)
        self.__dict__[name] = value

    @property
    def is_evaluated(self):
        return self._is_evaluated

    def init_status(self, is_evaluated=None):
        if is_evaluated is not None and is_evaluated != self.is_evaluated:
            logger.debug("[Op] update is_evaluated=%r for %r", is_evaluated, self)
            self._is_evaluated = is_evaluated

    @property
    def full_name(self):
        """
        Return a globally unique operator ID
        """
        return self.onnx_name

    @property
    def input_full_names(self):
        """
        Return all input variables' names
        """
        return [variable.full_name for variable in self.inputs]

    @property
    def output_full_names(self):
        """
        Return all output variables' names
        """
        return [variable.full_name for variable in self.outputs]

    @property
    def original_operator(self):
        """
        Return the original operator/layer
        """
        return self.raw_operator

    def infer_types(self):
        # Invoke a core inference function
        if self.type is None:
            raise MissingShapeCalculator(
                "Unable to find a shape calculator for type '{}'.".format(
                    type(self.raw_operator)
                )
            )
        try:
            shape_calc = _registration.get_shape_calculator(self.type)
        except ValueError:
            raise MissingShapeCalculator(
                "Unable to find a shape calculator for alias '{}' "
                "and type '{}'.".format(self.type, type(self.raw_operator))
            )
        if shape_calc is None:
            raise MissingShapeCalculator(
                "Unexpected shape calculator for alias '{}' "
                "and type '{}'.".format(self.type, type(self.raw_operator))
            )
        logger.debug(
            "[Shape-a] %r fed %r - %r",
            self,
            "".join(str(i.is_fed) for i in self.inputs),
            "".join(str(i.is_fed) for i in self.outputs),
        )
        shape_calc(self)
        logger.debug(
            "[Shape-b] %r inputs=%r - outputs=%r", self, self.inputs, self.outputs
        )


class Scope:
    """
    Every node of an *ONNX* graph must be unique. This class holds the list
    of existing name for every node already defined in graph. It also
    provides functions to create a unique unused name.
    """

    def __init__(
        self,
        name,
        target_opset=None,
        custom_shape_calculators=None,
        options=None,
        registered_models=None,
        naming=None,
    ):
        """
        :param name: A string, the unique ID of this scope in a
                     Topology object
        :param target_opset: The target opset number for the converted
                             model.
        :param custom_conversion_functions: a dictionary for specifying
                                the user customized conversion function
        :param custom_shape_calculators: a dictionary for specifying
                                the user customized shape calculator
        :param options: see :ref:`l-conv-options`
        :param naming: the user may want to change the way intermediate
            are named, this parameter can be a string (a prefix) or a
            function, which signature is the following:
            `get_name(name, existing_names)`, the library will then
            check this name is unique and modify it if not
        :param registered_models: registered models

        .. versionchanged:: 1.10.0
            Parameter *naming* was added.
        """
        self.name = name
        self.onnx_variable_names = set()
        self.onnx_operator_names = set()
        self.target_opset = target_opset
        self.custom_shape_calculators = custom_shape_calculators

        # An one-to-many map from raw variable name to ONNX variable
        # names. It looks like
        # (key, value) = (raw_name, [onnx_name, onnx_name1, onnx_name2, ..., onnx_nameN]) # noqa
        # The last name may hide all other names in this scope.
        self.variable_name_mapping = {}

        # A map of local variables defined in this scope.
        # (key, value) = (onnx_name, variable)
        self.variables = OrderedDict()
        self.input_variables = []
        self.output_variables = []

        # A map of local operators defined in this scope.
        # (key, value) = (onnx_name, operator)
        self.operators = {}

        # Additional options given to converters.
        self.options = options

        # Registered models
        self.registered_models = registered_models
        self.naming = naming

        if naming is None:
            self._naming = Topology._generate_unique_name
        elif isinstance(naming, str):
            self._naming = lambda seed, names: Topology._generate_unique_name(
                self.naming + seed, names
            )
        elif callable(self.naming):
            self._naming = lambda seed, names: Topology._generate_unique_name(
                self.naming(seed, names), names
            )
        else:
            raise TypeError("Unexpected type for parameter naming: %r." % type(naming))

    def get(self, var_name, default_value):
        "Returns variable with 'name' or default value is not found."
        return self.variables.get(var_name, default_value)

    def has_variable_name(self, name):
        """
        Tells if a variable is already registered.
        """
        return name in self.onnx_variable_names

    def get_shape_calculator(self, model_type):
        """
        Returns the shape calculator for the given model type.

        :param model_type: model type such as *LogisticRegression*
        :return: alias or None if not found
        """
        return self.custom_shape_calculators.get(model_type, None)

    def get_unique_variable_name(self, seed, rename=True):
        """
        Creates a unique variable ID based on the given seed.
        """
        if not isinstance(seed, str):
            raise TypeError(
                "Parameter seed must be a string not {}." "".format(type(seed))
            )
        if rename:
            name = self._naming(seed, self.onnx_variable_names)
        else:
            name = Topology._generate_unique_name(seed, self.onnx_variable_names)
        return name

    def get_unique_operator_name(self, seed):
        """
        Creates a unique operator ID based on the given seed.
        """
        return self._naming(seed, self.onnx_operator_names)

    def declare_local_variable(
        self, raw_name, type=None, prepend=False, missing_type=False, rename=True
    ):
        """
        This function may create a new variable in this scope. If
        *raw_name* has been used to create other variables, the new
        variable will hide all other variables created using *raw_name*.
        """
        if type is None and not missing_type:
            raise RuntimeError("Unknown type for %r (type=%r)." % (raw_name, type))
        # Get unique ID for the new variable
        onnx_name = self.get_unique_variable_name(raw_name, rename=rename)

        # Create the variable
        variable = Variable(raw_name, onnx_name, self.name, type)
        self.register_variable(variable, prepend=prepend)
        return variable

    def register_variable(self, var, prepend=False):
        "Adds a variable to the scope."
        if var.onnx_name in self.variables:
            raise RuntimeError(
                "Variable %r already registered (other=%r)."
                % (var, self.variables[var.onnx_name])
            )

        if var.raw_name in self.variable_name_mapping:
            # Hide existing variables with the same raw_name
            if not prepend:
                self.variable_name_mapping[var.raw_name].append(var.onnx_name)
            else:
                self.variable_name_mapping[var.raw_name].insert(0, var.onnx_name)
        else:
            self.variable_name_mapping[var.raw_name] = [var.onnx_name]

        self.variables[var.onnx_name] = var

    def declare_existing_subgraph_name(self, graph_proto):
        """
        Declare all name from a subgraph in order to avoid being picked twice.
        """
        output_name = {o.name for o in graph_proto.output}
        for node in graph_proto.node:
            for name in node.output:
                if name in output_name:
                    continue
                if self.has_variable_name(name):
                    raise NameError(
                        "Result name %r is already taken (outputs=%r) "
                        "(node=%r)." % (name, output_name, node)
                    )
                self.onnx_variable_names.add(name)
            if node.name in self.onnx_operator_names:
                raise NameError(
                    "Operator name %r is already taken "
                    "(node=%r)." % (node.name, node)
                )
            self.onnx_operator_names.add(node.name)

    def rename_onnx_name(self, old_name, new_name):
        if new_name in self.variables:
            raise RuntimeError(
                "Name %r already in variables (%r)."
                % (new_name, self.variables[new_name])
            )
        if old_name not in self.variables:
            raise RuntimeError("Unable to find name %r in variables." % old_name)
        logger.debug("[Scope] update onnx_name, from %r to %r", old_name, new_name)
        self.variables[new_name] = self.variables[old_name]
        del self.variables[old_name]

    def declare_local_input(self, raw_name, type=None, prepend=False, rename=True):
        """
        Calls `declare_local_variable`. Registers this variable
        as an input.
        """
        var = self.declare_local_variable(
            raw_name, type=type, prepend=prepend, rename=rename
        )
        self.input_variables.append(var)
        return var

    def declare_local_output(
        self, raw_name, type=None, prepend=False, missing_type=False
    ):
        """
        Calls `declare_local_variable`. Registers this variable
        as an output.
        """
        var = self.declare_local_variable(
            raw_name, type=type, prepend=prepend, missing_type=missing_type
        )
        self.output_variables.append(var)
        return var

    def declare_local_operator(self, type, raw_model=None):
        """
        This function is used to declare new local operator.
        """
        onnx_name = self.get_unique_operator_name(str(type))
        operator = Operator(
            onnx_name, self.name, type, raw_model, self.target_opset, scope_inst=self
        )
        self.operators[onnx_name] = operator
        return operator

    def _get_allowed_options(self, model, fail=True):
        if self.registered_models is not None:
            if type(model) not in self.registered_models["aliases"]:
                if fail:
                    raise NotImplementedError(
                        "No registered models, no known allowed options "
                        "for model '{}'.".format(model.__class__.__name__)
                    )
                return {}
            alias = self.registered_models["aliases"][type(model)]
            conv = self.registered_models["conv"][alias]
            allowed = conv.get_allowed_options()
            return allowed
        raise NotImplementedError(
            "No registered models, no known allowed options "
            "for model '{}'.".format(model.__class__.__name__)
        )

    def add_options(self, model_id, options):
        """
        Adds an option, for example,
        ``add_options(id(clr), {'raw_scores': True})``
        tells the converter associated to ``clr`` to
        use raw score instead of probabilities.

        :param model_id: class or ``id(instance)``
        :param options: dictionary with the new values
        """
        if options is None:
            return
        if self.options is None:
            self.options = {}
        if model_id not in self.options:
            self.options[model_id] = None
        if self.options[model_id] is None:
            self.options[model_id] = {}
        self.options[model_id].update(options)

    def get_options(self, model, default_values=None, fail=True):
        """
        Returns additional options for a model.
        It first looks by class then by id (``id(model)``).
        :param model: model being converted
        :param default_values: default options (it is modified by
                               the function)
        :param fail: fails if option it not found
        :return: dictionary
        """
        return _build_options(
            model,
            self.options,
            default_values,
            self._get_allowed_options(model, fail=fail),
            fail=fail,
        )

    def replace_raw_operator(self, op1, op2, alias):
        """
        Replaces every raw operator op1 by op2.
        The function uses `id()` to detect op1.
        """
        for v in self.operators.values():
            if id(v.raw_operator) == id(op1):
                logger.debug(
                    "[Scope] replace %d by %d in %r.", id(v.raw_operator), id(op1), v
                )
                v.raw_operator = op2
                v.type = alias


class Topology:
    """
    Holds instances on :class:`Scope <skl2onnx.common._topology.Scope>` and
    :class:`SklearnModelContainer
    <skl2onnx.common._container.SklearnModelContainer>`.
    These are filled by the converters while a pipeline is being converted.
    """

    def __init__(
        self,
        model,
        default_batch_size=1,
        initial_types=None,
        target_opset=None,
        custom_conversion_functions=None,
        custom_shape_calculators=None,
        registered_models=None,
    ):
        """
        Initializes a *Topology* object, which is an intermediate
        representation of a computational graph.

        :param model: RawModelContainer object or one of its derived
                      classes. It contains the original model.
        :param default_batch_size: batch_size prepend to scalar and
                                   array types from CoreML. It's usually
                                   1 or None.
        :param initial_types: A list providing some types for some
            root variables.
            Each element is a tuple of a variable name and a type defined
            in *data_types.py*.
        :param custom_conversion_functions: a dictionary for specifying
                                the user customized conversion function
        :param custom_shape_calculators: a dictionary for specifying the
                                        user customized shape calculator
        :param registered_models: registered models
        """
        self.scopes = []
        self.raw_model = model
        self.scope_names = set()
        self.initial_types = initial_types if initial_types else list()
        self.default_batch_size = default_batch_size
        self.target_opset = target_opset
        self.custom_conversion_functions = (
            custom_conversion_functions if custom_conversion_functions else {}
        )
        self.custom_shape_calculators = (
            custom_shape_calculators if custom_shape_calculators else {}
        )

        for k in self.custom_conversion_functions:
            if not callable(k):
                raise TypeError(
                    "Keys in custom_conversion_functions must be types not strings."
                )
        for k in self.custom_shape_calculators:
            if not callable(k):
                raise TypeError(
                    "Keys in custom_shape_calculators must be types not strings."
                )

        # A map of local overwritten model aliases.
        self.model_aliases = {}
        all_model_types = set(self.custom_conversion_functions) | set(
            self.custom_shape_calculators
        )
        for mtype in all_model_types:
            alias = "{}_{}".format(mtype.__name__, id(self))
            self.model_aliases[mtype] = alias

        # Registered models
        if registered_models is None:
            raise AssertionError()
        self.registered_models = registered_models

    @property
    def scope(self):
        if len(self.scopes) != 1:
            raise RuntimeError("Only one scope is allowed not %d." % len(self.scopes))
        return self.scopes[0]

    @staticmethod
    def _generate_unique_name(seed, existing_names):
        """
        Produce an unique string based on the seed
        :param seed: a string
        :param existing_names: a set containing strings which cannot be
                               produced
        :return: a string similar to the seed
        """
        if seed == "":
            raise ValueError("Name seed must be a non-empty string.")

        # Make the seed meet C-style naming convention
        # Only alphabets and numbers are allowed
        seed = re.sub("[^\\w+]", "_", seed)
        # The first symbol cannot be a number
        if re.match("^[0-9]", seed):
            seed = "_" + seed

        # If seed has never been seen, we return it as it is. Otherwise,
        # we will append an number to make it unique.
        if seed not in existing_names:
            existing_names.add(seed)
            return seed
        else:
            i = 1
            while seed + str(i) in existing_names:
                i += 1
            new_name = seed + str(i)
            existing_names.add(new_name)
            return new_name

    def get_unique_scope_name(self, seed):
        return Topology._generate_unique_name(seed, self.scope_names)

    def declare_scope(self, seed, parent_scopes=None, options=None, naming=None):
        """
        Creates a new :class:`Scope <skl2onnx.common._topology.Scope>`
        and appends it to the list of existing scopes.
        """
        if len(self.scopes) != 0:
            raise RuntimeError("Only one scope can be created.")
        scope = Scope(
            self.get_unique_scope_name(seed),
            target_opset=self.target_opset,
            custom_shape_calculators=self.custom_shape_calculators,
            options=options,
            registered_models=self.registered_models,
            naming=naming,
        )

        # Declare input variables.
        # They should be the inputs of the scikit-learn
        # model you want to convert into ONNX.
        for var_name, initial_type in self.initial_types:
            scope.declare_local_input(var_name, initial_type, rename=False)
        self.scopes.append(scope)
        return scope

    def unordered_operator_iterator(self):
        for scope in self.scopes:
            for operator in scope.operators.values():
                yield operator

    def unordered_variable_iterator(self):
        for scope in self.scopes:
            for variable in scope.variables.values():
                yield variable

    def call_converter(self, operator, container, verbose=0):
        "Calls converter for operator *operator*."
        mtype = type(operator.raw_operator)
        if mtype in self.custom_conversion_functions:
            conv = self.custom_conversion_functions[mtype]
        elif operator.type in self.custom_conversion_functions:
            conv = self.custom_conversion_functions[operator.type]
        elif hasattr(operator.raw_operator, "onnx_converter"):
            conv = operator.raw_operator.onnx_converter()
        else:
            # Convert the selected operator into some ONNX objects and
            # save them into the container
            try:
                conv = _registration.get_converter(operator.type)
            except ValueError:
                raise MissingConverter(
                    "Unable to find converter for alias '{}' type "
                    "'{}'. You may raise an issue at "
                    "https://github.com/onnx/sklearn-onnx/issues."
                    "".format(operator.type, type(getattr(operator, "raw_model", None)))
                )

        container.validate_options(operator)
        if verbose > 0:
            print("[call_converter] call converter for %r." % operator.type)
        logger.debug(
            "[Conv] call %r fed %r - %r",
            operator,
            "".join(str(i.is_fed) for i in operator.inputs),
            "".join(str(i.is_fed) for i in operator.outputs),
        )
        conv(self.scopes[0], operator, container)
        logger.debug("[Conv] end - %r", operator)

    def call_shape_calculator(self, operator):
        "Calls shape_calculator for operator *operator*."
        mtype = type(operator.raw_operator)
        if mtype in self.custom_shape_calculators:
            # overwritten operator.
            source = "custom"
            shape_calc = self.custom_shape_calculators[mtype]
        elif operator.type in self.custom_shape_calculators:
            source = "custom"
            shape_calc = self.custom_shape_calculators[operator.type]
        elif hasattr(operator.raw_operator, "onnx_shape_calculator"):
            source = "onnx_shape_calculator"
            shape_calc = operator.raw_operator.onnx_shape_calculator()
        else:
            source = ""
            shape_calc = None

        if shape_calc is not None:
            logger.debug(
                "[Shape1] %r fed %r - %r (source=%r)",
                operator,
                ",".join(str(i.is_fed) for i in operator.inputs),
                ",".join(str(i.is_fed) for i in operator.outputs),
                source,
            )
            shape_calc(operator)
        else:
            logger.debug("[Shape2] call infer_types for %r", operator)
            operator.infer_types()

    def _initialize_graph_status_for_traversing(self):
        """
        Initialize the status of all variables and operators before
        traversing the graph. Only used by convert_operators.
        """
        if len(self.scopes) != 1:
            raise RuntimeError("Only one scope is allowed not %d." % len(self.scopes))
        input_names = set(v.onnx_name for v in self.scopes[0].input_variables)
        if len(input_names) == 0:
            raise RuntimeError("No detected inputs.")
        for variable in self.unordered_variable_iterator():
            is_input = variable.onnx_name in input_names
            variable.init_status(is_fed=is_input)

        for operator in self.unordered_operator_iterator():
            operator.init_status(is_evaluated=False)

    def _propagate_status(self, operator, container, fed_variables, verbose=0):
        """
        Propagates status *is_fed* based on output variable
        and node added in the container.
        """
        if verbose > 1:
            print("[_propagate_status] after op=%r" % operator)
        vars = {}
        for node in container.nodes:
            for i in node.input:
                if i not in vars:
                    vars[i] = []
                vars[i].append(node)

        if verbose > 1:
            print(
                "[_propagate_status] newly fed=%r"
                % list(v.onnx_name for v in operator.outputs if v.is_fed)
            )
        stack = list(fed_variables)
        scope = self.scopes[0]
        while len(stack) > 0:
            nodes = {}
            for name in stack:
                if name not in vars:
                    continue
                for n in vars[name]:
                    nodes[id(n)] = n
            stack = []
            for node in nodes.values():
                if all(fed_variables.get(n, False) for n in node.input):
                    for o in node.output:
                        if o not in fed_variables:
                            if verbose > 1:
                                print("[_propagate_status] add=%r" % o)
                            fed_variables[o] = o
                            stack.append(o)
                            if o in scope.variables:
                                var = scope.variables[o]
                                var.init_status(is_fed=True)
                                if verbose > 1:
                                    print("[_propagate_status] fed=%r" % var)

    def convert_operators(self, container=None, verbose=0):
        """
        Calls all converters and shape_calculator for existing
        operators. It also processes new operators created by
        converters.
        """

        def _check_operator_(operator):
            if not isinstance(operator.inputs, Operator.OperatorList):
                raise TypeError(
                    "operator.inputs must be a Operator.OperatorList "
                    "not %r." % type(operator.inputs)
                )
            if not isinstance(operator.outputs, Operator.OperatorList):
                raise TypeError(
                    "operator.outputs must be a Operator.OperatorList "
                    "not %r." % type(operator.outputs)
                )
            if any(not isinstance(i, Variable) for i in operator.inputs):
                raise TypeError(
                    "One input is not a Variable for operator %r - %r."
                    "" % (type(operator.raw_operator), operator)
                )
            if any(not isinstance(i, Variable) for i in operator.outputs):
                raise TypeError(
                    "One output is not a Variable for operator %r - %r."
                    "" % (type(operator.raw_operator), operator)
                )

        def _check_variable_in_(variable, operator):
            idop = id(operator)
            ids = set(id(op) for op in variable.operators_inputs_)
            if idop not in ids:
                raise RuntimeError(
                    "Operator %r not registered in the list of operators "
                    "of %r taking it as an input [\n%s]."
                    % (
                        operator,
                        variable,
                        "\n".join(map(str, variable.operators_inputs_)),
                    )
                )

        def _check_variable_out_(variable, operator):
            if variable.is_fed:
                add = ["", "--DEBUG-INFO--"]
                for scope in self.scopes:
                    add.append("---")
                    add.append(pprint.pformat(scope.variable_name_mapping))
                    add.append("---")
                    for var in scope.variables.values():
                        add.append(
                            "   is_fed=%s %s - n_in=%d n_out=%d"
                            % (
                                getattr(var, "is_fed", "?"),
                                var,
                                len(var.operators_inputs_),
                                len(var.operators_outputs_),
                            )
                        )
                    add.append("---")
                    for op in scope.operators.values():
                        add.append(
                            "   is_evaluated=%s %s"
                            % (getattr(op, "is_evaluated", "?"), op)
                        )
                add.append("---")
                for v in operator.inputs:
                    add.append(" inputs={}".format(v))
                for v in operator.outputs:
                    add.append(" outputs={}".format(v))
                add.append("--- operator producing this variable--")
                for op in variable.operators_outputs_:
                    add.append(str(op))
                raise RuntimeError(
                    "A variable is already assigned ({}) "
                    "for operator '{}' (name='{}'). "
                    "operator.is_evaluated={}, inputs.is_fed={}, "
                    "outputs.is_fed={}. "
                    "This may still happen if a converter is a "
                    "combination of sub-estimators and one "
                    "of them is producing this output. "
                    "In that case, an identity node must be "
                    "added.{}".format(
                        variable,
                        operator.type,
                        operator.onnx_name,
                        operator.is_evaluated,
                        [v.is_fed for v in operator.inputs],
                        [v.is_fed for v in operator.outputs],
                        "\n".join(add),
                    )
                )

        if verbose > 0:
            print("[convert_operators] begin")
        self._initialize_graph_status_for_traversing()
        fed_variables = {i.name: i for i in container.initializers}
        changes = 1
        n_iter = 0
        while changes > 0:
            n_iter += 1
            changes = 0
            ops = list(self.unordered_operator_iterator())
            if verbose > 0:
                print(
                    "[convert_operators] iteration %d - n_vars=%d "
                    "n_ops=%d" % (n_iter, len(fed_variables), len(ops))
                )
            for operator in ops:
                _check_operator_(operator)
                for var in operator.inputs:
                    if var.is_fed:
                        fed_variables[var.onnx_name] = var
                if (
                    all(variable.is_fed for variable in operator.inputs)
                    and not operator.is_evaluated
                ):
                    for variable in operator.inputs:
                        _check_variable_in_(variable, operator)
                    for variable in operator.outputs:
                        _check_variable_out_(variable, operator)

                    self.call_shape_calculator(operator)
                    self.call_converter(operator, container, verbose=verbose)

                    # If an operator contains a sequence of operators,
                    # output variables are not necessarily known at this stage.
                    operator.init_status(is_evaluated=True)
                    for variable in operator.outputs:
                        if all(op.is_evaluated for op in variable.operators_outputs_):
                            variable.init_status(is_fed=True)
                            fed_variables[variable.onnx_name] = variable
                    fed_variables.update(
                        {
                            i.name: i
                            for i in container.initializers
                            if i.name not in fed_variables
                        }
                    )
                    self._propagate_status(
                        operator, container, fed_variables, verbose=verbose
                    )

                    # unfed some variables (it happens when a node
                    # shares an output with another node)
                    rem = []
                    for n, var in fed_variables.items():
                        if not hasattr(var, "operators_outputs_"):
                            # initializer
                            continue
                        if any(not o.is_evaluated for o in var.operators_outputs_):
                            rem.append(n)
                    for r in rem:
                        v = fed_variables[r]
                        v.init_status(is_fed=False)
                        del fed_variables[v.onnx_name]
                    changes += 1

            if verbose > 0:
                print(
                    "[convert_operators] end iter: %d - n_vars=%d"
                    % (n_iter, len(fed_variables))
                )
        if verbose > 0:
            print("[convert_operators] end.")

        # Last verification.
        not_evaluated = []
        for op in self.unordered_operator_iterator():
            if not op.is_evaluated:
                not_evaluated.append(op)
        if len(not_evaluated) > 0:
            rows = ["---VARS---"]
            for var in self.unordered_variable_iterator():
                rows.append(
                    "is_fed=%r is_leaf=%r is_root=%r - %r - n_in=%d n_out=%d"
                    ""
                    % (
                        var.is_fed,
                        var.is_leaf,
                        var.is_root,
                        var,
                        len(var.operators_inputs_),
                        len(var.operators_outputs_),
                    )
                )
            rows.append("---OPERATORS---")
            for op in self.unordered_operator_iterator():
                rows.append("is_eval=%r - %r" % (op.is_evaluated, op))
            rows.append("---NODES---")
            for node in container.nodes:
                rows.append("%s: %r -> %r" % (node.op_type, node.input, node.output))
            raise RuntimeError(
                "Not all operators have been evaluated. A variable name "
                "is probably misspelled.\n%s"
                "" % "\n".join(rows)
            )

        # Input and output
        if len(self.scopes[0].input_variables) > 0:
            inputs = self.scopes[0].input_variables
        else:
            inputs = [v for v in self.unordered_variable_iterator() if v.is_root]
        for i in inputs:
            container.add_input(i)
        outputs = [v for v in self.unordered_variable_iterator() if v.is_leaf]

        # The function checks that for output variable,
        # raw_name equal onnx_name. It swaps names if it is not the case.
        to_swap = []
        for out in outputs:
            if out.raw_name != out.onnx_name:
                to_swap.append(out)
        if len(to_swap) != 0:
            swaped = set()
            for var in to_swap:
                if var.raw_name in swaped:
                    continue
                swaped.add(var.raw_name)
                if verbose > 1:
                    print(
                        "[convert_operators] %r <-> %r." % (var.raw_name, var.onnx_name)
                    )
                old_name = var.onnx_name
                new_name = var.raw_name

                try:
                    container.swap_names(old_name, new_name)
                except NotImplementedError as e:
                    logger.debug(
                        "[Topo] unable to swap %r and %r (%r).", old_name, new_name, e
                    )
                    continue

                for v in self.unordered_variable_iterator():
                    if v.onnx_name == old_name:
                        v.set_onnx_name(new_name)
                    elif v.onnx_name == new_name:
                        v.set_onnx_name(old_name)

        for o in outputs:
            container.add_output(o)


def convert_topology(
    topology,
    model_name,
    doc_string,
    target_opset,
    options=None,
    remove_identity=True,
    verbose=0,
):
    """
    This function is used to convert our Topology object defined in
    _parser.py into a ONNX model (type: ModelProto).

    :param topology: The Topology object we are going to convert
    :param model_name: GraphProto's name. Let "model" denote the
        returned model. The string "model_name" would be
        assigned to "model.graph.name."
    :param doc_string: A string attached to the produced model
    :param target_opset: number or dictionary,
        for example, 7 for ONNX 1.2, and 8 for ONNX 1.3,
        a dictionary is used to indicate different opset for
        different domains
    :param options: see :ref:`l-conv-options`
    :param remove_identity: removes identity nodes
        include '1.1.2', '1.2', and so on.
    :param verbose: displays information while converting
    :return: a ONNX ModelProto
    """
    if target_opset is None:
        target_opset = get_latest_tested_opset_version()
    if isinstance(target_opset, dict):
        onnx_target_opset = target_opset.get("", get_latest_tested_opset_version())
    else:
        onnx_target_opset = target_opset
    if onnx_target_opset > get_opset_number_from_onnx():
        found = get_opset_number_from_onnx()
        raise RuntimeError(
            "Parameter target_opset {} > {} is higher than the "
            "version of the installed onnx package. See "
            "https://github.com/onnx/onnx/blob/master/docs/"
            "Versioning.md#released-versions"
            ".".format(onnx_target_opset, found)
        )
    if onnx_target_opset > get_latest_tested_opset_version():
        warnings.warn(
            "Parameter target_opset {} > {} is higher than the "
            "the latest tested version"
            ".".format(onnx_target_opset, get_latest_tested_opset_version())
        )

    container = ModelComponentContainer(
        target_opset,
        options=options,
        registered_models=topology.registered_models,
        white_op=topology.raw_model._white_op,
        black_op=topology.raw_model._black_op,
        verbose=verbose,
    )

    # Traverse the graph from roots to leaves
    # This loop could eventually be parallelized.
    topology.convert_operators(container=container, verbose=verbose)
    container.ensure_topological_order()

    if len(container.inputs) == 0:
        raise RuntimeError("No detected inputs after conversion.")
    if len(container.outputs) == 0:
        raise RuntimeError("No detected outputs after conversion.")
    if verbose >= 2:
        print("---NODES---")
        for node in container.nodes:
            print(
                "  %s - %s: %r -> %r"
                % (node.op_type, node.name, node.input, node.output)
            )

    # Create a graph from its main components
    if container.target_opset_onnx < 9:
        # When calling ModelComponentContainer's add_initializer(...),
        # nothing is added into the input list. However, for ONNX target
        # opset < 9, initializers should also be a part of model's
        # (GraphProto) inputs. Thus, we create ValueInfoProto objects
        # from initializers (type: TensorProto) directly and then add
        # them into model's input list.
        extra_inputs = []  # ValueInfoProto list of the initializers
        for tensor in container.initializers:
            # Sometimes (especially when creating optional input values
            # such as RNN's initial hidden state), an initializer is also
            # one of the original model's input, so it has been added into
            # the container's input list. If this is the case, we need to
            # skip one iteration to avoid duplicated inputs.
            if tensor.name in [value_info.name for value_info in container.inputs]:
                continue

            # Initializers are always tensors so we can just call
            # make_tensor_value_info(...).
            value_info = make_tensor_value_info(
                tensor.name, tensor.data_type, tensor.dims
            )
            extra_inputs.append(value_info)

        # Before ONNX opset 9, initializers were needed to be passed in
        # with inputs.
        graph = make_graph(
            container.nodes,
            model_name,
            container.inputs + extra_inputs,
            container.outputs,
            container.initializers,
        )
    else:
        # In ONNX opset 9 and above, initializers are included as
        # operator inputs and therefore do not need to be passed as
        # extra_inputs.
        graph = make_graph(
            container.nodes,
            model_name,
            container.inputs,
            container.outputs,
            container.initializers,
        )

    # Add extra information related to the graph
    graph.value_info.extend(container.value_info)

    # Create model
    onnx_model = make_model(graph)

    # Update domain version
    opv = min(
        onnx_target_opset, _get_main_opset_version(onnx_model) or onnx_target_opset
    )
    if not _update_domain_version(container, onnx_model, verbose=verbose):
        # Main opset was not added. Doing it here.
        op_set = onnx_model.opset_import.add()
        op_set.domain = ""
        op_set.version = opv
        if verbose > 0:
            print("[convert_topology] +opset: name=%r, version=%s" % ("", opv))

    # Add extra information
    irv = OPSET_TO_IR_VERSION.get(opv, onnx_proto.IR_VERSION)
    onnx_model.ir_version = irv
    onnx_model.producer_name = utils.get_producer()
    onnx_model.producer_version = utils.get_producer_version()
    onnx_model.domain = utils.get_domain()
    onnx_model.model_version = utils.get_model_version()
    onnx_model.doc_string = doc_string

    # Removes many identity nodes,
    # the converter may introduct identity nodes
    # after a zipmap operator and onnx <= 1.7 does not
    # support that. It does not use onnxconverter-common
    # as the optimizer only support opset >= 9.
    if remove_identity:
        onnx_model = onnx_remove_node_identity(onnx_model)

    return onnx_model


def _update_domain_version(container, onnx_model, verbose=0):
    # Merge operator sets for the same domain, the largest version
    # number would be kept
    purified_operator_set = dict()
    for op_domain, op_version in container.node_domain_version_pair_sets:
        if op_domain not in purified_operator_set:
            purified_operator_set[op_domain] = op_version
        else:
            purified_operator_set[op_domain] = max(
                purified_operator_set[op_domain], op_version
            )

    # Fill operator sets
    i = 0
    for op_domain, op_version in purified_operator_set.items():
        if op_version is None:
            continue
        if i == 0 and len(onnx_model.opset_import) == 1:
            # Overwrite the default operator set created by
            # make_model(...)
            op_set = onnx_model.opset_import[0]
        else:
            # Just create one ONNX element in opset_import
            op_set = onnx_model.opset_import.add()
        if verbose > 0:
            print(
                "[_update_domain_version] +opset %d: name=%r, version=%s"
                % (i, op_domain, op_version)
            )
        op_set.domain = op_domain
        if op_set != "":
            max_supported = get_default_opset_for_domain(op_domain)
            if max_supported is not None and max_supported < op_version:
                raise RuntimeError(
                    "The model is using version %d of domain %r not supported "
                    "yet by this library. You need to specify "
                    "target_opset={%r: %r}."
                    % (op_version, op_domain, op_domain, max_supported)
                )
        op_set.version = op_version

        i += 1
        if container.target_opset_any_domain(op_domain) < op_version:
            raise RuntimeError(
                "The specified opset %d is too low to convert "
                "this model, which requires at least opset "
                "%d." % (container.target_opset_any_domain(op_domain), op_version)
            )
    return "" in purified_operator_set


def _get_main_opset_version(model):
    """
    Returns the main opset version.
    """
    mld = None
    for op in model.opset_import:
        if op.domain == "":
            return op.version
        if op.domain == "ai.onnx.ml":
            mld = op.version
    if mld is not None:
        return OPSET_ML_TO_OPSET.get(mld, None)
    return None
