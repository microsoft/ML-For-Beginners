from typing import (
    Any,
    Callable,
    Dict,
    Set,
    Sequence,
    Tuple,
    NamedTuple,
    Type,
    Literal,
    Union,
    TYPE_CHECKING,
)
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType

from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.decorators import undoc


if TYPE_CHECKING or GENERATING_DOCUMENTATION:
    from typing_extensions import Protocol
else:
    # do not require on runtime
    Protocol = object  # requires Python >=3.8


@undoc
class HasGetItem(Protocol):
    def __getitem__(self, key) -> None:
        ...


@undoc
class InstancesHaveGetItem(Protocol):
    def __call__(self, *args, **kwargs) -> HasGetItem:
        ...


@undoc
class HasGetAttr(Protocol):
    def __getattr__(self, key) -> None:
        ...


@undoc
class DoesNotHaveGetAttr(Protocol):
    pass


# By default `__getattr__` is not explicitly implemented on most objects
MayHaveGetattr = Union[HasGetAttr, DoesNotHaveGetAttr]


def _unbind_method(func: Callable) -> Union[Callable, None]:
    """Get unbound method for given bound method.

    Returns None if cannot get unbound method, or method is already unbound.
    """
    owner = getattr(func, "__self__", None)
    owner_class = type(owner)
    name = getattr(func, "__name__", None)
    instance_dict_overrides = getattr(owner, "__dict__", None)
    if (
        owner is not None
        and name
        and (
            not instance_dict_overrides
            or (instance_dict_overrides and name not in instance_dict_overrides)
        )
    ):
        return getattr(owner_class, name)
    return None


@undoc
@dataclass
class EvaluationPolicy:
    """Definition of evaluation policy."""

    allow_locals_access: bool = False
    allow_globals_access: bool = False
    allow_item_access: bool = False
    allow_attr_access: bool = False
    allow_builtins_access: bool = False
    allow_all_operations: bool = False
    allow_any_calls: bool = False
    allowed_calls: Set[Callable] = field(default_factory=set)

    def can_get_item(self, value, item):
        return self.allow_item_access

    def can_get_attr(self, value, attr):
        return self.allow_attr_access

    def can_operate(self, dunders: Tuple[str, ...], a, b=None):
        if self.allow_all_operations:
            return True

    def can_call(self, func):
        if self.allow_any_calls:
            return True

        if func in self.allowed_calls:
            return True

        owner_method = _unbind_method(func)

        if owner_method and owner_method in self.allowed_calls:
            return True


def _get_external(module_name: str, access_path: Sequence[str]):
    """Get value from external module given a dotted access path.

    Raises:
    * `KeyError` if module is removed not found, and
    * `AttributeError` if acess path does not match an exported object
    """
    member_type = sys.modules[module_name]
    for attr in access_path:
        member_type = getattr(member_type, attr)
    return member_type


def _has_original_dunder_external(
    value,
    module_name: str,
    access_path: Sequence[str],
    method_name: str,
):
    if module_name not in sys.modules:
        # LBYLB as it is faster
        return False
    try:
        member_type = _get_external(module_name, access_path)
        value_type = type(value)
        if type(value) == member_type:
            return True
        if method_name == "__getattribute__":
            # we have to short-circuit here due to an unresolved issue in
            # `isinstance` implementation: https://bugs.python.org/issue32683
            return False
        if isinstance(value, member_type):
            method = getattr(value_type, method_name, None)
            member_method = getattr(member_type, method_name, None)
            if member_method == method:
                return True
    except (AttributeError, KeyError):
        return False


def _has_original_dunder(
    value, allowed_types, allowed_methods, allowed_external, method_name
):
    # note: Python ignores `__getattr__`/`__getitem__` on instances,
    # we only need to check at class level
    value_type = type(value)

    # strict type check passes → no need to check method
    if value_type in allowed_types:
        return True

    method = getattr(value_type, method_name, None)

    if method is None:
        return None

    if method in allowed_methods:
        return True

    for module_name, *access_path in allowed_external:
        if _has_original_dunder_external(value, module_name, access_path, method_name):
            return True

    return False


@undoc
@dataclass
class SelectivePolicy(EvaluationPolicy):
    allowed_getitem: Set[InstancesHaveGetItem] = field(default_factory=set)
    allowed_getitem_external: Set[Tuple[str, ...]] = field(default_factory=set)

    allowed_getattr: Set[MayHaveGetattr] = field(default_factory=set)
    allowed_getattr_external: Set[Tuple[str, ...]] = field(default_factory=set)

    allowed_operations: Set = field(default_factory=set)
    allowed_operations_external: Set[Tuple[str, ...]] = field(default_factory=set)

    _operation_methods_cache: Dict[str, Set[Callable]] = field(
        default_factory=dict, init=False
    )

    def can_get_attr(self, value, attr):
        has_original_attribute = _has_original_dunder(
            value,
            allowed_types=self.allowed_getattr,
            allowed_methods=self._getattribute_methods,
            allowed_external=self.allowed_getattr_external,
            method_name="__getattribute__",
        )
        has_original_attr = _has_original_dunder(
            value,
            allowed_types=self.allowed_getattr,
            allowed_methods=self._getattr_methods,
            allowed_external=self.allowed_getattr_external,
            method_name="__getattr__",
        )

        accept = False

        # Many objects do not have `__getattr__`, this is fine.
        if has_original_attr is None and has_original_attribute:
            accept = True
        else:
            # Accept objects without modifications to `__getattr__` and `__getattribute__`
            accept = has_original_attr and has_original_attribute

        if accept:
            # We still need to check for overriden properties.

            value_class = type(value)
            if not hasattr(value_class, attr):
                return True

            class_attr_val = getattr(value_class, attr)
            is_property = isinstance(class_attr_val, property)

            if not is_property:
                return True

            # Properties in allowed types are ok (although we do not include any
            # properties in our default allow list currently).
            if type(value) in self.allowed_getattr:
                return True  # pragma: no cover

            # Properties in subclasses of allowed types may be ok if not changed
            for module_name, *access_path in self.allowed_getattr_external:
                try:
                    external_class = _get_external(module_name, access_path)
                    external_class_attr_val = getattr(external_class, attr)
                except (KeyError, AttributeError):
                    return False  # pragma: no cover
                return class_attr_val == external_class_attr_val

        return False

    def can_get_item(self, value, item):
        """Allow accessing `__getiitem__` of allow-listed instances unless it was not modified."""
        return _has_original_dunder(
            value,
            allowed_types=self.allowed_getitem,
            allowed_methods=self._getitem_methods,
            allowed_external=self.allowed_getitem_external,
            method_name="__getitem__",
        )

    def can_operate(self, dunders: Tuple[str, ...], a, b=None):
        objects = [a]
        if b is not None:
            objects.append(b)
        return all(
            [
                _has_original_dunder(
                    obj,
                    allowed_types=self.allowed_operations,
                    allowed_methods=self._operator_dunder_methods(dunder),
                    allowed_external=self.allowed_operations_external,
                    method_name=dunder,
                )
                for dunder in dunders
                for obj in objects
            ]
        )

    def _operator_dunder_methods(self, dunder: str) -> Set[Callable]:
        if dunder not in self._operation_methods_cache:
            self._operation_methods_cache[dunder] = self._safe_get_methods(
                self.allowed_operations, dunder
            )
        return self._operation_methods_cache[dunder]

    @cached_property
    def _getitem_methods(self) -> Set[Callable]:
        return self._safe_get_methods(self.allowed_getitem, "__getitem__")

    @cached_property
    def _getattr_methods(self) -> Set[Callable]:
        return self._safe_get_methods(self.allowed_getattr, "__getattr__")

    @cached_property
    def _getattribute_methods(self) -> Set[Callable]:
        return self._safe_get_methods(self.allowed_getattr, "__getattribute__")

    def _safe_get_methods(self, classes, name) -> Set[Callable]:
        return {
            method
            for class_ in classes
            for method in [getattr(class_, name, None)]
            if method
        }


class _DummyNamedTuple(NamedTuple):
    """Used internally to retrieve methods of named tuple instance."""


class EvaluationContext(NamedTuple):
    #: Local namespace
    locals: dict
    #: Global namespace
    globals: dict
    #: Evaluation policy identifier
    evaluation: Literal[
        "forbidden", "minimal", "limited", "unsafe", "dangerous"
    ] = "forbidden"
    #: Whether the evalution of code takes place inside of a subscript.
    #: Useful for evaluating ``:-1, 'col'`` in ``df[:-1, 'col']``.
    in_subscript: bool = False


class _IdentitySubscript:
    """Returns the key itself when item is requested via subscript."""

    def __getitem__(self, key):
        return key


IDENTITY_SUBSCRIPT = _IdentitySubscript()
SUBSCRIPT_MARKER = "__SUBSCRIPT_SENTINEL__"


class GuardRejection(Exception):
    """Exception raised when guard rejects evaluation attempt."""

    pass


def guarded_eval(code: str, context: EvaluationContext):
    """Evaluate provided code in the evaluation context.

    If evaluation policy given by context is set to ``forbidden``
    no evaluation will be performed; if it is set to ``dangerous``
    standard :func:`eval` will be used; finally, for any other,
    policy :func:`eval_node` will be called on parsed AST.
    """
    locals_ = context.locals

    if context.evaluation == "forbidden":
        raise GuardRejection("Forbidden mode")

    # note: not using `ast.literal_eval` as it does not implement
    # getitem at all, for example it fails on simple `[0][1]`

    if context.in_subscript:
        # syntatic sugar for ellipsis (:) is only available in susbcripts
        # so we need to trick the ast parser into thinking that we have
        # a subscript, but we need to be able to later recognise that we did
        # it so we can ignore the actual __getitem__ operation
        if not code:
            return tuple()
        locals_ = locals_.copy()
        locals_[SUBSCRIPT_MARKER] = IDENTITY_SUBSCRIPT
        code = SUBSCRIPT_MARKER + "[" + code + "]"
        context = EvaluationContext(**{**context._asdict(), **{"locals": locals_}})

    if context.evaluation == "dangerous":
        return eval(code, context.globals, context.locals)

    expression = ast.parse(code, mode="eval")

    return eval_node(expression, context)


BINARY_OP_DUNDERS: Dict[Type[ast.operator], Tuple[str]] = {
    ast.Add: ("__add__",),
    ast.Sub: ("__sub__",),
    ast.Mult: ("__mul__",),
    ast.Div: ("__truediv__",),
    ast.FloorDiv: ("__floordiv__",),
    ast.Mod: ("__mod__",),
    ast.Pow: ("__pow__",),
    ast.LShift: ("__lshift__",),
    ast.RShift: ("__rshift__",),
    ast.BitOr: ("__or__",),
    ast.BitXor: ("__xor__",),
    ast.BitAnd: ("__and__",),
    ast.MatMult: ("__matmul__",),
}

COMP_OP_DUNDERS: Dict[Type[ast.cmpop], Tuple[str, ...]] = {
    ast.Eq: ("__eq__",),
    ast.NotEq: ("__ne__", "__eq__"),
    ast.Lt: ("__lt__", "__gt__"),
    ast.LtE: ("__le__", "__ge__"),
    ast.Gt: ("__gt__", "__lt__"),
    ast.GtE: ("__ge__", "__le__"),
    ast.In: ("__contains__",),
    # Note: ast.Is, ast.IsNot, ast.NotIn are handled specially
}

UNARY_OP_DUNDERS: Dict[Type[ast.unaryop], Tuple[str, ...]] = {
    ast.USub: ("__neg__",),
    ast.UAdd: ("__pos__",),
    # we have to check both __inv__ and __invert__!
    ast.Invert: ("__invert__", "__inv__"),
    ast.Not: ("__not__",),
}


def _find_dunder(node_op, dunders) -> Union[Tuple[str, ...], None]:
    dunder = None
    for op, candidate_dunder in dunders.items():
        if isinstance(node_op, op):
            dunder = candidate_dunder
    return dunder


def eval_node(node: Union[ast.AST, None], context: EvaluationContext):
    """Evaluate AST node in provided context.

    Applies evaluation restrictions defined in the context. Currently does not support evaluation of functions with keyword arguments.

    Does not evaluate actions that always have side effects:

    - class definitions (``class sth: ...``)
    - function definitions (``def sth: ...``)
    - variable assignments (``x = 1``)
    - augmented assignments (``x += 1``)
    - deletions (``del x``)

    Does not evaluate operations which do not return values:

    - assertions (``assert x``)
    - pass (``pass``)
    - imports (``import x``)
    - control flow:

        - conditionals (``if x:``) except for ternary IfExp (``a if x else b``)
        - loops (``for`` and `while``)
        - exception handling

    The purpose of this function is to guard against unwanted side-effects;
    it does not give guarantees on protection from malicious code execution.
    """
    policy = EVALUATION_POLICIES[context.evaluation]
    if node is None:
        return None
    if isinstance(node, ast.Expression):
        return eval_node(node.body, context)
    if isinstance(node, ast.BinOp):
        left = eval_node(node.left, context)
        right = eval_node(node.right, context)
        dunders = _find_dunder(node.op, BINARY_OP_DUNDERS)
        if dunders:
            if policy.can_operate(dunders, left, right):
                return getattr(left, dunders[0])(right)
            else:
                raise GuardRejection(
                    f"Operation (`{dunders}`) for",
                    type(left),
                    f"not allowed in {context.evaluation} mode",
                )
    if isinstance(node, ast.Compare):
        left = eval_node(node.left, context)
        all_true = True
        negate = False
        for op, right in zip(node.ops, node.comparators):
            right = eval_node(right, context)
            dunder = None
            dunders = _find_dunder(op, COMP_OP_DUNDERS)
            if not dunders:
                if isinstance(op, ast.NotIn):
                    dunders = COMP_OP_DUNDERS[ast.In]
                    negate = True
                if isinstance(op, ast.Is):
                    dunder = "is_"
                if isinstance(op, ast.IsNot):
                    dunder = "is_"
                    negate = True
            if not dunder and dunders:
                dunder = dunders[0]
            if dunder:
                a, b = (right, left) if dunder == "__contains__" else (left, right)
                if dunder == "is_" or dunders and policy.can_operate(dunders, a, b):
                    result = getattr(operator, dunder)(a, b)
                    if negate:
                        result = not result
                    if not result:
                        all_true = False
                    left = right
                else:
                    raise GuardRejection(
                        f"Comparison (`{dunder}`) for",
                        type(left),
                        f"not allowed in {context.evaluation} mode",
                    )
            else:
                raise ValueError(
                    f"Comparison `{dunder}` not supported"
                )  # pragma: no cover
        return all_true
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(eval_node(e, context) for e in node.elts)
    if isinstance(node, ast.List):
        return [eval_node(e, context) for e in node.elts]
    if isinstance(node, ast.Set):
        return {eval_node(e, context) for e in node.elts}
    if isinstance(node, ast.Dict):
        return dict(
            zip(
                [eval_node(k, context) for k in node.keys],
                [eval_node(v, context) for v in node.values],
            )
        )
    if isinstance(node, ast.Slice):
        return slice(
            eval_node(node.lower, context),
            eval_node(node.upper, context),
            eval_node(node.step, context),
        )
    if isinstance(node, ast.UnaryOp):
        value = eval_node(node.operand, context)
        dunders = _find_dunder(node.op, UNARY_OP_DUNDERS)
        if dunders:
            if policy.can_operate(dunders, value):
                return getattr(value, dunders[0])()
            else:
                raise GuardRejection(
                    f"Operation (`{dunders}`) for",
                    type(value),
                    f"not allowed in {context.evaluation} mode",
                )
    if isinstance(node, ast.Subscript):
        value = eval_node(node.value, context)
        slice_ = eval_node(node.slice, context)
        if policy.can_get_item(value, slice_):
            return value[slice_]
        raise GuardRejection(
            "Subscript access (`__getitem__`) for",
            type(value),  # not joined to avoid calling `repr`
            f" not allowed in {context.evaluation} mode",
        )
    if isinstance(node, ast.Name):
        if policy.allow_locals_access and node.id in context.locals:
            return context.locals[node.id]
        if policy.allow_globals_access and node.id in context.globals:
            return context.globals[node.id]
        if policy.allow_builtins_access and hasattr(builtins, node.id):
            # note: do not use __builtins__, it is implementation detail of cPython
            return getattr(builtins, node.id)
        if not policy.allow_globals_access and not policy.allow_locals_access:
            raise GuardRejection(
                f"Namespace access not allowed in {context.evaluation} mode"
            )
        else:
            raise NameError(f"{node.id} not found in locals, globals, nor builtins")
    if isinstance(node, ast.Attribute):
        value = eval_node(node.value, context)
        if policy.can_get_attr(value, node.attr):
            return getattr(value, node.attr)
        raise GuardRejection(
            "Attribute access (`__getattr__`) for",
            type(value),  # not joined to avoid calling `repr`
            f"not allowed in {context.evaluation} mode",
        )
    if isinstance(node, ast.IfExp):
        test = eval_node(node.test, context)
        if test:
            return eval_node(node.body, context)
        else:
            return eval_node(node.orelse, context)
    if isinstance(node, ast.Call):
        func = eval_node(node.func, context)
        if policy.can_call(func) and not node.keywords:
            args = [eval_node(arg, context) for arg in node.args]
            return func(*args)
        raise GuardRejection(
            "Call for",
            func,  # not joined to avoid calling `repr`
            f"not allowed in {context.evaluation} mode",
        )
    raise ValueError("Unhandled node", ast.dump(node))


SUPPORTED_EXTERNAL_GETITEM = {
    ("pandas", "core", "indexing", "_iLocIndexer"),
    ("pandas", "core", "indexing", "_LocIndexer"),
    ("pandas", "DataFrame"),
    ("pandas", "Series"),
    ("numpy", "ndarray"),
    ("numpy", "void"),
}


BUILTIN_GETITEM: Set[InstancesHaveGetItem] = {
    dict,
    str,  # type: ignore[arg-type]
    bytes,  # type: ignore[arg-type]
    list,
    tuple,
    collections.defaultdict,
    collections.deque,
    collections.OrderedDict,
    collections.ChainMap,
    collections.UserDict,
    collections.UserList,
    collections.UserString,  # type: ignore[arg-type]
    _DummyNamedTuple,
    _IdentitySubscript,
}


def _list_methods(cls, source=None):
    """For use on immutable objects or with methods returning a copy"""
    return [getattr(cls, k) for k in (source if source else dir(cls))]


dict_non_mutating_methods = ("copy", "keys", "values", "items")
list_non_mutating_methods = ("copy", "index", "count")
set_non_mutating_methods = set(dir(set)) & set(dir(frozenset))


dict_keys: Type[collections.abc.KeysView] = type({}.keys())

NUMERICS = {int, float, complex}

ALLOWED_CALLS = {
    bytes,
    *_list_methods(bytes),
    dict,
    *_list_methods(dict, dict_non_mutating_methods),
    dict_keys.isdisjoint,
    list,
    *_list_methods(list, list_non_mutating_methods),
    set,
    *_list_methods(set, set_non_mutating_methods),
    frozenset,
    *_list_methods(frozenset),
    range,
    str,
    *_list_methods(str),
    tuple,
    *_list_methods(tuple),
    *NUMERICS,
    *[method for numeric_cls in NUMERICS for method in _list_methods(numeric_cls)],
    collections.deque,
    *_list_methods(collections.deque, list_non_mutating_methods),
    collections.defaultdict,
    *_list_methods(collections.defaultdict, dict_non_mutating_methods),
    collections.OrderedDict,
    *_list_methods(collections.OrderedDict, dict_non_mutating_methods),
    collections.UserDict,
    *_list_methods(collections.UserDict, dict_non_mutating_methods),
    collections.UserList,
    *_list_methods(collections.UserList, list_non_mutating_methods),
    collections.UserString,
    *_list_methods(collections.UserString, dir(str)),
    collections.Counter,
    *_list_methods(collections.Counter, dict_non_mutating_methods),
    collections.Counter.elements,
    collections.Counter.most_common,
}

BUILTIN_GETATTR: Set[MayHaveGetattr] = {
    *BUILTIN_GETITEM,
    set,
    frozenset,
    object,
    type,  # `type` handles a lot of generic cases, e.g. numbers as in `int.real`.
    *NUMERICS,
    dict_keys,
    MethodDescriptorType,
    ModuleType,
}


BUILTIN_OPERATIONS = {*BUILTIN_GETATTR}

EVALUATION_POLICIES = {
    "minimal": EvaluationPolicy(
        allow_builtins_access=True,
        allow_locals_access=False,
        allow_globals_access=False,
        allow_item_access=False,
        allow_attr_access=False,
        allowed_calls=set(),
        allow_any_calls=False,
        allow_all_operations=False,
    ),
    "limited": SelectivePolicy(
        allowed_getitem=BUILTIN_GETITEM,
        allowed_getitem_external=SUPPORTED_EXTERNAL_GETITEM,
        allowed_getattr=BUILTIN_GETATTR,
        allowed_getattr_external={
            # pandas Series/Frame implements custom `__getattr__`
            ("pandas", "DataFrame"),
            ("pandas", "Series"),
        },
        allowed_operations=BUILTIN_OPERATIONS,
        allow_builtins_access=True,
        allow_locals_access=True,
        allow_globals_access=True,
        allowed_calls=ALLOWED_CALLS,
    ),
    "unsafe": EvaluationPolicy(
        allow_builtins_access=True,
        allow_locals_access=True,
        allow_globals_access=True,
        allow_attr_access=True,
        allow_item_access=True,
        allow_any_calls=True,
        allow_all_operations=True,
    ),
}


__all__ = [
    "guarded_eval",
    "eval_node",
    "GuardRejection",
    "EvaluationContext",
    "_unbind_method",
]
