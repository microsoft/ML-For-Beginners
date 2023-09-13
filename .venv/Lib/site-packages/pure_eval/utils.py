from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing


class CannotEval(Exception):
    def __repr__(self):
        return self.__class__.__name__

    __str__ = __repr__


def is_any(x, *args):
    return any(
        x is arg
        for arg in args
    )


def of_type(x, *types):
    if is_any(type(x), *types):
        return x
    else:
        raise CannotEval


def of_standard_types(x, *, check_dict_values: bool, deep: bool):
    if is_standard_types(x, check_dict_values=check_dict_values, deep=deep):
        return x
    else:
        raise CannotEval


def is_standard_types(x, *, check_dict_values: bool, deep: bool):
    try:
        return _is_standard_types_deep(x, check_dict_values, deep)[0]
    except RecursionError:
        return False


def _is_standard_types_deep(x, check_dict_values: bool, deep: bool):
    typ = type(x)
    if is_any(
        typ,
        str,
        int,
        bool,
        float,
        bytes,
        complex,
        date,
        time,
        datetime,
        Fraction,
        Decimal,
        type(None),
        object,
    ):
        return True, 0

    if is_any(typ, tuple, frozenset, list, set, dict, OrderedDict, deque, slice):
        if typ in [slice]:
            length = 0
        else:
            length = len(x)
        assert isinstance(deep, bool)
        if not deep:
            return True, length

        if check_dict_values and typ in (dict, OrderedDict):
            items = (v for pair in x.items() for v in pair)
        elif typ is slice:
            items = [x.start, x.stop, x.step]
        else:
            items = x
        for item in items:
            if length > 100000:
                return False, length
            is_standard, item_length = _is_standard_types_deep(
                item, check_dict_values, deep
            )
            if not is_standard:
                return False, length
            length += item_length
        return True, length

    return False, 0


class _E(enum.Enum):
    pass


class _C:
    def foo(self): pass  # pragma: nocover

    def bar(self): pass  # pragma: nocover

    @classmethod
    def cm(cls): pass  # pragma: nocover

    @staticmethod
    def sm(): pass  # pragma: nocover


safe_name_samples = {
    "len": len,
    "append": list.append,
    "__add__": list.__add__,
    "insert": [].insert,
    "__mul__": [].__mul__,
    "fromkeys": dict.__dict__['fromkeys'],
    "is_any": is_any,
    "__repr__": CannotEval.__repr__,
    "foo": _C().foo,
    "bar": _C.bar,
    "cm": _C.cm,
    "sm": _C.sm,
    "ast": ast,
    "CannotEval": CannotEval,
    "_E": _E,
}

typing_annotation_samples = {
    name: getattr(typing, name)
    for name in "List Dict Tuple Set Callable Mapping".split()
}

safe_name_types = tuple({
    type(f)
    for f in safe_name_samples.values()
})


typing_annotation_types = tuple({
    type(f)
    for f in typing_annotation_samples.values()
})


def eq_checking_types(a, b):
    return type(a) is type(b) and a == b


def ast_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    else:
        return None


def safe_name(value):
    typ = type(value)
    if is_any(typ, *safe_name_types):
        return value.__name__
    elif value is typing.Optional:
        return "Optional"
    elif value is typing.Union:
        return "Union"
    elif is_any(typ, *typing_annotation_types):
        return getattr(value, "__name__", None) or getattr(value, "_name", None)
    else:
        return None


def has_ast_name(value, node):
    value_name = safe_name(value)
    if type(value_name) is not str:
        return False
    return eq_checking_types(ast_name(node), value_name)


def copy_ast_without_context(x):
    if isinstance(x, ast.AST):
        kwargs = {
            field: copy_ast_without_context(getattr(x, field))
            for field in x._fields
            if field != 'ctx'
            if hasattr(x, field)
        }
        return type(x)(**kwargs)
    elif isinstance(x, list):
        return list(map(copy_ast_without_context, x))
    else:
        return x


def ensure_dict(x):
    """
    Handles invalid non-dict inputs
    """
    try:
        return dict(x)
    except Exception:
        return {}
