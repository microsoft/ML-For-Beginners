from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
    EvaluationContext,
    GuardRejection,
    guarded_eval,
    _unbind_method,
)
from IPython.testing import decorators as dec
import pytest


def create_context(evaluation: str, **kwargs):
    return EvaluationContext(locals=kwargs, globals={}, evaluation=evaluation)


forbidden = partial(create_context, "forbidden")
minimal = partial(create_context, "minimal")
limited = partial(create_context, "limited")
unsafe = partial(create_context, "unsafe")
dangerous = partial(create_context, "dangerous")

LIMITED_OR_HIGHER = [limited, unsafe, dangerous]
MINIMAL_OR_HIGHER = [minimal, *LIMITED_OR_HIGHER]


@contextmanager
def module_not_installed(module: str):
    import sys

    try:
        to_restore = sys.modules[module]
        del sys.modules[module]
    except KeyError:
        to_restore = None
    try:
        yield
    finally:
        sys.modules[module] = to_restore


def test_external_not_installed():
    """
    Because attribute check requires checking if object is not of allowed
    external type, this tests logic for absence of external module.
    """

    class Custom:
        def __init__(self):
            self.test = 1

        def __getattr__(self, key):
            return key

    with module_not_installed("pandas"):
        context = limited(x=Custom())
        with pytest.raises(GuardRejection):
            guarded_eval("x.test", context)


@dec.skip_without("pandas")
def test_external_changed_api(monkeypatch):
    """Check that the execution rejects if external API changed paths"""
    import pandas as pd

    series = pd.Series([1], index=["a"])

    with monkeypatch.context() as m:
        m.delattr(pd, "Series")
        context = limited(data=series)
        with pytest.raises(GuardRejection):
            guarded_eval("data.iloc[0]", context)


@dec.skip_without("pandas")
def test_pandas_series_iloc():
    import pandas as pd

    series = pd.Series([1], index=["a"])
    context = limited(data=series)
    assert guarded_eval("data.iloc[0]", context) == 1


def test_rejects_custom_properties():
    class BadProperty:
        @property
        def iloc(self):
            return [None]

    series = BadProperty()
    context = limited(data=series)

    with pytest.raises(GuardRejection):
        guarded_eval("data.iloc[0]", context)


@dec.skip_without("pandas")
def test_accepts_non_overriden_properties():
    import pandas as pd

    class GoodProperty(pd.Series):
        pass

    series = GoodProperty([1], index=["a"])
    context = limited(data=series)

    assert guarded_eval("data.iloc[0]", context) == 1


@dec.skip_without("pandas")
def test_pandas_series():
    import pandas as pd

    context = limited(data=pd.Series([1], index=["a"]))
    assert guarded_eval('data["a"]', context) == 1
    with pytest.raises(KeyError):
        guarded_eval('data["c"]', context)


@dec.skip_without("pandas")
def test_pandas_bad_series():
    import pandas as pd

    class BadItemSeries(pd.Series):
        def __getitem__(self, key):
            return "CUSTOM_ITEM"

    class BadAttrSeries(pd.Series):
        def __getattr__(self, key):
            return "CUSTOM_ATTR"

    bad_series = BadItemSeries([1], index=["a"])
    context = limited(data=bad_series)

    with pytest.raises(GuardRejection):
        guarded_eval('data["a"]', context)
    with pytest.raises(GuardRejection):
        guarded_eval('data["c"]', context)

    # note: here result is a bit unexpected because
    # pandas `__getattr__` calls `__getitem__`;
    # FIXME - special case to handle it?
    assert guarded_eval("data.a", context) == "CUSTOM_ITEM"

    context = unsafe(data=bad_series)
    assert guarded_eval('data["a"]', context) == "CUSTOM_ITEM"

    bad_attr_series = BadAttrSeries([1], index=["a"])
    context = limited(data=bad_attr_series)
    assert guarded_eval('data["a"]', context) == 1
    with pytest.raises(GuardRejection):
        guarded_eval("data.a", context)


@dec.skip_without("pandas")
def test_pandas_dataframe_loc():
    import pandas as pd
    from pandas.testing import assert_series_equal

    data = pd.DataFrame([{"a": 1}])
    context = limited(data=data)
    assert_series_equal(guarded_eval('data.loc[:, "a"]', context), data["a"])


def test_named_tuple():
    class GoodNamedTuple(NamedTuple):
        a: str
        pass

    class BadNamedTuple(NamedTuple):
        a: str

        def __getitem__(self, key):
            return None

    good = GoodNamedTuple(a="x")
    bad = BadNamedTuple(a="x")

    context = limited(data=good)
    assert guarded_eval("data[0]", context) == "x"

    context = limited(data=bad)
    with pytest.raises(GuardRejection):
        guarded_eval("data[0]", context)


def test_dict():
    context = limited(data={"a": 1, "b": {"x": 2}, ("x", "y"): 3})
    assert guarded_eval('data["a"]', context) == 1
    assert guarded_eval('data["b"]', context) == {"x": 2}
    assert guarded_eval('data["b"]["x"]', context) == 2
    assert guarded_eval('data["x", "y"]', context) == 3

    assert guarded_eval("data.keys", context)


def test_set():
    context = limited(data={"a", "b"})
    assert guarded_eval("data.difference", context)


def test_list():
    context = limited(data=[1, 2, 3])
    assert guarded_eval("data[1]", context) == 2
    assert guarded_eval("data.copy", context)


def test_dict_literal():
    context = limited()
    assert guarded_eval("{}", context) == {}
    assert guarded_eval('{"a": 1}', context) == {"a": 1}


def test_list_literal():
    context = limited()
    assert guarded_eval("[]", context) == []
    assert guarded_eval('[1, "a"]', context) == [1, "a"]


def test_set_literal():
    context = limited()
    assert guarded_eval("set()", context) == set()
    assert guarded_eval('{"a"}', context) == {"a"}


def test_evaluates_if_expression():
    context = limited()
    assert guarded_eval("2 if True else 3", context) == 2
    assert guarded_eval("4 if False else 5", context) == 5


def test_object():
    obj = object()
    context = limited(obj=obj)
    assert guarded_eval("obj.__dir__", context) == obj.__dir__


@pytest.mark.parametrize(
    "code,expected",
    [
        ["int.numerator", int.numerator],
        ["float.is_integer", float.is_integer],
        ["complex.real", complex.real],
    ],
)
def test_number_attributes(code, expected):
    assert guarded_eval(code, limited()) == expected


def test_method_descriptor():
    context = limited()
    assert guarded_eval("list.copy.__name__", context) == "copy"


@pytest.mark.parametrize(
    "data,good,bad,expected",
    [
        [[1, 2, 3], "data.index(2)", "data.append(4)", 1],
        [{"a": 1}, "data.keys().isdisjoint({})", "data.update()", True],
    ],
)
def test_evaluates_calls(data, good, bad, expected):
    context = limited(data=data)
    assert guarded_eval(good, context) == expected

    with pytest.raises(GuardRejection):
        guarded_eval(bad, context)


@pytest.mark.parametrize(
    "code,expected",
    [
        ["(1\n+\n1)", 2],
        ["list(range(10))[-1:]", [9]],
        ["list(range(20))[3:-2:3]", [3, 6, 9, 12, 15]],
    ],
)
@pytest.mark.parametrize("context", LIMITED_OR_HIGHER)
def test_evaluates_complex_cases(code, expected, context):
    assert guarded_eval(code, context()) == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ["1", 1],
        ["1.0", 1.0],
        ["0xdeedbeef", 0xDEEDBEEF],
        ["True", True],
        ["None", None],
        ["{}", {}],
        ["[]", []],
    ],
)
@pytest.mark.parametrize("context", MINIMAL_OR_HIGHER)
def test_evaluates_literals(code, expected, context):
    assert guarded_eval(code, context()) == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ["-5", -5],
        ["+5", +5],
        ["~5", -6],
    ],
)
@pytest.mark.parametrize("context", LIMITED_OR_HIGHER)
def test_evaluates_unary_operations(code, expected, context):
    assert guarded_eval(code, context()) == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ["1 + 1", 2],
        ["3 - 1", 2],
        ["2 * 3", 6],
        ["5 // 2", 2],
        ["5 / 2", 2.5],
        ["5**2", 25],
        ["2 >> 1", 1],
        ["2 << 1", 4],
        ["1 | 2", 3],
        ["1 & 1", 1],
        ["1 & 2", 0],
    ],
)
@pytest.mark.parametrize("context", LIMITED_OR_HIGHER)
def test_evaluates_binary_operations(code, expected, context):
    assert guarded_eval(code, context()) == expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ["2 > 1", True],
        ["2 < 1", False],
        ["2 <= 1", False],
        ["2 <= 2", True],
        ["1 >= 2", False],
        ["2 >= 2", True],
        ["2 == 2", True],
        ["1 == 2", False],
        ["1 != 2", True],
        ["1 != 1", False],
        ["1 < 4 < 3", False],
        ["(1 < 4) < 3", True],
        ["4 > 3 > 2 > 1", True],
        ["4 > 3 > 2 > 9", False],
        ["1 < 2 < 3 < 4", True],
        ["9 < 2 < 3 < 4", False],
        ["1 < 2 > 1 > 0 > -1 < 1", True],
        ["1 in [1] in [[1]]", True],
        ["1 in [1] in [[2]]", False],
        ["1 in [1]", True],
        ["0 in [1]", False],
        ["1 not in [1]", False],
        ["0 not in [1]", True],
        ["True is True", True],
        ["False is False", True],
        ["True is False", False],
        ["True is not True", False],
        ["False is not True", True],
    ],
)
@pytest.mark.parametrize("context", LIMITED_OR_HIGHER)
def test_evaluates_comparisons(code, expected, context):
    assert guarded_eval(code, context()) == expected


def test_guards_comparisons():
    class GoodEq(int):
        pass

    class BadEq(int):
        def __eq__(self, other):
            assert False

    context = limited(bad=BadEq(1), good=GoodEq(1))

    with pytest.raises(GuardRejection):
        guarded_eval("bad == 1", context)

    with pytest.raises(GuardRejection):
        guarded_eval("bad != 1", context)

    with pytest.raises(GuardRejection):
        guarded_eval("1 == bad", context)

    with pytest.raises(GuardRejection):
        guarded_eval("1 != bad", context)

    assert guarded_eval("good == 1", context) is True
    assert guarded_eval("good != 1", context) is False
    assert guarded_eval("1 == good", context) is True
    assert guarded_eval("1 != good", context) is False


def test_guards_unary_operations():
    class GoodOp(int):
        pass

    class BadOpInv(int):
        def __inv__(self, other):
            assert False

    class BadOpInverse(int):
        def __inv__(self, other):
            assert False

    context = limited(good=GoodOp(1), bad1=BadOpInv(1), bad2=BadOpInverse(1))

    with pytest.raises(GuardRejection):
        guarded_eval("~bad1", context)

    with pytest.raises(GuardRejection):
        guarded_eval("~bad2", context)


def test_guards_binary_operations():
    class GoodOp(int):
        pass

    class BadOp(int):
        def __add__(self, other):
            assert False

    context = limited(good=GoodOp(1), bad=BadOp(1))

    with pytest.raises(GuardRejection):
        guarded_eval("1 + bad", context)

    with pytest.raises(GuardRejection):
        guarded_eval("bad + 1", context)

    assert guarded_eval("good + 1", context) == 2
    assert guarded_eval("1 + good", context) == 2


def test_guards_attributes():
    class GoodAttr(float):
        pass

    class BadAttr1(float):
        def __getattr__(self, key):
            assert False

    class BadAttr2(float):
        def __getattribute__(self, key):
            assert False

    context = limited(good=GoodAttr(0.5), bad1=BadAttr1(0.5), bad2=BadAttr2(0.5))

    with pytest.raises(GuardRejection):
        guarded_eval("bad1.as_integer_ratio", context)

    with pytest.raises(GuardRejection):
        guarded_eval("bad2.as_integer_ratio", context)

    assert guarded_eval("good.as_integer_ratio()", context) == (1, 2)


@pytest.mark.parametrize("context", MINIMAL_OR_HIGHER)
def test_access_builtins(context):
    assert guarded_eval("round", context()) == round


def test_access_builtins_fails():
    context = limited()
    with pytest.raises(NameError):
        guarded_eval("this_is_not_builtin", context)


def test_rejects_forbidden():
    context = forbidden()
    with pytest.raises(GuardRejection):
        guarded_eval("1", context)


def test_guards_locals_and_globals():
    context = EvaluationContext(
        locals={"local_a": "a"}, globals={"global_b": "b"}, evaluation="minimal"
    )

    with pytest.raises(GuardRejection):
        guarded_eval("local_a", context)

    with pytest.raises(GuardRejection):
        guarded_eval("global_b", context)


def test_access_locals_and_globals():
    context = EvaluationContext(
        locals={"local_a": "a"}, globals={"global_b": "b"}, evaluation="limited"
    )
    assert guarded_eval("local_a", context) == "a"
    assert guarded_eval("global_b", context) == "b"


@pytest.mark.parametrize(
    "code",
    ["def func(): pass", "class C: pass", "x = 1", "x += 1", "del x", "import ast"],
)
@pytest.mark.parametrize("context", [minimal(), limited(), unsafe()])
def test_rejects_side_effect_syntax(code, context):
    with pytest.raises(SyntaxError):
        guarded_eval(code, context)


def test_subscript():
    context = EvaluationContext(
        locals={}, globals={}, evaluation="limited", in_subscript=True
    )
    empty_slice = slice(None, None, None)
    assert guarded_eval("", context) == tuple()
    assert guarded_eval(":", context) == empty_slice
    assert guarded_eval("1:2:3", context) == slice(1, 2, 3)
    assert guarded_eval(':, "a"', context) == (empty_slice, "a")


def test_unbind_method():
    class X(list):
        def index(self, k):
            return "CUSTOM"

    x = X()
    assert _unbind_method(x.index) is X.index
    assert _unbind_method([].index) is list.index
    assert _unbind_method(list.index) is None


def test_assumption_instance_attr_do_not_matter():
    """This is semi-specified in Python documentation.

    However, since the specification says 'not guaranted
    to work' rather than 'is forbidden to work', future
    versions could invalidate this assumptions. This test
    is meant to catch such a change if it ever comes true.
    """

    class T:
        def __getitem__(self, k):
            return "a"

        def __getattr__(self, k):
            return "a"

    def f(self):
        return "b"

    t = T()
    t.__getitem__ = f
    t.__getattr__ = f
    assert t[1] == "a"
    assert t[1] == "a"


def test_assumption_named_tuples_share_getitem():
    """Check assumption on named tuples sharing __getitem__"""
    from typing import NamedTuple

    class A(NamedTuple):
        pass

    class B(NamedTuple):
        pass

    assert A.__getitem__ == B.__getitem__


@dec.skip_without("numpy")
def test_module_access():
    import numpy

    context = limited(numpy=numpy)
    assert guarded_eval("numpy.linalg.norm", context) == numpy.linalg.norm

    context = minimal(numpy=numpy)
    with pytest.raises(GuardRejection):
        guarded_eval("np.linalg.norm", context)
