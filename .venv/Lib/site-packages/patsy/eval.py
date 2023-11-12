# This file is part of Patsy
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Utilities that require an over-intimate knowledge of Python's execution
# environment.

# NB: if you add any __future__ imports to this file then you'll have to
# adjust the tests that deal with checking the caller's execution environment
# for __future__ flags!

# These are made available in the patsy.* namespace
__all__ = ["EvalEnvironment", "EvalFactor"]

import sys
import __future__
import inspect
import tokenize
import ast
import numbers
import six
from patsy import PatsyError
from patsy.util import PushbackAdapter, no_pickling, assert_no_pickling
from patsy.tokens import (pretty_untokenize, normalize_token_spacing,
                             python_tokenize)
from patsy.compat import call_and_wrap_exc

def _all_future_flags():
    flags = 0
    for feature_name in __future__.all_feature_names:
        feature = getattr(__future__, feature_name)
        mr = feature.getMandatoryRelease()
        # None means a planned feature was dropped, or at least postponed
        # without a final decision; see, for example,
        # https://docs.python.org/3.11/library/__future__.html#id2.
        if mr is None or mr > sys.version_info:
            flags |= feature.compiler_flag
    return flags

_ALL_FUTURE_FLAGS = _all_future_flags()

# This is just a minimal dict-like object that does lookup in a 'stack' of
# dicts -- first it checks the first, then the second, etc. Assignments go
# into an internal, zeroth dict.
class VarLookupDict(object):
    def __init__(self, dicts):
        self._dicts = [{}] + list(dicts)

    def __getitem__(self, key):
        for d in self._dicts:
            try:
                return d[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        self._dicts[0][key] = value

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._dicts)

    __getstate__ = no_pickling


def test_VarLookupDict():
    d1 = {"a": 1}
    d2 = {"a": 2, "b": 3}
    ds = VarLookupDict([d1, d2])
    assert ds["a"] == 1
    assert ds["b"] == 3
    assert "a" in ds
    assert "c" not in ds
    import pytest
    pytest.raises(KeyError, ds.__getitem__, "c")
    ds["a"] = 10
    assert ds["a"] == 10
    assert d1["a"] == 1
    assert ds.get("c") is None
    assert isinstance(repr(ds), six.string_types)

    assert_no_pickling(ds)

def ast_names(code):
    """Iterator that yields all the (ast) names in a Python expression.

    :arg code: A string containing a Python expression.
    """
    # Syntax that allows new name bindings to be introduced is tricky to
    # handle here, so we just refuse to do so.
    disallowed_ast_nodes = (ast.Lambda, ast.ListComp, ast.GeneratorExp)
    if sys.version_info >= (2, 7):
        disallowed_ast_nodes += (ast.DictComp, ast.SetComp)

    for node in ast.walk(ast.parse(code)):
        if isinstance(node, disallowed_ast_nodes):
            raise PatsyError("Lambda, list/dict/set comprehension, generator "
                             "expression in patsy formula not currently supported.")
        if isinstance(node, ast.Name):
            yield node.id

def test_ast_names():
    test_data = [('np.log(x)', ['np', 'x']),
                 ('x', ['x']),
                 ('center(x + 1)', ['center', 'x']),
                 ('dt.date.dt.month', ['dt'])]
    for code, expected in test_data:
        assert set(ast_names(code)) == set(expected)

def test_ast_names_disallowed_nodes():
    import pytest
    def list_ast_names(code):
        return list(ast_names(code))
    pytest.raises(PatsyError, list_ast_names, "lambda x: x + y")
    pytest.raises(PatsyError, list_ast_names, "[x + 1 for x in range(10)]")
    pytest.raises(PatsyError, list_ast_names, "(x + 1 for x in range(10))")
    if sys.version_info >= (2, 7):
        pytest.raises(PatsyError, list_ast_names, "{x: True for x in range(10)}")
        pytest.raises(PatsyError, list_ast_names, "{x + 1 for x in range(10)}")

class EvalEnvironment(object):
    """Represents a Python execution environment.

    Encapsulates a namespace for variable lookup and set of __future__
    flags."""
    def __init__(self, namespaces, flags=0):
        assert not flags & ~_ALL_FUTURE_FLAGS
        self._namespaces = list(namespaces)
        self.flags = flags

    @property
    def namespace(self):
        """A dict-like object that can be used to look up variables accessible
        from the encapsulated environment."""
        return VarLookupDict(self._namespaces)

    def with_outer_namespace(self, outer_namespace):
        """Return a new EvalEnvironment with an extra namespace added.

        This namespace will be used only for variables that are not found in
        any existing namespace, i.e., it is "outside" them all."""
        return self.__class__(self._namespaces + [outer_namespace],
                              self.flags)

    def eval(self, expr, source_name="<string>", inner_namespace={}):
        """Evaluate some Python code in the encapsulated environment.

        :arg expr: A string containing a Python expression.
        :arg source_name: A name for this string, for use in tracebacks.
        :arg inner_namespace: A dict-like object that will be checked first
          when `expr` attempts to access any variables.
        :returns: The value of `expr`.
        """
        code = compile(expr, source_name, "eval", self.flags, False)
        return eval(code, {}, VarLookupDict([inner_namespace]
                                            + self._namespaces))

    @classmethod
    def capture(cls, eval_env=0, reference=0):
        """Capture an execution environment from the stack.

        If `eval_env` is already an :class:`EvalEnvironment`, it is returned
        unchanged. Otherwise, we walk up the stack by ``eval_env + reference``
        steps and capture that function's evaluation environment.

        For ``eval_env=0`` and ``reference=0``, the default, this captures the
        stack frame of the function that calls :meth:`capture`. If ``eval_env
        + reference`` is 1, then we capture that function's caller, etc.

        This somewhat complicated calling convention is designed to be
        convenient for functions which want to capture their caller's
        environment by default, but also allow explicit environments to be
        specified. See the second example.

        Example::

          x = 1
          this_env = EvalEnvironment.capture()
          assert this_env.namespace["x"] == 1
          def child_func():
              return EvalEnvironment.capture(1)
          this_env_from_child = child_func()
          assert this_env_from_child.namespace["x"] == 1

        Example::

          # This function can be used like:
          #   my_model(formula_like, data)
          #     -> evaluates formula_like in caller's environment
          #   my_model(formula_like, data, eval_env=1)
          #     -> evaluates formula_like in caller's caller's environment
          #   my_model(formula_like, data, eval_env=my_env)
          #     -> evaluates formula_like in environment 'my_env'
          def my_model(formula_like, data, eval_env=0):
              eval_env = EvalEnvironment.capture(eval_env, reference=1)
              return model_setup_helper(formula_like, data, eval_env)

        This is how :func:`dmatrix` works.

        .. versionadded: 0.2.0
           The ``reference`` argument.
        """
        if isinstance(eval_env, cls):
            return eval_env
        elif isinstance(eval_env, numbers.Integral):
            depth = eval_env + reference
        else:
            raise TypeError("Parameter 'eval_env' must be either an integer "
                            "or an instance of patsy.EvalEnvironment.")
        frame = inspect.currentframe()
        try:
            for i in range(depth + 1):
                if frame is None:
                    raise ValueError("call-stack is not that deep!")
                frame = frame.f_back
            return cls([frame.f_locals, frame.f_globals],
                       frame.f_code.co_flags & _ALL_FUTURE_FLAGS)
        # The try/finally is important to avoid a potential reference cycle --
        # any exception traceback will carry a reference to *our* frame, which
        # contains a reference to our local variables, which would otherwise
        # carry a reference to some parent frame, where the exception was
        # caught...:
        finally:
            del frame

    def subset(self, names):
        """Creates a new, flat EvalEnvironment that contains only
        the variables specified."""
        vld = VarLookupDict(self._namespaces)
        new_ns = dict((name, vld[name]) for name in names)
        return EvalEnvironment([new_ns], self.flags)

    def _namespace_ids(self):
        return [id(n) for n in self._namespaces]

    def __eq__(self, other):
        return (isinstance(other, EvalEnvironment)
                and self.flags == other.flags
                and self._namespace_ids() == other._namespace_ids())

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((EvalEnvironment,
                     self.flags,
                     tuple(self._namespace_ids())))

    __getstate__ = no_pickling

def _a(): # pragma: no cover
    _a = 1
    return _b()

def _b(): # pragma: no cover
    _b = 1
    return _c()

def _c(): # pragma: no cover
    _c = 1
    return [EvalEnvironment.capture(),
            EvalEnvironment.capture(0),
            EvalEnvironment.capture(1),
            EvalEnvironment.capture(0, reference=1),
            EvalEnvironment.capture(2),
            EvalEnvironment.capture(0, 2),
            ]

def test_EvalEnvironment_capture_namespace():
    c0, c, b1, b2, a1, a2 = _a()
    assert "test_EvalEnvironment_capture_namespace" in c0.namespace
    assert "test_EvalEnvironment_capture_namespace" in c.namespace
    assert "test_EvalEnvironment_capture_namespace" in b1.namespace
    assert "test_EvalEnvironment_capture_namespace" in b2.namespace
    assert "test_EvalEnvironment_capture_namespace" in a1.namespace
    assert "test_EvalEnvironment_capture_namespace" in a2.namespace
    assert c0.namespace["_c"] == 1
    assert c.namespace["_c"] == 1
    assert b1.namespace["_b"] == 1
    assert b2.namespace["_b"] == 1
    assert a1.namespace["_a"] == 1
    assert a2.namespace["_a"] == 1
    assert b1.namespace["_c"] is _c
    assert b2.namespace["_c"] is _c
    import pytest
    pytest.raises(ValueError, EvalEnvironment.capture, 10 ** 6)

    assert EvalEnvironment.capture(b1) is b1

    pytest.raises(TypeError, EvalEnvironment.capture, 1.2)

    assert_no_pickling(EvalEnvironment.capture())

def test_EvalEnvironment_capture_flags():
    if sys.version_info >= (3,):
        # This is the only __future__ feature currently usable in Python
        # 3... fortunately it is probably not going anywhere.
        TEST_FEATURE = "barry_as_FLUFL"
    else:
        TEST_FEATURE = "division"
    test_flag = getattr(__future__, TEST_FEATURE).compiler_flag
    assert test_flag & _ALL_FUTURE_FLAGS
    source = ("def f():\n"
              "    in_f = 'hi from f'\n"
              "    global RETURN_INNER, RETURN_OUTER, RETURN_INNER_FROM_OUTER\n"
              "    RETURN_INNER = EvalEnvironment.capture(0)\n"
              "    RETURN_OUTER = call_capture_0()\n"
              "    RETURN_INNER_FROM_OUTER = call_capture_1()\n"
              "f()\n")
    code = compile(source, "<test string>", "exec", 0, 1)
    env = {"EvalEnvironment": EvalEnvironment,
           "call_capture_0": lambda: EvalEnvironment.capture(0),
           "call_capture_1": lambda: EvalEnvironment.capture(1),
           }
    env2 = dict(env)
    six.exec_(code, env)
    assert env["RETURN_INNER"].namespace["in_f"] == "hi from f"
    assert env["RETURN_INNER_FROM_OUTER"].namespace["in_f"] == "hi from f"
    assert "in_f" not in env["RETURN_OUTER"].namespace
    assert env["RETURN_INNER"].flags & _ALL_FUTURE_FLAGS == 0
    assert env["RETURN_OUTER"].flags & _ALL_FUTURE_FLAGS == 0
    assert env["RETURN_INNER_FROM_OUTER"].flags & _ALL_FUTURE_FLAGS == 0

    code2 = compile(("from __future__ import %s\n" % (TEST_FEATURE,))
                    + source,
                    "<test string 2>", "exec", 0, 1)
    six.exec_(code2, env2)
    assert env2["RETURN_INNER"].namespace["in_f"] == "hi from f"
    assert env2["RETURN_INNER_FROM_OUTER"].namespace["in_f"] == "hi from f"
    assert "in_f" not in env2["RETURN_OUTER"].namespace
    assert env2["RETURN_INNER"].flags & _ALL_FUTURE_FLAGS == test_flag
    assert env2["RETURN_OUTER"].flags & _ALL_FUTURE_FLAGS == 0
    assert env2["RETURN_INNER_FROM_OUTER"].flags & _ALL_FUTURE_FLAGS == test_flag

def test_EvalEnvironment_eval_namespace():
    env = EvalEnvironment([{"a": 1}])
    assert env.eval("2 * a") == 2
    assert env.eval("2 * a", inner_namespace={"a": 2}) == 4
    import pytest
    pytest.raises(NameError, env.eval, "2 * b")
    a = 3
    env2 = EvalEnvironment.capture(0)
    assert env2.eval("2 * a") == 6

    env3 = env.with_outer_namespace({"a": 10, "b": 3})
    assert env3.eval("2 * a") == 2
    assert env3.eval("2 * b") == 6

def test_EvalEnvironment_eval_flags():
    import pytest
    if sys.version_info >= (3,):
        # This joke __future__ statement replaces "!=" with "<>":
        #   http://www.python.org/dev/peps/pep-0401/
        test_flag = __future__.barry_as_FLUFL.compiler_flag
        assert test_flag & _ALL_FUTURE_FLAGS

        env = EvalEnvironment([{"a": 11}], flags=0)
        assert env.eval("a != 0") == True
        pytest.raises(SyntaxError, env.eval, "a <> 0")
        assert env.subset(["a"]).flags == 0
        assert env.with_outer_namespace({"b": 10}).flags == 0

        env2 = EvalEnvironment([{"a": 11}], flags=test_flag)
        assert env2.eval("a <> 0") == True
        pytest.raises(SyntaxError, env2.eval, "a != 0")
        assert env2.subset(["a"]).flags == test_flag
        assert env2.with_outer_namespace({"b": 10}).flags == test_flag
    else:
        test_flag = __future__.division.compiler_flag
        assert test_flag & _ALL_FUTURE_FLAGS

        env = EvalEnvironment([{"a": 11}], flags=0)
        assert env.eval("a / 2") == 11 // 2 == 5
        assert env.subset(["a"]).flags == 0
        assert env.with_outer_namespace({"b": 10}).flags == 0

        env2 = EvalEnvironment([{"a": 11}], flags=test_flag)
        assert env2.eval("a / 2") == 11 * 1. / 2 != 5
        env2.subset(["a"]).flags == test_flag
        assert env2.with_outer_namespace({"b": 10}).flags == test_flag

def test_EvalEnvironment_subset():
    env = EvalEnvironment([{"a": 1}, {"b": 2}, {"c": 3}])

    subset_a = env.subset(["a"])
    assert subset_a.eval("a") == 1
    import pytest
    pytest.raises(NameError, subset_a.eval, "b")
    pytest.raises(NameError, subset_a.eval, "c")

    subset_bc = env.subset(["b", "c"])
    assert subset_bc.eval("b * c") == 6
    pytest.raises(NameError, subset_bc.eval, "a")

def test_EvalEnvironment_eq():
    # Two environments are eq only if they refer to exactly the same
    # global/local dicts
    env1 = EvalEnvironment.capture(0)
    env2 = EvalEnvironment.capture(0)
    assert env1 == env2
    assert hash(env1) == hash(env2)
    capture_local_env = lambda: EvalEnvironment.capture(0)
    env3 = capture_local_env()
    env4 = capture_local_env()
    assert env3 != env4

_builtins_dict = {}
six.exec_("from patsy.builtins import *", {}, _builtins_dict)
# This is purely to make the existence of patsy.builtins visible to systems
# like py2app and py2exe. It's basically free, since the above line guarantees
# that patsy.builtins will be present in sys.modules in any case.
import patsy.builtins

class EvalFactor(object):
    def __init__(self, code, origin=None):
        """A factor class that executes arbitrary Python code and supports
        stateful transforms.

        :arg code: A string containing a Python expression, that will be
          evaluated to produce this factor's value.

        This is the standard factor class that is used when parsing formula
        strings and implements the standard stateful transform processing. See
        :ref:`stateful-transforms` and :ref:`expert-model-specification`.

        Two EvalFactor's are considered equal (e.g., for purposes of
        redundancy detection) if they contain the same token stream. Basically
        this means that the source code must be identical except for
        whitespace::

          assert EvalFactor("a + b") == EvalFactor("a+b")
          assert EvalFactor("a + b") != EvalFactor("b + a")
        """

        # For parsed formulas, the code will already have been normalized by
        # the parser. But let's normalize anyway, so we can be sure of having
        # consistent semantics for __eq__ and __hash__.
        self.code = normalize_token_spacing(code)
        self.origin = origin

    def name(self):
        return self.code

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.code)

    def __eq__(self, other):
        return (isinstance(other, EvalFactor)
                and self.code == other.code)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((EvalFactor, self.code))

    def memorize_passes_needed(self, state, eval_env):
        # 'state' is just an empty dict which we can do whatever we want with,
        # and that will be passed back to later memorize functions
        state["transforms"] = {}

        eval_env = eval_env.with_outer_namespace(_builtins_dict)
        env_namespace = eval_env.namespace
        subset_names = [name for name in ast_names(self.code)
                        if name in env_namespace]
        eval_env = eval_env.subset(subset_names)
        state["eval_env"] = eval_env

        # example code: == "2 * center(x)"
        i = [0]
        def new_name_maker(token):
            value = eval_env.namespace.get(token)
            if hasattr(value, "__patsy_stateful_transform__"):
                obj_name = "_patsy_stobj%s__%s__" % (i[0], token)
                i[0] += 1
                obj = value.__patsy_stateful_transform__()
                state["transforms"][obj_name] = obj
                return obj_name + ".transform"
            else:
                return token
        # example eval_code: == "2 * _patsy_stobj0__center__.transform(x)"
        eval_code = replace_bare_funcalls(self.code, new_name_maker)
        state["eval_code"] = eval_code
        # paranoia: verify that none of our new names appeared anywhere in the
        # original code
        if has_bare_variable_reference(state["transforms"], self.code):
            raise PatsyError("names of this form are reserved for "
                                "internal use (%s)" % (token,), token.origin)
        # Pull out all the '_patsy_stobj0__center__.transform(x)' pieces
        # to make '_patsy_stobj0__center__.memorize_chunk(x)' pieces
        state["memorize_code"] = {}
        for obj_name in state["transforms"]:
            transform_calls = capture_obj_method_calls(obj_name, eval_code)
            assert len(transform_calls) == 1
            transform_call = transform_calls[0]
            transform_call_name, transform_call_code = transform_call
            assert transform_call_name == obj_name + ".transform"
            assert transform_call_code.startswith(transform_call_name + "(")
            memorize_code = (obj_name
                             + ".memorize_chunk"
                             + transform_call_code[len(transform_call_name):])
            state["memorize_code"][obj_name] = memorize_code
        # Then sort the codes into bins, so that every item in bin number i
        # depends only on items in bin (i-1) or less. (By 'depends', we mean
        # that in something like:
        #   spline(center(x))
        # we have to first run:
        #    center.memorize_chunk(x)
        # then
        #    center.memorize_finish(x)
        # and only then can we run:
        #    spline.memorize_chunk(center.transform(x))
        # Since all of our objects have unique names, figuring out who
        # depends on who is pretty easy -- we just check whether the
        # memorization code for spline:
        #    spline.memorize_chunk(center.transform(x))
        # mentions the variable 'center' (which in the example, of course, it
        # does).
        pass_bins = []
        unsorted = set(state["transforms"])
        while unsorted:
            pass_bin = set()
            for obj_name in unsorted:
                other_objs = unsorted.difference([obj_name])
                memorize_code = state["memorize_code"][obj_name]
                if not has_bare_variable_reference(other_objs, memorize_code):
                    pass_bin.add(obj_name)
            assert pass_bin
            unsorted.difference_update(pass_bin)
            pass_bins.append(pass_bin)
        state["pass_bins"] = pass_bins

        return len(pass_bins)

    def _eval(self, code, memorize_state, data):
        inner_namespace = VarLookupDict([data, memorize_state["transforms"]])
        return call_and_wrap_exc("Error evaluating factor",
                                 self,
                                 memorize_state["eval_env"].eval,
                                 code,
                                 inner_namespace=inner_namespace)

    def memorize_chunk(self, state, which_pass, data):
        for obj_name in state["pass_bins"][which_pass]:
            self._eval(state["memorize_code"][obj_name],
                       state,
                       data)

    def memorize_finish(self, state, which_pass):
        for obj_name in state["pass_bins"][which_pass]:
            state["transforms"][obj_name].memorize_finish()

    def eval(self, memorize_state, data):
        return self._eval(memorize_state["eval_code"],
                          memorize_state,
                          data)

    __getstate__ = no_pickling

def test_EvalFactor_basics():
    e = EvalFactor("a+b")
    assert e.code == "a + b"
    assert e.name() == "a + b"
    e2 = EvalFactor("a    +b", origin="asdf")
    assert e == e2
    assert hash(e) == hash(e2)
    assert e.origin is None
    assert e2.origin == "asdf"

    assert_no_pickling(e)

def test_EvalFactor_memorize_passes_needed():
    from patsy.state import stateful_transform
    foo = stateful_transform(lambda: "FOO-OBJ")
    bar = stateful_transform(lambda: "BAR-OBJ")
    quux = stateful_transform(lambda: "QUUX-OBJ")
    e = EvalFactor("foo(x) + bar(foo(y)) + quux(z, w)")

    state = {}
    eval_env = EvalEnvironment.capture(0)
    passes = e.memorize_passes_needed(state, eval_env)
    print(passes)
    print(state)
    assert passes == 2
    for name in ["foo", "bar", "quux"]:
        assert state["eval_env"].namespace[name] is locals()[name]
    for name in ["w", "x", "y", "z", "e", "state"]:
        assert name not in state["eval_env"].namespace
    assert state["transforms"] == {"_patsy_stobj0__foo__": "FOO-OBJ",
                                   "_patsy_stobj1__bar__": "BAR-OBJ",
                                   "_patsy_stobj2__foo__": "FOO-OBJ",
                                   "_patsy_stobj3__quux__": "QUUX-OBJ"}
    assert (state["eval_code"]
            == "_patsy_stobj0__foo__.transform(x)"
               " + _patsy_stobj1__bar__.transform("
               "_patsy_stobj2__foo__.transform(y))"
               " + _patsy_stobj3__quux__.transform(z, w)")

    assert (state["memorize_code"]
            == {"_patsy_stobj0__foo__":
                    "_patsy_stobj0__foo__.memorize_chunk(x)",
                "_patsy_stobj1__bar__":
                    "_patsy_stobj1__bar__.memorize_chunk(_patsy_stobj2__foo__.transform(y))",
                "_patsy_stobj2__foo__":
                    "_patsy_stobj2__foo__.memorize_chunk(y)",
                "_patsy_stobj3__quux__":
                    "_patsy_stobj3__quux__.memorize_chunk(z, w)",
                })
    assert state["pass_bins"] == [set(["_patsy_stobj0__foo__",
                                       "_patsy_stobj2__foo__",
                                       "_patsy_stobj3__quux__"]),
                                  set(["_patsy_stobj1__bar__"])]

class _MockTransform(object):
    # Adds up all memorized data, then subtracts that sum from each datum
    def __init__(self):
        self._sum = 0
        self._memorize_chunk_called = 0
        self._memorize_finish_called = 0

    def memorize_chunk(self, data):
        self._memorize_chunk_called += 1
        import numpy as np
        self._sum += np.sum(data)

    def memorize_finish(self):
        self._memorize_finish_called += 1

    def transform(self, data):
        return data - self._sum

def test_EvalFactor_end_to_end():
    from patsy.state import stateful_transform
    foo = stateful_transform(_MockTransform)
    e = EvalFactor("foo(x) + foo(foo(y))")
    state = {}
    eval_env = EvalEnvironment.capture(0)
    passes = e.memorize_passes_needed(state, eval_env)
    print(passes)
    print(state)
    assert passes == 2
    assert state["eval_env"].namespace["foo"] is foo
    for name in ["x", "y", "e", "state"]:
        assert name not in state["eval_env"].namespace
    import numpy as np
    e.memorize_chunk(state, 0,
                     {"x": np.array([1, 2]),
                      "y": np.array([10, 11])})
    assert state["transforms"]["_patsy_stobj0__foo__"]._memorize_chunk_called == 1
    assert state["transforms"]["_patsy_stobj2__foo__"]._memorize_chunk_called == 1
    e.memorize_chunk(state, 0, {"x": np.array([12, -10]),
                                "y": np.array([100, 3])})
    assert state["transforms"]["_patsy_stobj0__foo__"]._memorize_chunk_called == 2
    assert state["transforms"]["_patsy_stobj2__foo__"]._memorize_chunk_called == 2
    assert state["transforms"]["_patsy_stobj0__foo__"]._memorize_finish_called == 0
    assert state["transforms"]["_patsy_stobj2__foo__"]._memorize_finish_called == 0
    e.memorize_finish(state, 0)
    assert state["transforms"]["_patsy_stobj0__foo__"]._memorize_finish_called == 1
    assert state["transforms"]["_patsy_stobj2__foo__"]._memorize_finish_called == 1
    assert state["transforms"]["_patsy_stobj1__foo__"]._memorize_chunk_called == 0
    assert state["transforms"]["_patsy_stobj1__foo__"]._memorize_finish_called == 0
    e.memorize_chunk(state, 1, {"x": np.array([1, 2]),
                                "y": np.array([10, 11])})
    e.memorize_chunk(state, 1, {"x": np.array([12, -10]),
                                "y": np.array([100, 3])})
    e.memorize_finish(state, 1)
    for transform in six.itervalues(state["transforms"]):
        assert transform._memorize_chunk_called == 2
        assert transform._memorize_finish_called == 1
    # sums:
    # 0: 1 + 2 + 12 + -10 == 5
    # 2: 10 + 11 + 100 + 3 == 124
    # 1: (10 - 124) + (11 - 124) + (100 - 124) + (3 - 124) == -372
    # results:
    # 0: -4, -3, 7, -15
    # 2: -114, -113, -24, -121
    # 1: 258, 259, 348, 251
    # 0 + 1: 254, 256, 355, 236
    assert np.all(e.eval(state,
                         {"x": np.array([1, 2, 12, -10]),
                          "y": np.array([10, 11, 100, 3])})
                  == [254, 256, 355, 236])

def annotated_tokens(code):
    prev_was_dot = False
    it = PushbackAdapter(python_tokenize(code))
    for (token_type, token, origin) in it:
        props = {}
        props["bare_ref"] = (not prev_was_dot and token_type == tokenize.NAME)
        props["bare_funcall"] = (props["bare_ref"]
                                 and it.has_more() and it.peek()[1] == "(")
        yield (token_type, token, origin, props)
        prev_was_dot = (token == ".")

def test_annotated_tokens():
    tokens_without_origins = [(token_type, token, props)
                              for (token_type, token, origin, props)
                              in (annotated_tokens("a(b) + c.d"))]
    assert (tokens_without_origins
            == [(tokenize.NAME, "a", {"bare_ref": True, "bare_funcall": True}),
                (tokenize.OP, "(", {"bare_ref": False, "bare_funcall": False}),
                (tokenize.NAME, "b", {"bare_ref": True, "bare_funcall": False}),
                (tokenize.OP, ")", {"bare_ref": False, "bare_funcall": False}),
                (tokenize.OP, "+", {"bare_ref": False, "bare_funcall": False}),
                (tokenize.NAME, "c", {"bare_ref": True, "bare_funcall": False}),
                (tokenize.OP, ".", {"bare_ref": False, "bare_funcall": False}),
                (tokenize.NAME, "d",
                    {"bare_ref": False, "bare_funcall": False}),
                ])

    # This was a bug:
    assert len(list(annotated_tokens("x"))) == 1

def has_bare_variable_reference(names, code):
    for (_, token, _, props) in annotated_tokens(code):
        if props["bare_ref"] and token in names:
            return True
    return False

def replace_bare_funcalls(code, replacer):
    tokens = []
    for (token_type, token, origin, props) in annotated_tokens(code):
        if props["bare_ref"] and props["bare_funcall"]:
            token = replacer(token)
        tokens.append((token_type, token))
    return pretty_untokenize(tokens)

def test_replace_bare_funcalls():
    def replacer1(token):
        return {"a": "b", "foo": "_internal.foo.process"}.get(token, token)
    def t1(code, expected):
        replaced = replace_bare_funcalls(code, replacer1)
        print("%r -> %r" % (code, replaced))
        print("(wanted %r)" % (expected,))
        assert replaced == expected
    t1("foobar()", "foobar()")
    t1("a()", "b()")
    t1("foobar.a()", "foobar.a()")
    t1("foo()", "_internal.foo.process()")
    t1("a + 1", "a + 1")
    t1("b() + a() * x[foo(2 ** 3)]",
       "b() + b() * x[_internal.foo.process(2 ** 3)]")

class _FuncallCapturer(object):
    # captures the next funcall
    def __init__(self, start_token_type, start_token):
        self.func = [start_token]
        self.tokens = [(start_token_type, start_token)]
        self.paren_depth = 0
        self.started = False
        self.done = False

    def add_token(self, token_type, token):
        if self.done:
            return
        self.tokens.append((token_type, token))
        if token in ["(", "{", "["]:
            self.paren_depth += 1
        if token in [")", "}", "]"]:
            self.paren_depth -= 1
        assert self.paren_depth >= 0
        if not self.started:
            if token == "(":
                self.started = True
            else:
                assert token_type == tokenize.NAME or token == "."
                self.func.append(token)
        if self.started and self.paren_depth == 0:
            self.done = True

# This is not a very general function -- it assumes that all references to the
# given object are of the form '<obj_name>.something(method call)'.
def capture_obj_method_calls(obj_name, code):
    capturers = []
    for (token_type, token, origin, props) in annotated_tokens(code):
        for capturer in capturers:
            capturer.add_token(token_type, token)
        if props["bare_ref"] and token == obj_name:
            capturers.append(_FuncallCapturer(token_type, token))
    return [("".join(capturer.func), pretty_untokenize(capturer.tokens))
            for capturer in capturers]

def test_capture_obj_method_calls():
    assert (capture_obj_method_calls("foo", "a + foo.baz(bar) + b.c(d)")
            == [("foo.baz", "foo.baz(bar)")])
    assert (capture_obj_method_calls("b", "a + foo.baz(bar) + b.c(d)")
            == [("b.c", "b.c(d)")])
    assert (capture_obj_method_calls("foo", "foo.bar(foo.baz(quux))")
            == [("foo.bar", "foo.bar(foo.baz(quux))"),
                ("foo.baz", "foo.baz(quux)")])
    assert (capture_obj_method_calls("bar", "foo[bar.baz(x(z[asdf])) ** 2]")
            == [("bar.baz", "bar.baz(x(z[asdf]))")])
