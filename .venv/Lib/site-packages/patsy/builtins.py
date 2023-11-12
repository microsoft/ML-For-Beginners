# This file is part of Patsy
# Copyright (C) 2011-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This module sets up the namespace of stuff that is available to formulas by
# default. All formulas are interpreted in an environment that acts as if
#   from patsy.builtins import *
# has been executed. (Of course, you can also execute this yourself if you
# want to use these in your regular code for some reason.)

__all__ = ["I", "Q"]

from patsy.contrasts import ContrastMatrix, Treatment, Poly, Sum, Helmert, Diff
__all__ += ["ContrastMatrix", "Treatment", "Poly", "Sum", "Helmert", "Diff"]

from patsy.categorical import C
__all__ += ["C"]

from patsy.state import center, standardize, scale
__all__ += ["center", "standardize", "scale"]

from patsy.splines import bs
__all__ += ["bs"]

from patsy.mgcv_cubic_splines import cr, cc, te
__all__ += ["cr", "cc", "te"]

def I(x):
    """The identity function. Simply returns its input unchanged.

    Since Patsy's formula parser ignores anything inside a function call
    syntax, this is useful to 'hide' arithmetic operations from it. For
    instance::

      y ~ x1 + x2

    has ``x1`` and ``x2`` as two separate predictors. But in::

      y ~ I(x1 + x2)

    we instead have a single predictor, defined to be the sum of ``x1`` and
    ``x2``."""
    return x

def test_I():
    assert I(1) == 1
    assert I(None) is None

def Q(name):
    """A way to 'quote' variable names, especially ones that do not otherwise
    meet Python's variable name rules.

    If ``x`` is a variable, ``Q("x")`` returns the value of ``x``. (Note that
    ``Q`` takes the *string* ``"x"``, not the value of ``x`` itself.) This
    works even if instead of ``x``, we have a variable name that would not
    otherwise be legal in Python.

    For example, if you have a column of data named ``weight.in.kg``, then you
    can't write::

      y ~ weight.in.kg

    because Python will try to find a variable named ``weight``, that has an
    attribute named ``in``, that has an attribute named ``kg``. (And worse
    yet, ``in`` is a reserved word, which makes this example doubly broken.)
    Instead, write::

      y ~ Q("weight.in.kg")

    and all will be well. Note, though, that this requires embedding a Python
    string inside your formula, which may require some care with your quote
    marks. Some standard options include::

      my_fit_function("y ~ Q('weight.in.kg')", ...)
      my_fit_function('y ~ Q("weight.in.kg")', ...)
      my_fit_function("y ~ Q(\\"weight.in.kg\\")", ...)

    Note also that ``Q`` is an ordinary Python function, which means that you
    can use it in more complex expressions. For example, this is a legal
    formula::

      y ~ np.sqrt(Q("weight.in.kg"))
    """
    from patsy.eval import EvalEnvironment
    env = EvalEnvironment.capture(1)
    try:
        return env.namespace[name]
    except KeyError:
        raise NameError("no data named %r found" % (name,))

def test_Q():
    a = 1
    assert Q("a") == 1
    assert Q("Q") is Q
    import pytest
    pytest.raises(NameError, Q, "asdfsadfdsad")

