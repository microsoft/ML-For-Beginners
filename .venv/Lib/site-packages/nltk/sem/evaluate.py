# Natural Language Toolkit: Models for first-order languages with lambda
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>,
# URL: <https://www.nltk.org>
# For license information, see LICENSE.TXT

# TODO:
# - fix tracing
# - fix iterator-based approach to existentials

"""
This module provides data structures for representing first-order
models.
"""

import inspect
import re
import sys
import textwrap
from pprint import pformat

from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
    AbstractVariableExpression,
    AllExpression,
    AndExpression,
    ApplicationExpression,
    EqualityExpression,
    ExistsExpression,
    Expression,
    IffExpression,
    ImpExpression,
    IndividualVariableExpression,
    IotaExpression,
    LambdaExpression,
    NegatedExpression,
    OrExpression,
    Variable,
    is_indvar,
)


class Error(Exception):
    pass


class Undefined(Error):
    pass


def trace(f, *args, **kw):
    argspec = inspect.getfullargspec(f)
    d = dict(zip(argspec[0], args))
    if d.pop("trace", None):
        print()
        for item in d.items():
            print("%s => %s" % item)
    return f(*args, **kw)


def is_rel(s):
    """
    Check whether a set represents a relation (of any arity).

    :param s: a set containing tuples of str elements
    :type s: set
    :rtype: bool
    """
    # we have the empty relation, i.e. set()
    if len(s) == 0:
        return True
    # all the elements are tuples of the same length
    elif all(isinstance(el, tuple) for el in s) and len(max(s)) == len(min(s)):
        return True
    else:
        raise ValueError("Set %r contains sequences of different lengths" % s)


def set2rel(s):
    """
    Convert a set containing individuals (strings or numbers) into a set of
    unary tuples. Any tuples of strings already in the set are passed through
    unchanged.

    For example:
      - set(['a', 'b']) => set([('a',), ('b',)])
      - set([3, 27]) => set([('3',), ('27',)])

    :type s: set
    :rtype: set of tuple of str
    """
    new = set()
    for elem in s:
        if isinstance(elem, str):
            new.add((elem,))
        elif isinstance(elem, int):
            new.add(str(elem))
        else:
            new.add(elem)
    return new


def arity(rel):
    """
    Check the arity of a relation.
    :type rel: set of tuples
    :rtype: int of tuple of str
    """
    if len(rel) == 0:
        return 0
    return len(list(rel)[0])


class Valuation(dict):
    """
    A dictionary which represents a model-theoretic Valuation of non-logical constants.
    Keys are strings representing the constants to be interpreted, and values correspond
    to individuals (represented as strings) and n-ary relations (represented as sets of tuples
    of strings).

    An instance of ``Valuation`` will raise a KeyError exception (i.e.,
    just behave like a standard  dictionary) if indexed with an expression that
    is not in its list of symbols.
    """

    def __init__(self, xs):
        """
        :param xs: a list of (symbol, value) pairs.
        """
        super().__init__()
        for (sym, val) in xs:
            if isinstance(val, str) or isinstance(val, bool):
                self[sym] = val
            elif isinstance(val, set):
                self[sym] = set2rel(val)
            else:
                msg = textwrap.fill(
                    "Error in initializing Valuation. "
                    "Unrecognized value for symbol '%s':\n%s" % (sym, val),
                    width=66,
                )

                raise ValueError(msg)

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        else:
            raise Undefined("Unknown expression: '%s'" % key)

    def __str__(self):
        return pformat(self)

    @property
    def domain(self):
        """Set-theoretic domain of the value-space of a Valuation."""
        dom = []
        for val in self.values():
            if isinstance(val, str):
                dom.append(val)
            elif not isinstance(val, bool):
                dom.extend(
                    [elem for tuple_ in val for elem in tuple_ if elem is not None]
                )
        return set(dom)

    @property
    def symbols(self):
        """The non-logical constants which the Valuation recognizes."""
        return sorted(self.keys())

    @classmethod
    def fromstring(cls, s):
        return read_valuation(s)


##########################################
# REs used by the _read_valuation function
##########################################
_VAL_SPLIT_RE = re.compile(r"\s*=+>\s*")
_ELEMENT_SPLIT_RE = re.compile(r"\s*,\s*")
_TUPLES_RE = re.compile(
    r"""\s*
                                (\([^)]+\))  # tuple-expression
                                \s*""",
    re.VERBOSE,
)


def _read_valuation_line(s):
    """
    Read a line in a valuation file.

    Lines are expected to be of the form::

      noosa => n
      girl => {g1, g2}
      chase => {(b1, g1), (b2, g1), (g1, d1), (g2, d2)}

    :param s: input line
    :type s: str
    :return: a pair (symbol, value)
    :rtype: tuple
    """
    pieces = _VAL_SPLIT_RE.split(s)
    symbol = pieces[0]
    value = pieces[1]
    # check whether the value is meant to be a set
    if value.startswith("{"):
        value = value[1:-1]
        tuple_strings = _TUPLES_RE.findall(value)
        # are the set elements tuples?
        if tuple_strings:
            set_elements = []
            for ts in tuple_strings:
                ts = ts[1:-1]
                element = tuple(_ELEMENT_SPLIT_RE.split(ts))
                set_elements.append(element)
        else:
            set_elements = _ELEMENT_SPLIT_RE.split(value)
        value = set(set_elements)
    return symbol, value


def read_valuation(s, encoding=None):
    """
    Convert a valuation string into a valuation.

    :param s: a valuation string
    :type s: str
    :param encoding: the encoding of the input string, if it is binary
    :type encoding: str
    :return: a ``nltk.sem`` valuation
    :rtype: Valuation
    """
    if encoding is not None:
        s = s.decode(encoding)
    statements = []
    for linenum, line in enumerate(s.splitlines()):
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        try:
            statements.append(_read_valuation_line(line))
        except ValueError as e:
            raise ValueError(f"Unable to parse line {linenum}: {line}") from e
    return Valuation(statements)


class Assignment(dict):
    r"""
    A dictionary which represents an assignment of values to variables.

    An assignment can only assign values from its domain.

    If an unknown expression *a* is passed to a model *M*\ 's
    interpretation function *i*, *i* will first check whether *M*\ 's
    valuation assigns an interpretation to *a* as a constant, and if
    this fails, *i* will delegate the interpretation of *a* to
    *g*. *g* only assigns values to individual variables (i.e.,
    members of the class ``IndividualVariableExpression`` in the ``logic``
    module. If a variable is not assigned a value by *g*, it will raise
    an ``Undefined`` exception.

    A variable *Assignment* is a mapping from individual variables to
    entities in the domain. Individual variables are usually indicated
    with the letters ``'x'``, ``'y'``, ``'w'`` and ``'z'``, optionally
    followed by an integer (e.g., ``'x0'``, ``'y332'``).  Assignments are
    created using the ``Assignment`` constructor, which also takes the
    domain as a parameter.

        >>> from nltk.sem.evaluate import Assignment
        >>> dom = set(['u1', 'u2', 'u3', 'u4'])
        >>> g3 = Assignment(dom, [('x', 'u1'), ('y', 'u2')])
        >>> g3 == {'x': 'u1', 'y': 'u2'}
        True

    There is also a ``print`` format for assignments which uses a notation
    closer to that in logic textbooks:

        >>> print(g3)
        g[u1/x][u2/y]

    It is also possible to update an assignment using the ``add`` method:

        >>> dom = set(['u1', 'u2', 'u3', 'u4'])
        >>> g4 = Assignment(dom)
        >>> g4.add('x', 'u1')
        {'x': 'u1'}

    With no arguments, ``purge()`` is equivalent to ``clear()`` on a dictionary:

        >>> g4.purge()
        >>> g4
        {}

    :param domain: the domain of discourse
    :type domain: set
    :param assign: a list of (varname, value) associations
    :type assign: list
    """

    def __init__(self, domain, assign=None):
        super().__init__()
        self.domain = domain
        if assign:
            for (var, val) in assign:
                assert val in self.domain, "'{}' is not in the domain: {}".format(
                    val,
                    self.domain,
                )
                assert is_indvar(var), (
                    "Wrong format for an Individual Variable: '%s'" % var
                )
                self[var] = val
        self.variant = None
        self._addvariant()

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        else:
            raise Undefined("Not recognized as a variable: '%s'" % key)

    def copy(self):
        new = Assignment(self.domain)
        new.update(self)
        return new

    def purge(self, var=None):
        """
        Remove one or all keys (i.e. logic variables) from an
        assignment, and update ``self.variant``.

        :param var: a Variable acting as a key for the assignment.
        """
        if var:
            del self[var]
        else:
            self.clear()
        self._addvariant()
        return None

    def __str__(self):
        """
        Pretty printing for assignments. {'x', 'u'} appears as 'g[u/x]'
        """
        gstring = "g"
        # Deterministic output for unit testing.
        variant = sorted(self.variant)
        for (val, var) in variant:
            gstring += f"[{val}/{var}]"
        return gstring

    def _addvariant(self):
        """
        Create a more pretty-printable version of the assignment.
        """
        list_ = []
        for item in self.items():
            pair = (item[1], item[0])
            list_.append(pair)
        self.variant = list_
        return None

    def add(self, var, val):
        """
        Add a new variable-value pair to the assignment, and update
        ``self.variant``.

        """
        assert val in self.domain, f"{val} is not in the domain {self.domain}"
        assert is_indvar(var), "Wrong format for an Individual Variable: '%s'" % var
        self[var] = val
        self._addvariant()
        return self


class Model:
    """
    A first order model is a domain *D* of discourse and a valuation *V*.

    A domain *D* is a set, and a valuation *V* is a map that associates
    expressions with values in the model.
    The domain of *V* should be a subset of *D*.

    Construct a new ``Model``.

    :type domain: set
    :param domain: A set of entities representing the domain of discourse of the model.
    :type valuation: Valuation
    :param valuation: the valuation of the model.
    :param prop: If this is set, then we are building a propositional\
    model and don't require the domain of *V* to be subset of *D*.
    """

    def __init__(self, domain, valuation):
        assert isinstance(domain, set)
        self.domain = domain
        self.valuation = valuation
        if not domain.issuperset(valuation.domain):
            raise Error(
                "The valuation domain, %s, must be a subset of the model's domain, %s"
                % (valuation.domain, domain)
            )

    def __repr__(self):
        return f"({self.domain!r}, {self.valuation!r})"

    def __str__(self):
        return f"Domain = {self.domain},\nValuation = \n{self.valuation}"

    def evaluate(self, expr, g, trace=None):
        """
        Read input expressions, and provide a handler for ``satisfy``
        that blocks further propagation of the ``Undefined`` error.
        :param expr: An ``Expression`` of ``logic``.
        :type g: Assignment
        :param g: an assignment to individual variables.
        :rtype: bool or 'Undefined'
        """
        try:
            parsed = Expression.fromstring(expr)
            value = self.satisfy(parsed, g, trace=trace)
            if trace:
                print()
                print(f"'{expr}' evaluates to {value} under M, {g}")
            return value
        except Undefined:
            if trace:
                print()
                print(f"'{expr}' is undefined under M, {g}")
            return "Undefined"

    def satisfy(self, parsed, g, trace=None):
        """
        Recursive interpretation function for a formula of first-order logic.

        Raises an ``Undefined`` error when ``parsed`` is an atomic string
        but is not a symbol or an individual variable.

        :return: Returns a truth value or ``Undefined`` if ``parsed`` is\
        complex, and calls the interpretation function ``i`` if ``parsed``\
        is atomic.

        :param parsed: An expression of ``logic``.
        :type g: Assignment
        :param g: an assignment to individual variables.
        """

        if isinstance(parsed, ApplicationExpression):
            function, arguments = parsed.uncurry()
            if isinstance(function, AbstractVariableExpression):
                # It's a predicate expression ("P(x,y)"), so used uncurried arguments
                funval = self.satisfy(function, g)
                argvals = tuple(self.satisfy(arg, g) for arg in arguments)
                return argvals in funval
            else:
                # It must be a lambda expression, so use curried form
                funval = self.satisfy(parsed.function, g)
                argval = self.satisfy(parsed.argument, g)
                return funval[argval]
        elif isinstance(parsed, NegatedExpression):
            return not self.satisfy(parsed.term, g)
        elif isinstance(parsed, AndExpression):
            return self.satisfy(parsed.first, g) and self.satisfy(parsed.second, g)
        elif isinstance(parsed, OrExpression):
            return self.satisfy(parsed.first, g) or self.satisfy(parsed.second, g)
        elif isinstance(parsed, ImpExpression):
            return (not self.satisfy(parsed.first, g)) or self.satisfy(parsed.second, g)
        elif isinstance(parsed, IffExpression):
            return self.satisfy(parsed.first, g) == self.satisfy(parsed.second, g)
        elif isinstance(parsed, EqualityExpression):
            return self.satisfy(parsed.first, g) == self.satisfy(parsed.second, g)
        elif isinstance(parsed, AllExpression):
            new_g = g.copy()
            for u in self.domain:
                new_g.add(parsed.variable.name, u)
                if not self.satisfy(parsed.term, new_g):
                    return False
            return True
        elif isinstance(parsed, ExistsExpression):
            new_g = g.copy()
            for u in self.domain:
                new_g.add(parsed.variable.name, u)
                if self.satisfy(parsed.term, new_g):
                    return True
            return False
        elif isinstance(parsed, IotaExpression):
            new_g = g.copy()
            for u in self.domain:
                new_g.add(parsed.variable.name, u)
                if self.satisfy(parsed.term, new_g):
                    return True
            return False
        elif isinstance(parsed, LambdaExpression):
            cf = {}
            var = parsed.variable.name
            for u in self.domain:
                val = self.satisfy(parsed.term, g.add(var, u))
                # NB the dict would be a lot smaller if we do this:
                # if val: cf[u] = val
                # But then need to deal with cases where f(a) should yield
                # a function rather than just False.
                cf[u] = val
            return cf
        else:
            return self.i(parsed, g, trace)

    # @decorator(trace_eval)
    def i(self, parsed, g, trace=False):
        """
        An interpretation function.

        Assuming that ``parsed`` is atomic:

        - if ``parsed`` is a non-logical constant, calls the valuation *V*
        - else if ``parsed`` is an individual variable, calls assignment *g*
        - else returns ``Undefined``.

        :param parsed: an ``Expression`` of ``logic``.
        :type g: Assignment
        :param g: an assignment to individual variables.
        :return: a semantic value
        """
        # If parsed is a propositional letter 'p', 'q', etc, it could be in valuation.symbols
        # and also be an IndividualVariableExpression. We want to catch this first case.
        # So there is a procedural consequence to the ordering of clauses here:
        if parsed.variable.name in self.valuation.symbols:
            return self.valuation[parsed.variable.name]
        elif isinstance(parsed, IndividualVariableExpression):
            return g[parsed.variable.name]

        else:
            raise Undefined("Can't find a value for %s" % parsed)

    def satisfiers(self, parsed, varex, g, trace=None, nesting=0):
        """
        Generate the entities from the model's domain that satisfy an open formula.

        :param parsed: an open formula
        :type parsed: Expression
        :param varex: the relevant free individual variable in ``parsed``.
        :type varex: VariableExpression or str
        :param g: a variable assignment
        :type g:  Assignment
        :return: a set of the entities that satisfy ``parsed``.
        """

        spacer = "   "
        indent = spacer + (spacer * nesting)
        candidates = []

        if isinstance(varex, str):
            var = Variable(varex)
        else:
            var = varex

        if var in parsed.free():
            if trace:
                print()
                print(
                    (spacer * nesting)
                    + f"Open formula is '{parsed}' with assignment {g}"
                )
            for u in self.domain:
                new_g = g.copy()
                new_g.add(var.name, u)
                if trace and trace > 1:
                    lowtrace = trace - 1
                else:
                    lowtrace = 0
                value = self.satisfy(parsed, new_g, lowtrace)

                if trace:
                    print(indent + "(trying assignment %s)" % new_g)

                # parsed == False under g[u/var]?
                if value == False:
                    if trace:
                        print(indent + f"value of '{parsed}' under {new_g} is False")

                # so g[u/var] is a satisfying assignment
                else:
                    candidates.append(u)
                    if trace:
                        print(indent + f"value of '{parsed}' under {new_g} is {value}")

            result = {c for c in candidates}
        # var isn't free in parsed
        else:
            raise Undefined(f"{var.name} is not free in {parsed}")

        return result


# //////////////////////////////////////////////////////////////////////
# Demo..
# //////////////////////////////////////////////////////////////////////
# number of spacer chars
mult = 30

# Demo 1: Propositional Logic
#################
def propdemo(trace=None):
    """Example of a propositional model."""

    global val1, dom1, m1, g1
    val1 = Valuation([("P", True), ("Q", True), ("R", False)])
    dom1 = set()
    m1 = Model(dom1, val1)
    g1 = Assignment(dom1)

    print()
    print("*" * mult)
    print("Propositional Formulas Demo")
    print("*" * mult)
    print("(Propositional constants treated as nullary predicates)")
    print()
    print("Model m1:\n", m1)
    print("*" * mult)
    sentences = [
        "(P & Q)",
        "(P & R)",
        "- P",
        "- R",
        "- - P",
        "- (P & R)",
        "(P | R)",
        "(R | P)",
        "(R | R)",
        "(- P | R)",
        "(P | - P)",
        "(P -> Q)",
        "(P -> R)",
        "(R -> P)",
        "(P <-> P)",
        "(R <-> R)",
        "(P <-> R)",
    ]

    for sent in sentences:
        if trace:
            print()
            m1.evaluate(sent, g1, trace)
        else:
            print(f"The value of '{sent}' is: {m1.evaluate(sent, g1)}")


# Demo 2: FOL Model
#############


def folmodel(quiet=False, trace=None):
    """Example of a first-order model."""

    global val2, v2, dom2, m2, g2

    v2 = [
        ("adam", "b1"),
        ("betty", "g1"),
        ("fido", "d1"),
        ("girl", {"g1", "g2"}),
        ("boy", {"b1", "b2"}),
        ("dog", {"d1"}),
        ("love", {("b1", "g1"), ("b2", "g2"), ("g1", "b1"), ("g2", "b1")}),
    ]
    val2 = Valuation(v2)
    dom2 = val2.domain
    m2 = Model(dom2, val2)
    g2 = Assignment(dom2, [("x", "b1"), ("y", "g2")])

    if not quiet:
        print()
        print("*" * mult)
        print("Models Demo")
        print("*" * mult)
        print("Model m2:\n", "-" * 14, "\n", m2)
        print("Variable assignment = ", g2)

        exprs = ["adam", "boy", "love", "walks", "x", "y", "z"]
        parsed_exprs = [Expression.fromstring(e) for e in exprs]

        print()
        for parsed in parsed_exprs:
            try:
                print(
                    "The interpretation of '%s' in m2 is %s"
                    % (parsed, m2.i(parsed, g2))
                )
            except Undefined:
                print("The interpretation of '%s' in m2 is Undefined" % parsed)

        applications = [
            ("boy", ("adam")),
            ("walks", ("adam",)),
            ("love", ("adam", "y")),
            ("love", ("y", "adam")),
        ]

        for (fun, args) in applications:
            try:
                funval = m2.i(Expression.fromstring(fun), g2)
                argsval = tuple(m2.i(Expression.fromstring(arg), g2) for arg in args)
                print(f"{fun}({args}) evaluates to {argsval in funval}")
            except Undefined:
                print(f"{fun}({args}) evaluates to Undefined")


# Demo 3: FOL
#########


def foldemo(trace=None):
    """
    Interpretation of closed expressions in a first-order model.
    """
    folmodel(quiet=True)

    print()
    print("*" * mult)
    print("FOL Formulas Demo")
    print("*" * mult)

    formulas = [
        "love (adam, betty)",
        "(adam = mia)",
        "\\x. (boy(x) | girl(x))",
        "\\x. boy(x)(adam)",
        "\\x y. love(x, y)",
        "\\x y. love(x, y)(adam)(betty)",
        "\\x y. love(x, y)(adam, betty)",
        "\\x y. (boy(x) & love(x, y))",
        "\\x. exists y. (boy(x) & love(x, y))",
        "exists z1. boy(z1)",
        "exists x. (boy(x) &  -(x = adam))",
        "exists x. (boy(x) & all y. love(y, x))",
        "all x. (boy(x) | girl(x))",
        "all x. (girl(x) -> exists y. boy(y) & love(x, y))",  # Every girl loves exists boy.
        "exists x. (boy(x) & all y. (girl(y) -> love(y, x)))",  # There is exists boy that every girl loves.
        "exists x. (boy(x) & all y. (girl(y) -> love(x, y)))",  # exists boy loves every girl.
        "all x. (dog(x) -> - girl(x))",
        "exists x. exists y. (love(x, y) & love(x, y))",
    ]

    for fmla in formulas:
        g2.purge()
        if trace:
            m2.evaluate(fmla, g2, trace)
        else:
            print(f"The value of '{fmla}' is: {m2.evaluate(fmla, g2)}")


# Demo 3: Satisfaction
#############


def satdemo(trace=None):
    """Satisfiers of an open formula in a first order model."""

    print()
    print("*" * mult)
    print("Satisfiers Demo")
    print("*" * mult)

    folmodel(quiet=True)

    formulas = [
        "boy(x)",
        "(x = x)",
        "(boy(x) | girl(x))",
        "(boy(x) & girl(x))",
        "love(adam, x)",
        "love(x, adam)",
        "-(x = adam)",
        "exists z22. love(x, z22)",
        "exists y. love(y, x)",
        "all y. (girl(y) -> love(x, y))",
        "all y. (girl(y) -> love(y, x))",
        "all y. (girl(y) -> (boy(x) & love(y, x)))",
        "(boy(x) & all y. (girl(y) -> love(x, y)))",
        "(boy(x) & all y. (girl(y) -> love(y, x)))",
        "(boy(x) & exists y. (girl(y) & love(y, x)))",
        "(girl(x) -> dog(x))",
        "all y. (dog(y) -> (x = y))",
        "exists y. love(y, x)",
        "exists y. (love(adam, y) & love(y, x))",
    ]

    if trace:
        print(m2)

    for fmla in formulas:
        print(fmla)
        Expression.fromstring(fmla)

    parsed = [Expression.fromstring(fmla) for fmla in formulas]

    for p in parsed:
        g2.purge()
        print(
            "The satisfiers of '{}' are: {}".format(p, m2.satisfiers(p, "x", g2, trace))
        )


def demo(num=0, trace=None):
    """
    Run exists demos.

     - num = 1: propositional logic demo
     - num = 2: first order model demo (only if trace is set)
     - num = 3: first order sentences demo
     - num = 4: satisfaction of open formulas demo
     - any other value: run all the demos

    :param trace: trace = 1, or trace = 2 for more verbose tracing
    """
    demos = {1: propdemo, 2: folmodel, 3: foldemo, 4: satdemo}

    try:
        demos[num](trace=trace)
    except KeyError:
        for num in demos:
            demos[num](trace=trace)


if __name__ == "__main__":
    demo(2, trace=0)
