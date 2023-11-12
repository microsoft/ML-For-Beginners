# Natural Language Toolkit: Combinatory Categorial Grammar
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Graeme Gange <ggange@csse.unimelb.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
CCG Combinators
"""

from abc import ABCMeta, abstractmethod

from nltk.ccg.api import FunctionalCategory


class UndirectedBinaryCombinator(metaclass=ABCMeta):
    """
    Abstract class for representing a binary combinator.
    Merely defines functions for checking if the function and argument
    are able to be combined, and what the resulting category is.

    Note that as no assumptions are made as to direction, the unrestricted
    combinators can perform all backward, forward and crossed variations
    of the combinators; these restrictions must be added in the rule
    class.
    """

    @abstractmethod
    def can_combine(self, function, argument):
        pass

    @abstractmethod
    def combine(self, function, argument):
        pass


class DirectedBinaryCombinator(metaclass=ABCMeta):
    """
    Wrapper for the undirected binary combinator.
    It takes left and right categories, and decides which is to be
    the function, and which the argument.
    It then decides whether or not they can be combined.
    """

    @abstractmethod
    def can_combine(self, left, right):
        pass

    @abstractmethod
    def combine(self, left, right):
        pass


class ForwardCombinator(DirectedBinaryCombinator):
    """
    Class representing combinators where the primary functor is on the left.

    Takes an undirected combinator, and a predicate which adds constraints
    restricting the cases in which it may apply.
    """

    def __init__(self, combinator, predicate, suffix=""):
        self._combinator = combinator
        self._predicate = predicate
        self._suffix = suffix

    def can_combine(self, left, right):
        return self._combinator.can_combine(left, right) and self._predicate(
            left, right
        )

    def combine(self, left, right):
        yield from self._combinator.combine(left, right)

    def __str__(self):
        return f">{self._combinator}{self._suffix}"


class BackwardCombinator(DirectedBinaryCombinator):
    """
    The backward equivalent of the ForwardCombinator class.
    """

    def __init__(self, combinator, predicate, suffix=""):
        self._combinator = combinator
        self._predicate = predicate
        self._suffix = suffix

    def can_combine(self, left, right):
        return self._combinator.can_combine(right, left) and self._predicate(
            left, right
        )

    def combine(self, left, right):
        yield from self._combinator.combine(right, left)

    def __str__(self):
        return f"<{self._combinator}{self._suffix}"


class UndirectedFunctionApplication(UndirectedBinaryCombinator):
    """
    Class representing function application.
    Implements rules of the form:
    X/Y Y -> X (>)
    And the corresponding backwards application rule
    """

    def can_combine(self, function, argument):
        if not function.is_function():
            return False

        return not function.arg().can_unify(argument) is None

    def combine(self, function, argument):
        if not function.is_function():
            return

        subs = function.arg().can_unify(argument)
        if subs is None:
            return

        yield function.res().substitute(subs)

    def __str__(self):
        return ""


# Predicates for function application.

# Ensures the left functor takes an argument on the right
def forwardOnly(left, right):
    return left.dir().is_forward()


# Ensures the right functor takes an argument on the left
def backwardOnly(left, right):
    return right.dir().is_backward()


# Application combinator instances
ForwardApplication = ForwardCombinator(UndirectedFunctionApplication(), forwardOnly)
BackwardApplication = BackwardCombinator(UndirectedFunctionApplication(), backwardOnly)


class UndirectedComposition(UndirectedBinaryCombinator):
    """
    Functional composition (harmonic) combinator.
    Implements rules of the form
    X/Y Y/Z -> X/Z (B>)
    And the corresponding backwards and crossed variations.
    """

    def can_combine(self, function, argument):
        # Can only combine two functions, and both functions must
        # allow composition.
        if not (function.is_function() and argument.is_function()):
            return False
        if function.dir().can_compose() and argument.dir().can_compose():
            return not function.arg().can_unify(argument.res()) is None
        return False

    def combine(self, function, argument):
        if not (function.is_function() and argument.is_function()):
            return
        if function.dir().can_compose() and argument.dir().can_compose():
            subs = function.arg().can_unify(argument.res())
            if subs is not None:
                yield FunctionalCategory(
                    function.res().substitute(subs),
                    argument.arg().substitute(subs),
                    argument.dir(),
                )

    def __str__(self):
        return "B"


# Predicates for restricting application of straight composition.
def bothForward(left, right):
    return left.dir().is_forward() and right.dir().is_forward()


def bothBackward(left, right):
    return left.dir().is_backward() and right.dir().is_backward()


# Predicates for crossed composition
def crossedDirs(left, right):
    return left.dir().is_forward() and right.dir().is_backward()


def backwardBxConstraint(left, right):
    # The functors must be crossed inwards
    if not crossedDirs(left, right):
        return False
    # Permuting combinators must be allowed
    if not left.dir().can_cross() and right.dir().can_cross():
        return False
    # The resulting argument category is restricted to be primitive
    return left.arg().is_primitive()


# Straight composition combinators
ForwardComposition = ForwardCombinator(UndirectedComposition(), forwardOnly)
BackwardComposition = BackwardCombinator(UndirectedComposition(), backwardOnly)

# Backward crossed composition
BackwardBx = BackwardCombinator(
    UndirectedComposition(), backwardBxConstraint, suffix="x"
)


class UndirectedSubstitution(UndirectedBinaryCombinator):
    r"""
    Substitution (permutation) combinator.
    Implements rules of the form
    Y/Z (X\Y)/Z -> X/Z (<Sx)
    And other variations.
    """

    def can_combine(self, function, argument):
        if function.is_primitive() or argument.is_primitive():
            return False

        # These could potentially be moved to the predicates, as the
        # constraints may not be general to all languages.
        if function.res().is_primitive():
            return False
        if not function.arg().is_primitive():
            return False

        if not (function.dir().can_compose() and argument.dir().can_compose()):
            return False
        return (function.res().arg() == argument.res()) and (
            function.arg() == argument.arg()
        )

    def combine(self, function, argument):
        if self.can_combine(function, argument):
            yield FunctionalCategory(
                function.res().res(), argument.arg(), argument.dir()
            )

    def __str__(self):
        return "S"


# Predicate for forward substitution
def forwardSConstraint(left, right):
    if not bothForward(left, right):
        return False
    return left.res().dir().is_forward() and left.arg().is_primitive()


# Predicate for backward crossed substitution
def backwardSxConstraint(left, right):
    if not left.dir().can_cross() and right.dir().can_cross():
        return False
    if not bothForward(left, right):
        return False
    return right.res().dir().is_backward() and right.arg().is_primitive()


# Instances of substitution combinators
ForwardSubstitution = ForwardCombinator(UndirectedSubstitution(), forwardSConstraint)
BackwardSx = BackwardCombinator(UndirectedSubstitution(), backwardSxConstraint, "x")


# Retrieves the left-most functional category.
# ie, (N\N)/(S/NP) => N\N
def innermostFunction(categ):
    while categ.res().is_function():
        categ = categ.res()
    return categ


class UndirectedTypeRaise(UndirectedBinaryCombinator):
    """
    Undirected combinator for type raising.
    """

    def can_combine(self, function, arg):
        # The argument must be a function.
        # The restriction that arg.res() must be a function
        # merely reduces redundant type-raising; if arg.res() is
        # primitive, we have:
        # X Y\X =>(<T) Y/(Y\X) Y\X =>(>) Y
        # which is equivalent to
        # X Y\X =>(<) Y
        if not (arg.is_function() and arg.res().is_function()):
            return False

        arg = innermostFunction(arg)

        # left, arg_categ are undefined!
        subs = left.can_unify(arg_categ.arg())
        if subs is not None:
            return True
        return False

    def combine(self, function, arg):
        if not (
            function.is_primitive() and arg.is_function() and arg.res().is_function()
        ):
            return

        # Type-raising matches only the innermost application.
        arg = innermostFunction(arg)

        subs = function.can_unify(arg.arg())
        if subs is not None:
            xcat = arg.res().substitute(subs)
            yield FunctionalCategory(
                xcat, FunctionalCategory(xcat, function, arg.dir()), -(arg.dir())
            )

    def __str__(self):
        return "T"


# Predicates for type-raising
# The direction of the innermost category must be towards
# the primary functor.
# The restriction that the variable must be primitive is not
# common to all versions of CCGs; some authors have other restrictions.
def forwardTConstraint(left, right):
    arg = innermostFunction(right)
    return arg.dir().is_backward() and arg.res().is_primitive()


def backwardTConstraint(left, right):
    arg = innermostFunction(left)
    return arg.dir().is_forward() and arg.res().is_primitive()


# Instances of type-raising combinators
ForwardT = ForwardCombinator(UndirectedTypeRaise(), forwardTConstraint)
BackwardT = BackwardCombinator(UndirectedTypeRaise(), backwardTConstraint)
