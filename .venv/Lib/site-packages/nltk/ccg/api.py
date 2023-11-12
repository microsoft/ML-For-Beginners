# Natural Language Toolkit: CCG Categories
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Graeme Gange <ggange@csse.unimelb.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from abc import ABCMeta, abstractmethod
from functools import total_ordering

from nltk.internals import raise_unorderable_types


@total_ordering
class AbstractCCGCategory(metaclass=ABCMeta):
    """
    Interface for categories in combinatory grammars.
    """

    @abstractmethod
    def is_primitive(self):
        """
        Returns true if the category is primitive.
        """

    @abstractmethod
    def is_function(self):
        """
        Returns true if the category is a function application.
        """

    @abstractmethod
    def is_var(self):
        """
        Returns true if the category is a variable.
        """

    @abstractmethod
    def substitute(self, substitutions):
        """
        Takes a set of (var, category) substitutions, and replaces every
        occurrence of the variable with the corresponding category.
        """

    @abstractmethod
    def can_unify(self, other):
        """
        Determines whether two categories can be unified.
         - Returns None if they cannot be unified
         - Returns a list of necessary substitutions if they can.
        """

    # Utility functions: comparison, strings and hashing.
    @abstractmethod
    def __str__(self):
        pass

    def __eq__(self, other):
        return (
            self.__class__ is other.__class__
            and self._comparison_key == other._comparison_key
        )

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, AbstractCCGCategory):
            raise_unorderable_types("<", self, other)
        if self.__class__ is other.__class__:
            return self._comparison_key < other._comparison_key
        else:
            return self.__class__.__name__ < other.__class__.__name__

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self._comparison_key)
            return self._hash


class CCGVar(AbstractCCGCategory):
    """
    Class representing a variable CCG category.
    Used for conjunctions (and possibly type-raising, if implemented as a
    unary rule).
    """

    _maxID = 0

    def __init__(self, prim_only=False):
        """Initialize a variable (selects a new identifier)

        :param prim_only: a boolean that determines whether the variable is
                          restricted to primitives
        :type prim_only: bool
        """
        self._id = self.new_id()
        self._prim_only = prim_only
        self._comparison_key = self._id

    @classmethod
    def new_id(cls):
        """
        A class method allowing generation of unique variable identifiers.
        """
        cls._maxID = cls._maxID + 1
        return cls._maxID - 1

    @classmethod
    def reset_id(cls):
        cls._maxID = 0

    def is_primitive(self):
        return False

    def is_function(self):
        return False

    def is_var(self):
        return True

    def substitute(self, substitutions):
        """If there is a substitution corresponding to this variable,
        return the substituted category.
        """
        for (var, cat) in substitutions:
            if var == self:
                return cat
        return self

    def can_unify(self, other):
        """If the variable can be replaced with other
        a substitution is returned.
        """
        if other.is_primitive() or not self._prim_only:
            return [(self, other)]
        return None

    def id(self):
        return self._id

    def __str__(self):
        return "_var" + str(self._id)


@total_ordering
class Direction:
    """
    Class representing the direction of a function application.
    Also contains maintains information as to which combinators
    may be used with the category.
    """

    def __init__(self, dir, restrictions):
        self._dir = dir
        self._restrs = restrictions
        self._comparison_key = (dir, tuple(restrictions))

    # Testing the application direction
    def is_forward(self):
        return self._dir == "/"

    def is_backward(self):
        return self._dir == "\\"

    def dir(self):
        return self._dir

    def restrs(self):
        """A list of restrictions on the combinators.
        '.' denotes that permuting operations are disallowed
        ',' denotes that function composition is disallowed
        '_' denotes that the direction has variable restrictions.
        (This is redundant in the current implementation of type-raising)
        """
        return self._restrs

    def is_variable(self):
        return self._restrs == "_"

    # Unification and substitution of variable directions.
    # Used only if type-raising is implemented as a unary rule, as it
    # must inherit restrictions from the argument category.
    def can_unify(self, other):
        if other.is_variable():
            return [("_", self.restrs())]
        elif self.is_variable():
            return [("_", other.restrs())]
        else:
            if self.restrs() == other.restrs():
                return []
        return None

    def substitute(self, subs):
        if not self.is_variable():
            return self

        for (var, restrs) in subs:
            if var == "_":
                return Direction(self._dir, restrs)
        return self

    # Testing permitted combinators
    def can_compose(self):
        return "," not in self._restrs

    def can_cross(self):
        return "." not in self._restrs

    def __eq__(self, other):
        return (
            self.__class__ is other.__class__
            and self._comparison_key == other._comparison_key
        )

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, Direction):
            raise_unorderable_types("<", self, other)
        if self.__class__ is other.__class__:
            return self._comparison_key < other._comparison_key
        else:
            return self.__class__.__name__ < other.__class__.__name__

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self._comparison_key)
            return self._hash

    def __str__(self):
        r_str = ""
        for r in self._restrs:
            r_str = r_str + "%s" % r
        return f"{self._dir}{r_str}"

    # The negation operator reverses the direction of the application
    def __neg__(self):
        if self._dir == "/":
            return Direction("\\", self._restrs)
        else:
            return Direction("/", self._restrs)


class PrimitiveCategory(AbstractCCGCategory):
    """
    Class representing primitive categories.
    Takes a string representation of the category, and a
    list of strings specifying the morphological subcategories.
    """

    def __init__(self, categ, restrictions=[]):
        self._categ = categ
        self._restrs = restrictions
        self._comparison_key = (categ, tuple(restrictions))

    def is_primitive(self):
        return True

    def is_function(self):
        return False

    def is_var(self):
        return False

    def restrs(self):
        return self._restrs

    def categ(self):
        return self._categ

    # Substitution does nothing to a primitive category
    def substitute(self, subs):
        return self

    # A primitive can be unified with a class of the same
    # base category, given that the other category shares all
    # of its subclasses, or with a variable.
    def can_unify(self, other):
        if not other.is_primitive():
            return None
        if other.is_var():
            return [(other, self)]
        if other.categ() == self.categ():
            for restr in self._restrs:
                if restr not in other.restrs():
                    return None
            return []
        return None

    def __str__(self):
        if self._restrs == []:
            return "%s" % self._categ
        restrictions = "[%s]" % ",".join(repr(r) for r in self._restrs)
        return f"{self._categ}{restrictions}"


class FunctionalCategory(AbstractCCGCategory):
    """
    Class that represents a function application category.
    Consists of argument and result categories, together with
    an application direction.
    """

    def __init__(self, res, arg, dir):
        self._res = res
        self._arg = arg
        self._dir = dir
        self._comparison_key = (arg, dir, res)

    def is_primitive(self):
        return False

    def is_function(self):
        return True

    def is_var(self):
        return False

    # Substitution returns the category consisting of the
    # substitution applied to each of its constituents.
    def substitute(self, subs):
        sub_res = self._res.substitute(subs)
        sub_dir = self._dir.substitute(subs)
        sub_arg = self._arg.substitute(subs)
        return FunctionalCategory(sub_res, sub_arg, self._dir)

    # A function can unify with another function, so long as its
    # constituents can unify, or with an unrestricted variable.
    def can_unify(self, other):
        if other.is_var():
            return [(other, self)]
        if other.is_function():
            sa = self._res.can_unify(other.res())
            sd = self._dir.can_unify(other.dir())
            if sa is not None and sd is not None:
                sb = self._arg.substitute(sa).can_unify(other.arg().substitute(sa))
                if sb is not None:
                    return sa + sb
        return None

    # Constituent accessors
    def arg(self):
        return self._arg

    def res(self):
        return self._res

    def dir(self):
        return self._dir

    def __str__(self):
        return f"({self._res}{self._dir}{self._arg})"
