# Natural Language Toolkit: Linear Logic
#
# Author: Dan Garrette <dhgarrette@gmail.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from nltk.internals import Counter
from nltk.sem.logic import APP, LogicParser

_counter = Counter()


class Tokens:
    # Punctuation
    OPEN = "("
    CLOSE = ")"

    # Operations
    IMP = "-o"

    PUNCT = [OPEN, CLOSE]
    TOKENS = PUNCT + [IMP]


class LinearLogicParser(LogicParser):
    """A linear logic expression parser."""

    def __init__(self):
        LogicParser.__init__(self)

        self.operator_precedence = {APP: 1, Tokens.IMP: 2, None: 3}
        self.right_associated_operations += [Tokens.IMP]

    def get_all_symbols(self):
        return Tokens.TOKENS

    def handle(self, tok, context):
        if tok not in Tokens.TOKENS:
            return self.handle_variable(tok, context)
        elif tok == Tokens.OPEN:
            return self.handle_open(tok, context)

    def get_BooleanExpression_factory(self, tok):
        if tok == Tokens.IMP:
            return ImpExpression
        else:
            return None

    def make_BooleanExpression(self, factory, first, second):
        return factory(first, second)

    def attempt_ApplicationExpression(self, expression, context):
        """Attempt to make an application expression.  If the next tokens
        are an argument in parens, then the argument expression is a
        function being applied to the arguments.  Otherwise, return the
        argument expression."""
        if self.has_priority(APP, context):
            if self.inRange(0) and self.token(0) == Tokens.OPEN:
                self.token()  # swallow then open paren
                argument = self.process_next_expression(APP)
                self.assertNextToken(Tokens.CLOSE)
                expression = ApplicationExpression(expression, argument, None)
        return expression

    def make_VariableExpression(self, name):
        if name[0].isupper():
            return VariableExpression(name)
        else:
            return ConstantExpression(name)


class Expression:

    _linear_logic_parser = LinearLogicParser()

    @classmethod
    def fromstring(cls, s):
        return cls._linear_logic_parser.parse(s)

    def applyto(self, other, other_indices=None):
        return ApplicationExpression(self, other, other_indices)

    def __call__(self, other):
        return self.applyto(other)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"


class AtomicExpression(Expression):
    def __init__(self, name, dependencies=None):
        """
        :param name: str for the constant name
        :param dependencies: list of int for the indices on which this atom is dependent
        """
        assert isinstance(name, str)
        self.name = name

        if not dependencies:
            dependencies = []
        self.dependencies = dependencies

    def simplify(self, bindings=None):
        """
        If 'self' is bound by 'bindings', return the atomic to which it is bound.
        Otherwise, return self.

        :param bindings: ``BindingDict`` A dictionary of bindings used to simplify
        :return: ``AtomicExpression``
        """
        if bindings and self in bindings:
            return bindings[self]
        else:
            return self

    def compile_pos(self, index_counter, glueFormulaFactory):
        """
        From Iddo Lev's PhD Dissertation p108-109

        :param index_counter: ``Counter`` for unique indices
        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas
        :return: (``Expression``,set) for the compiled linear logic and any newly created glue formulas
        """
        self.dependencies = []
        return (self, [])

    def compile_neg(self, index_counter, glueFormulaFactory):
        """
        From Iddo Lev's PhD Dissertation p108-109

        :param index_counter: ``Counter`` for unique indices
        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas
        :return: (``Expression``,set) for the compiled linear logic and any newly created glue formulas
        """
        self.dependencies = []
        return (self, [])

    def initialize_labels(self, fstruct):
        self.name = fstruct.initialize_label(self.name.lower())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        accum = self.name
        if self.dependencies:
            accum += "%s" % self.dependencies
        return accum

    def __hash__(self):
        return hash(self.name)


class ConstantExpression(AtomicExpression):
    def unify(self, other, bindings):
        """
        If 'other' is a constant, then it must be equal to 'self'.  If 'other' is a variable,
        then it must not be bound to anything other than 'self'.

        :param other: ``Expression``
        :param bindings: ``BindingDict`` A dictionary of all current bindings
        :return: ``BindingDict`` A new combined dictionary of of 'bindings' and any new binding
        :raise UnificationException: If 'self' and 'other' cannot be unified in the context of 'bindings'
        """
        assert isinstance(other, Expression)
        if isinstance(other, VariableExpression):
            try:
                return bindings + BindingDict([(other, self)])
            except VariableBindingException:
                pass
        elif self == other:
            return bindings
        raise UnificationException(self, other, bindings)


class VariableExpression(AtomicExpression):
    def unify(self, other, bindings):
        """
        'self' must not be bound to anything other than 'other'.

        :param other: ``Expression``
        :param bindings: ``BindingDict`` A dictionary of all current bindings
        :return: ``BindingDict`` A new combined dictionary of of 'bindings' and the new binding
        :raise UnificationException: If 'self' and 'other' cannot be unified in the context of 'bindings'
        """
        assert isinstance(other, Expression)
        try:
            if self == other:
                return bindings
            else:
                return bindings + BindingDict([(self, other)])
        except VariableBindingException as e:
            raise UnificationException(self, other, bindings) from e


class ImpExpression(Expression):
    def __init__(self, antecedent, consequent):
        """
        :param antecedent: ``Expression`` for the antecedent
        :param consequent: ``Expression`` for the consequent
        """
        assert isinstance(antecedent, Expression)
        assert isinstance(consequent, Expression)
        self.antecedent = antecedent
        self.consequent = consequent

    def simplify(self, bindings=None):
        return self.__class__(
            self.antecedent.simplify(bindings), self.consequent.simplify(bindings)
        )

    def unify(self, other, bindings):
        """
        Both the antecedent and consequent of 'self' and 'other' must unify.

        :param other: ``ImpExpression``
        :param bindings: ``BindingDict`` A dictionary of all current bindings
        :return: ``BindingDict`` A new combined dictionary of of 'bindings' and any new bindings
        :raise UnificationException: If 'self' and 'other' cannot be unified in the context of 'bindings'
        """
        assert isinstance(other, ImpExpression)
        try:
            return (
                bindings
                + self.antecedent.unify(other.antecedent, bindings)
                + self.consequent.unify(other.consequent, bindings)
            )
        except VariableBindingException as e:
            raise UnificationException(self, other, bindings) from e

    def compile_pos(self, index_counter, glueFormulaFactory):
        """
        From Iddo Lev's PhD Dissertation p108-109

        :param index_counter: ``Counter`` for unique indices
        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas
        :return: (``Expression``,set) for the compiled linear logic and any newly created glue formulas
        """
        (a, a_new) = self.antecedent.compile_neg(index_counter, glueFormulaFactory)
        (c, c_new) = self.consequent.compile_pos(index_counter, glueFormulaFactory)
        return (ImpExpression(a, c), a_new + c_new)

    def compile_neg(self, index_counter, glueFormulaFactory):
        """
        From Iddo Lev's PhD Dissertation p108-109

        :param index_counter: ``Counter`` for unique indices
        :param glueFormulaFactory: ``GlueFormula`` for creating new glue formulas
        :return: (``Expression``,list of ``GlueFormula``) for the compiled linear logic and any newly created glue formulas
        """
        (a, a_new) = self.antecedent.compile_pos(index_counter, glueFormulaFactory)
        (c, c_new) = self.consequent.compile_neg(index_counter, glueFormulaFactory)
        fresh_index = index_counter.get()
        c.dependencies.append(fresh_index)
        new_v = glueFormulaFactory("v%s" % fresh_index, a, {fresh_index})
        return (c, a_new + c_new + [new_v])

    def initialize_labels(self, fstruct):
        self.antecedent.initialize_labels(fstruct)
        self.consequent.initialize_labels(fstruct)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.antecedent == other.antecedent
            and self.consequent == other.consequent
        )

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return "{}{} {} {}{}".format(
            Tokens.OPEN,
            self.antecedent,
            Tokens.IMP,
            self.consequent,
            Tokens.CLOSE,
        )

    def __hash__(self):
        return hash(f"{hash(self.antecedent)}{Tokens.IMP}{hash(self.consequent)}")


class ApplicationExpression(Expression):
    def __init__(self, function, argument, argument_indices=None):
        """
        :param function: ``Expression`` for the function
        :param argument: ``Expression`` for the argument
        :param argument_indices: set for the indices of the glue formula from which the argument came
        :raise LinearLogicApplicationException: If 'function' cannot be applied to 'argument' given 'argument_indices'.
        """
        function_simp = function.simplify()
        argument_simp = argument.simplify()

        assert isinstance(function_simp, ImpExpression)
        assert isinstance(argument_simp, Expression)

        bindings = BindingDict()

        try:
            if isinstance(function, ApplicationExpression):
                bindings += function.bindings
            if isinstance(argument, ApplicationExpression):
                bindings += argument.bindings
            bindings += function_simp.antecedent.unify(argument_simp, bindings)
        except UnificationException as e:
            raise LinearLogicApplicationException(
                f"Cannot apply {function_simp} to {argument_simp}. {e}"
            ) from e

        # If you are running it on complied premises, more conditions apply
        if argument_indices:
            # A.dependencies of (A -o (B -o C)) must be a proper subset of argument_indices
            if not set(function_simp.antecedent.dependencies) < argument_indices:
                raise LinearLogicApplicationException(
                    "Dependencies unfulfilled when attempting to apply Linear Logic formula %s to %s"
                    % (function_simp, argument_simp)
                )
            if set(function_simp.antecedent.dependencies) == argument_indices:
                raise LinearLogicApplicationException(
                    "Dependencies not a proper subset of indices when attempting to apply Linear Logic formula %s to %s"
                    % (function_simp, argument_simp)
                )

        self.function = function
        self.argument = argument
        self.bindings = bindings

    def simplify(self, bindings=None):
        """
        Since function is an implication, return its consequent.  There should be
        no need to check that the application is valid since the checking is done
        by the constructor.

        :param bindings: ``BindingDict`` A dictionary of bindings used to simplify
        :return: ``Expression``
        """
        if not bindings:
            bindings = self.bindings

        return self.function.simplify(bindings).consequent

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.function == other.function
            and self.argument == other.argument
        )

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return "%s" % self.function + Tokens.OPEN + "%s" % self.argument + Tokens.CLOSE

    def __hash__(self):
        return hash(f"{hash(self.antecedent)}{Tokens.OPEN}{hash(self.consequent)}")


class BindingDict:
    def __init__(self, bindings=None):
        """
        :param bindings:
            list [(``VariableExpression``, ``AtomicExpression``)] to initialize the dictionary
            dict {``VariableExpression``: ``AtomicExpression``} to initialize the dictionary
        """
        self.d = {}

        if isinstance(bindings, dict):
            bindings = bindings.items()

        if bindings:
            for (v, b) in bindings:
                self[v] = b

    def __setitem__(self, variable, binding):
        """
        A binding is consistent with the dict if its variable is not already bound, OR if its
        variable is already bound to its argument.

        :param variable: ``VariableExpression`` The variable bind
        :param binding: ``Expression`` The expression to which 'variable' should be bound
        :raise VariableBindingException: If the variable cannot be bound in this dictionary
        """
        assert isinstance(variable, VariableExpression)
        assert isinstance(binding, Expression)

        assert variable != binding

        existing = self.d.get(variable, None)

        if not existing or binding == existing:
            self.d[variable] = binding
        else:
            raise VariableBindingException(
                "Variable %s already bound to another value" % (variable)
            )

    def __getitem__(self, variable):
        """
        Return the expression to which 'variable' is bound
        """
        assert isinstance(variable, VariableExpression)

        intermediate = self.d[variable]
        while intermediate:
            try:
                intermediate = self.d[intermediate]
            except KeyError:
                return intermediate

    def __contains__(self, item):
        return item in self.d

    def __add__(self, other):
        """
        :param other: ``BindingDict`` The dict with which to combine self
        :return: ``BindingDict`` A new dict containing all the elements of both parameters
        :raise VariableBindingException: If the parameter dictionaries are not consistent with each other
        """
        try:
            combined = BindingDict()
            for v in self.d:
                combined[v] = self.d[v]
            for v in other.d:
                combined[v] = other.d[v]
            return combined
        except VariableBindingException as e:
            raise VariableBindingException(
                "Attempting to add two contradicting"
                " VariableBindingsLists: %s, %s" % (self, other)
            ) from e

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        if not isinstance(other, BindingDict):
            raise TypeError
        return self.d == other.d

    def __str__(self):
        return "{" + ", ".join(f"{v}: {self.d[v]}" for v in sorted(self.d.keys())) + "}"

    def __repr__(self):
        return "BindingDict: %s" % self


class VariableBindingException(Exception):
    pass


class UnificationException(Exception):
    def __init__(self, a, b, bindings):
        Exception.__init__(self, f"Cannot unify {a} with {b} given {bindings}")


class LinearLogicApplicationException(Exception):
    pass


def demo():
    lexpr = Expression.fromstring

    print(lexpr(r"f"))
    print(lexpr(r"(g -o f)"))
    print(lexpr(r"((g -o G) -o G)"))
    print(lexpr(r"g -o h -o f"))
    print(lexpr(r"(g -o f)(g)").simplify())
    print(lexpr(r"(H -o f)(g)").simplify())
    print(lexpr(r"((g -o G) -o G)((g -o f))").simplify())
    print(lexpr(r"(H -o H)((g -o f))").simplify())


if __name__ == "__main__":
    demo()
