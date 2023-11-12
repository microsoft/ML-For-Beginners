# Natural Language Toolkit: Discourse Representation Theory (DRT)
#
# Author: Dan Garrette <dhgarrette@gmail.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import operator
from functools import reduce
from itertools import chain

from nltk.sem.logic import (
    APP,
    AbstractVariableExpression,
    AllExpression,
    AndExpression,
    ApplicationExpression,
    BinaryExpression,
    BooleanExpression,
    ConstantExpression,
    EqualityExpression,
    EventVariableExpression,
    ExistsExpression,
    Expression,
    FunctionVariableExpression,
    ImpExpression,
    IndividualVariableExpression,
    LambdaExpression,
    LogicParser,
    NegatedExpression,
    OrExpression,
    Tokens,
    Variable,
    is_eventvar,
    is_funcvar,
    is_indvar,
    unique_variable,
)

# Import Tkinter-based modules if they are available
try:
    from tkinter import Canvas, Tk
    from tkinter.font import Font

    from nltk.util import in_idle

except ImportError:
    # No need to print a warning here, nltk.draw has already printed one.
    pass


class DrtTokens(Tokens):
    DRS = "DRS"
    DRS_CONC = "+"
    PRONOUN = "PRO"
    OPEN_BRACKET = "["
    CLOSE_BRACKET = "]"
    COLON = ":"

    PUNCT = [DRS_CONC, OPEN_BRACKET, CLOSE_BRACKET, COLON]

    SYMBOLS = Tokens.SYMBOLS + PUNCT

    TOKENS = Tokens.TOKENS + [DRS] + PUNCT


class DrtParser(LogicParser):
    """A lambda calculus expression parser."""

    def __init__(self):
        LogicParser.__init__(self)

        self.operator_precedence = dict(
            [(x, 1) for x in DrtTokens.LAMBDA_LIST]
            + [(x, 2) for x in DrtTokens.NOT_LIST]
            + [(APP, 3)]
            + [(x, 4) for x in DrtTokens.EQ_LIST + Tokens.NEQ_LIST]
            + [(DrtTokens.COLON, 5)]
            + [(DrtTokens.DRS_CONC, 6)]
            + [(x, 7) for x in DrtTokens.OR_LIST]
            + [(x, 8) for x in DrtTokens.IMP_LIST]
            + [(None, 9)]
        )

    def get_all_symbols(self):
        """This method exists to be overridden"""
        return DrtTokens.SYMBOLS

    def isvariable(self, tok):
        return tok not in DrtTokens.TOKENS

    def handle(self, tok, context):
        """This method is intended to be overridden for logics that
        use different operators or expressions"""
        if tok in DrtTokens.NOT_LIST:
            return self.handle_negation(tok, context)

        elif tok in DrtTokens.LAMBDA_LIST:
            return self.handle_lambda(tok, context)

        elif tok == DrtTokens.OPEN:
            if self.inRange(0) and self.token(0) == DrtTokens.OPEN_BRACKET:
                return self.handle_DRS(tok, context)
            else:
                return self.handle_open(tok, context)

        elif tok.upper() == DrtTokens.DRS:
            self.assertNextToken(DrtTokens.OPEN)
            return self.handle_DRS(tok, context)

        elif self.isvariable(tok):
            if self.inRange(0) and self.token(0) == DrtTokens.COLON:
                return self.handle_prop(tok, context)
            else:
                return self.handle_variable(tok, context)

    def make_NegatedExpression(self, expression):
        return DrtNegatedExpression(expression)

    def handle_DRS(self, tok, context):
        # a DRS
        refs = self.handle_refs()
        if (
            self.inRange(0) and self.token(0) == DrtTokens.COMMA
        ):  # if there is a comma (it's optional)
            self.token()  # swallow the comma
        conds = self.handle_conds(context)
        self.assertNextToken(DrtTokens.CLOSE)
        return DRS(refs, conds, None)

    def handle_refs(self):
        self.assertNextToken(DrtTokens.OPEN_BRACKET)
        refs = []
        while self.inRange(0) and self.token(0) != DrtTokens.CLOSE_BRACKET:
            # Support expressions like: DRS([x y],C) == DRS([x,y],C)
            if refs and self.token(0) == DrtTokens.COMMA:
                self.token()  # swallow the comma
            refs.append(self.get_next_token_variable("quantified"))
        self.assertNextToken(DrtTokens.CLOSE_BRACKET)
        return refs

    def handle_conds(self, context):
        self.assertNextToken(DrtTokens.OPEN_BRACKET)
        conds = []
        while self.inRange(0) and self.token(0) != DrtTokens.CLOSE_BRACKET:
            # Support expressions like: DRS([x y],C) == DRS([x, y],C)
            if conds and self.token(0) == DrtTokens.COMMA:
                self.token()  # swallow the comma
            conds.append(self.process_next_expression(context))
        self.assertNextToken(DrtTokens.CLOSE_BRACKET)
        return conds

    def handle_prop(self, tok, context):
        variable = self.make_VariableExpression(tok)
        self.assertNextToken(":")
        drs = self.process_next_expression(DrtTokens.COLON)
        return DrtProposition(variable, drs)

    def make_EqualityExpression(self, first, second):
        """This method serves as a hook for other logic parsers that
        have different equality expression classes"""
        return DrtEqualityExpression(first, second)

    def get_BooleanExpression_factory(self, tok):
        """This method serves as a hook for other logic parsers that
        have different boolean operators"""
        if tok == DrtTokens.DRS_CONC:
            return lambda first, second: DrtConcatenation(first, second, None)
        elif tok in DrtTokens.OR_LIST:
            return DrtOrExpression
        elif tok in DrtTokens.IMP_LIST:

            def make_imp_expression(first, second):
                if isinstance(first, DRS):
                    return DRS(first.refs, first.conds, second)
                if isinstance(first, DrtConcatenation):
                    return DrtConcatenation(first.first, first.second, second)
                raise Exception("Antecedent of implication must be a DRS")

            return make_imp_expression
        else:
            return None

    def make_BooleanExpression(self, factory, first, second):
        return factory(first, second)

    def make_ApplicationExpression(self, function, argument):
        return DrtApplicationExpression(function, argument)

    def make_VariableExpression(self, name):
        return DrtVariableExpression(Variable(name))

    def make_LambdaExpression(self, variables, term):
        return DrtLambdaExpression(variables, term)


class DrtExpression:
    """
    This is the base abstract DRT Expression from which every DRT
    Expression extends.
    """

    _drt_parser = DrtParser()

    @classmethod
    def fromstring(cls, s):
        return cls._drt_parser.parse(s)

    def applyto(self, other):
        return DrtApplicationExpression(self, other)

    def __neg__(self):
        return DrtNegatedExpression(self)

    def __and__(self, other):
        return NotImplemented

    def __or__(self, other):
        assert isinstance(other, DrtExpression)
        return DrtOrExpression(self, other)

    def __gt__(self, other):
        assert isinstance(other, DrtExpression)
        if isinstance(self, DRS):
            return DRS(self.refs, self.conds, other)
        if isinstance(self, DrtConcatenation):
            return DrtConcatenation(self.first, self.second, other)
        raise Exception("Antecedent of implication must be a DRS")

    def equiv(self, other, prover=None):
        """
        Check for logical equivalence.
        Pass the expression (self <-> other) to the theorem prover.
        If the prover says it is valid, then the self and other are equal.

        :param other: an ``DrtExpression`` to check equality against
        :param prover: a ``nltk.inference.api.Prover``
        """
        assert isinstance(other, DrtExpression)

        f1 = self.simplify().fol()
        f2 = other.simplify().fol()
        return f1.equiv(f2, prover)

    @property
    def type(self):
        raise AttributeError(
            "'%s' object has no attribute 'type'" % self.__class__.__name__
        )

    def typecheck(self, signature=None):
        raise NotImplementedError()

    def __add__(self, other):
        return DrtConcatenation(self, other, None)

    def get_refs(self, recursive=False):
        """
        Return the set of discourse referents in this DRS.
        :param recursive: bool Also find discourse referents in subterms?
        :return: list of ``Variable`` objects
        """
        raise NotImplementedError()

    def is_pronoun_function(self):
        """Is self of the form "PRO(x)"?"""
        return (
            isinstance(self, DrtApplicationExpression)
            and isinstance(self.function, DrtAbstractVariableExpression)
            and self.function.variable.name == DrtTokens.PRONOUN
            and isinstance(self.argument, DrtIndividualVariableExpression)
        )

    def make_EqualityExpression(self, first, second):
        return DrtEqualityExpression(first, second)

    def make_VariableExpression(self, variable):
        return DrtVariableExpression(variable)

    def resolve_anaphora(self):
        return resolve_anaphora(self)

    def eliminate_equality(self):
        return self.visit_structured(lambda e: e.eliminate_equality(), self.__class__)

    def pretty_format(self):
        """
        Draw the DRS
        :return: the pretty print string
        """
        return "\n".join(self._pretty())

    def pretty_print(self):
        print(self.pretty_format())

    def draw(self):
        DrsDrawer(self).draw()


class DRS(DrtExpression, Expression):
    """A Discourse Representation Structure."""

    def __init__(self, refs, conds, consequent=None):
        """
        :param refs: list of ``DrtIndividualVariableExpression`` for the
            discourse referents
        :param conds: list of ``Expression`` for the conditions
        """
        self.refs = refs
        self.conds = conds
        self.consequent = consequent

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """Replace all instances of variable v with expression E in self,
        where v is free in self."""
        if variable in self.refs:
            # if a bound variable is the thing being replaced
            if not replace_bound:
                return self
            else:
                i = self.refs.index(variable)
                if self.consequent:
                    consequent = self.consequent.replace(
                        variable, expression, True, alpha_convert
                    )
                else:
                    consequent = None
                return DRS(
                    self.refs[:i] + [expression.variable] + self.refs[i + 1 :],
                    [
                        cond.replace(variable, expression, True, alpha_convert)
                        for cond in self.conds
                    ],
                    consequent,
                )
        else:
            if alpha_convert:
                # any bound variable that appears in the expression must
                # be alpha converted to avoid a conflict
                for ref in set(self.refs) & expression.free():
                    newvar = unique_variable(ref)
                    newvarex = DrtVariableExpression(newvar)
                    i = self.refs.index(ref)
                    if self.consequent:
                        consequent = self.consequent.replace(
                            ref, newvarex, True, alpha_convert
                        )
                    else:
                        consequent = None
                    self = DRS(
                        self.refs[:i] + [newvar] + self.refs[i + 1 :],
                        [
                            cond.replace(ref, newvarex, True, alpha_convert)
                            for cond in self.conds
                        ],
                        consequent,
                    )

            # replace in the conditions
            if self.consequent:
                consequent = self.consequent.replace(
                    variable, expression, replace_bound, alpha_convert
                )
            else:
                consequent = None
            return DRS(
                self.refs,
                [
                    cond.replace(variable, expression, replace_bound, alpha_convert)
                    for cond in self.conds
                ],
                consequent,
            )

    def free(self):
        """:see: Expression.free()"""
        conds_free = reduce(operator.or_, [c.free() for c in self.conds], set())
        if self.consequent:
            conds_free.update(self.consequent.free())
        return conds_free - set(self.refs)

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        if recursive:
            conds_refs = self.refs + list(
                chain.from_iterable(c.get_refs(True) for c in self.conds)
            )
            if self.consequent:
                conds_refs.extend(self.consequent.get_refs(True))
            return conds_refs
        else:
            return self.refs

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        parts = list(map(function, self.conds))
        if self.consequent:
            parts.append(function(self.consequent))
        return combinator(parts)

    def visit_structured(self, function, combinator):
        """:see: Expression.visit_structured()"""
        consequent = function(self.consequent) if self.consequent else None
        return combinator(self.refs, list(map(function, self.conds)), consequent)

    def eliminate_equality(self):
        drs = self
        i = 0
        while i < len(drs.conds):
            cond = drs.conds[i]
            if (
                isinstance(cond, EqualityExpression)
                and isinstance(cond.first, AbstractVariableExpression)
                and isinstance(cond.second, AbstractVariableExpression)
            ):
                drs = DRS(
                    list(set(drs.refs) - {cond.second.variable}),
                    drs.conds[:i] + drs.conds[i + 1 :],
                    drs.consequent,
                )
                if cond.second.variable != cond.first.variable:
                    drs = drs.replace(cond.second.variable, cond.first, False, False)
                    i = 0
                i -= 1
            i += 1

        conds = []
        for cond in drs.conds:
            new_cond = cond.eliminate_equality()
            new_cond_simp = new_cond.simplify()
            if (
                not isinstance(new_cond_simp, DRS)
                or new_cond_simp.refs
                or new_cond_simp.conds
                or new_cond_simp.consequent
            ):
                conds.append(new_cond)

        consequent = drs.consequent.eliminate_equality() if drs.consequent else None
        return DRS(drs.refs, conds, consequent)

    def fol(self):
        if self.consequent:
            accum = None
            if self.conds:
                accum = reduce(AndExpression, [c.fol() for c in self.conds])

            if accum:
                accum = ImpExpression(accum, self.consequent.fol())
            else:
                accum = self.consequent.fol()

            for ref in self.refs[::-1]:
                accum = AllExpression(ref, accum)

            return accum

        else:
            if not self.conds:
                raise Exception("Cannot convert DRS with no conditions to FOL.")
            accum = reduce(AndExpression, [c.fol() for c in self.conds])
            for ref in map(Variable, self._order_ref_strings(self.refs)[::-1]):
                accum = ExistsExpression(ref, accum)
            return accum

    def _pretty(self):
        refs_line = " ".join(self._order_ref_strings(self.refs))

        cond_lines = [
            cond
            for cond_line in [
                filter(lambda s: s.strip(), cond._pretty()) for cond in self.conds
            ]
            for cond in cond_line
        ]
        length = max([len(refs_line)] + list(map(len, cond_lines)))
        drs = (
            [
                " _" + "_" * length + "_ ",
                "| " + refs_line.ljust(length) + " |",
                "|-" + "-" * length + "-|",
            ]
            + ["| " + line.ljust(length) + " |" for line in cond_lines]
            + ["|_" + "_" * length + "_|"]
        )
        if self.consequent:
            return DrtBinaryExpression._assemble_pretty(
                drs, DrtTokens.IMP, self.consequent._pretty()
            )
        return drs

    def _order_ref_strings(self, refs):
        strings = ["%s" % ref for ref in refs]
        ind_vars = []
        func_vars = []
        event_vars = []
        other_vars = []
        for s in strings:
            if is_indvar(s):
                ind_vars.append(s)
            elif is_funcvar(s):
                func_vars.append(s)
            elif is_eventvar(s):
                event_vars.append(s)
            else:
                other_vars.append(s)
        return (
            sorted(other_vars)
            + sorted(event_vars, key=lambda v: int([v[2:], -1][len(v[2:]) == 0]))
            + sorted(func_vars, key=lambda v: (v[0], int([v[1:], -1][len(v[1:]) == 0])))
            + sorted(ind_vars, key=lambda v: (v[0], int([v[1:], -1][len(v[1:]) == 0])))
        )

    def __eq__(self, other):
        r"""Defines equality modulo alphabetic variance.
        If we are comparing \x.M  and \y.N, then check equality of M and N[x/y]."""
        if isinstance(other, DRS):
            if len(self.refs) == len(other.refs):
                converted_other = other
                for (r1, r2) in zip(self.refs, converted_other.refs):
                    varex = self.make_VariableExpression(r1)
                    converted_other = converted_other.replace(r2, varex, True)
                if self.consequent == converted_other.consequent and len(
                    self.conds
                ) == len(converted_other.conds):
                    for c1, c2 in zip(self.conds, converted_other.conds):
                        if not (c1 == c2):
                            return False
                    return True
        return False

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__

    def __str__(self):
        drs = "([{}],[{}])".format(
            ",".join(self._order_ref_strings(self.refs)),
            ", ".join("%s" % cond for cond in self.conds),
        )  # map(str, self.conds)))
        if self.consequent:
            return (
                DrtTokens.OPEN
                + drs
                + " "
                + DrtTokens.IMP
                + " "
                + "%s" % self.consequent
                + DrtTokens.CLOSE
            )
        return drs


def DrtVariableExpression(variable):
    """
    This is a factory method that instantiates and returns a subtype of
    ``DrtAbstractVariableExpression`` appropriate for the given variable.
    """
    if is_indvar(variable.name):
        return DrtIndividualVariableExpression(variable)
    elif is_funcvar(variable.name):
        return DrtFunctionVariableExpression(variable)
    elif is_eventvar(variable.name):
        return DrtEventVariableExpression(variable)
    else:
        return DrtConstantExpression(variable)


class DrtAbstractVariableExpression(DrtExpression, AbstractVariableExpression):
    def fol(self):
        return self

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return []

    def _pretty(self):
        s = "%s" % self
        blank = " " * len(s)
        return [blank, blank, s, blank]

    def eliminate_equality(self):
        return self


class DrtIndividualVariableExpression(
    DrtAbstractVariableExpression, IndividualVariableExpression
):
    pass


class DrtFunctionVariableExpression(
    DrtAbstractVariableExpression, FunctionVariableExpression
):
    pass


class DrtEventVariableExpression(
    DrtIndividualVariableExpression, EventVariableExpression
):
    pass


class DrtConstantExpression(DrtAbstractVariableExpression, ConstantExpression):
    pass


class DrtProposition(DrtExpression, Expression):
    def __init__(self, variable, drs):
        self.variable = variable
        self.drs = drs

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        if self.variable == variable:
            assert isinstance(
                expression, DrtAbstractVariableExpression
            ), "Can only replace a proposition label with a variable"
            return DrtProposition(
                expression.variable,
                self.drs.replace(variable, expression, replace_bound, alpha_convert),
            )
        else:
            return DrtProposition(
                self.variable,
                self.drs.replace(variable, expression, replace_bound, alpha_convert),
            )

    def eliminate_equality(self):
        return DrtProposition(self.variable, self.drs.eliminate_equality())

    def get_refs(self, recursive=False):
        return self.drs.get_refs(True) if recursive else []

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.variable == other.variable
            and self.drs == other.drs
        )

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__

    def fol(self):
        return self.drs.fol()

    def _pretty(self):
        drs_s = self.drs._pretty()
        blank = " " * len("%s" % self.variable)
        return (
            [blank + " " + line for line in drs_s[:1]]
            + ["%s" % self.variable + ":" + line for line in drs_s[1:2]]
            + [blank + " " + line for line in drs_s[2:]]
        )

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.drs)])

    def visit_structured(self, function, combinator):
        """:see: Expression.visit_structured()"""
        return combinator(self.variable, function(self.drs))

    def __str__(self):
        return f"prop({self.variable}, {self.drs})"


class DrtNegatedExpression(DrtExpression, NegatedExpression):
    def fol(self):
        return NegatedExpression(self.term.fol())

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return self.term.get_refs(recursive)

    def _pretty(self):
        term_lines = self.term._pretty()
        return (
            ["    " + line for line in term_lines[:2]]
            + ["__  " + line for line in term_lines[2:3]]
            + ["  | " + line for line in term_lines[3:4]]
            + ["    " + line for line in term_lines[4:]]
        )


class DrtLambdaExpression(DrtExpression, LambdaExpression):
    def alpha_convert(self, newvar):
        """Rename all occurrences of the variable introduced by this variable
        binder in the expression to ``newvar``.
        :param newvar: ``Variable``, for the new variable
        """
        return self.__class__(
            newvar,
            self.term.replace(self.variable, DrtVariableExpression(newvar), True),
        )

    def fol(self):
        return LambdaExpression(self.variable, self.term.fol())

    def _pretty(self):
        variables = [self.variable]
        term = self.term
        while term.__class__ == self.__class__:
            variables.append(term.variable)
            term = term.term
        var_string = " ".join("%s" % v for v in variables) + DrtTokens.DOT
        term_lines = term._pretty()
        blank = " " * len(var_string)
        return (
            ["    " + blank + line for line in term_lines[:1]]
            + [r" \  " + blank + line for line in term_lines[1:2]]
            + [r" /\ " + var_string + line for line in term_lines[2:3]]
            + ["    " + blank + line for line in term_lines[3:]]
        )

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return (
            [self.variable] + self.term.get_refs(True) if recursive else [self.variable]
        )


class DrtBinaryExpression(DrtExpression, BinaryExpression):
    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return (
            self.first.get_refs(True) + self.second.get_refs(True) if recursive else []
        )

    def _pretty(self):
        return DrtBinaryExpression._assemble_pretty(
            self._pretty_subex(self.first),
            self.getOp(),
            self._pretty_subex(self.second),
        )

    @staticmethod
    def _assemble_pretty(first_lines, op, second_lines):
        max_lines = max(len(first_lines), len(second_lines))
        first_lines = _pad_vertically(first_lines, max_lines)
        second_lines = _pad_vertically(second_lines, max_lines)
        blank = " " * len(op)
        first_second_lines = list(zip(first_lines, second_lines))
        return (
            [
                " " + first_line + " " + blank + " " + second_line + " "
                for first_line, second_line in first_second_lines[:2]
            ]
            + [
                "(" + first_line + " " + op + " " + second_line + ")"
                for first_line, second_line in first_second_lines[2:3]
            ]
            + [
                " " + first_line + " " + blank + " " + second_line + " "
                for first_line, second_line in first_second_lines[3:]
            ]
        )

    def _pretty_subex(self, subex):
        return subex._pretty()


class DrtBooleanExpression(DrtBinaryExpression, BooleanExpression):
    pass


class DrtOrExpression(DrtBooleanExpression, OrExpression):
    def fol(self):
        return OrExpression(self.first.fol(), self.second.fol())

    def _pretty_subex(self, subex):
        if isinstance(subex, DrtOrExpression):
            return [line[1:-1] for line in subex._pretty()]
        return DrtBooleanExpression._pretty_subex(self, subex)


class DrtEqualityExpression(DrtBinaryExpression, EqualityExpression):
    def fol(self):
        return EqualityExpression(self.first.fol(), self.second.fol())


class DrtConcatenation(DrtBooleanExpression):
    """DRS of the form '(DRS + DRS)'"""

    def __init__(self, first, second, consequent=None):
        DrtBooleanExpression.__init__(self, first, second)
        self.consequent = consequent

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """Replace all instances of variable v with expression E in self,
        where v is free in self."""
        first = self.first
        second = self.second
        consequent = self.consequent

        # If variable is bound
        if variable in self.get_refs():
            if replace_bound:
                first = first.replace(
                    variable, expression, replace_bound, alpha_convert
                )
                second = second.replace(
                    variable, expression, replace_bound, alpha_convert
                )
                if consequent:
                    consequent = consequent.replace(
                        variable, expression, replace_bound, alpha_convert
                    )
        else:
            if alpha_convert:
                # alpha convert every ref that is free in 'expression'
                for ref in set(self.get_refs(True)) & expression.free():
                    v = DrtVariableExpression(unique_variable(ref))
                    first = first.replace(ref, v, True, alpha_convert)
                    second = second.replace(ref, v, True, alpha_convert)
                    if consequent:
                        consequent = consequent.replace(ref, v, True, alpha_convert)

            first = first.replace(variable, expression, replace_bound, alpha_convert)
            second = second.replace(variable, expression, replace_bound, alpha_convert)
            if consequent:
                consequent = consequent.replace(
                    variable, expression, replace_bound, alpha_convert
                )

        return self.__class__(first, second, consequent)

    def eliminate_equality(self):
        # TODO: at some point.  for now, simplify.
        drs = self.simplify()
        assert not isinstance(drs, DrtConcatenation)
        return drs.eliminate_equality()

    def simplify(self):
        first = self.first.simplify()
        second = self.second.simplify()
        consequent = self.consequent.simplify() if self.consequent else None

        if isinstance(first, DRS) and isinstance(second, DRS):
            # For any ref that is in both 'first' and 'second'
            for ref in set(first.get_refs(True)) & set(second.get_refs(True)):
                # alpha convert the ref in 'second' to prevent collision
                newvar = DrtVariableExpression(unique_variable(ref))
                second = second.replace(ref, newvar, True)

            return DRS(first.refs + second.refs, first.conds + second.conds, consequent)
        else:
            return self.__class__(first, second, consequent)

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        refs = self.first.get_refs(recursive) + self.second.get_refs(recursive)
        if self.consequent and recursive:
            refs.extend(self.consequent.get_refs(True))
        return refs

    def getOp(self):
        return DrtTokens.DRS_CONC

    def __eq__(self, other):
        r"""Defines equality modulo alphabetic variance.
        If we are comparing \x.M  and \y.N, then check equality of M and N[x/y]."""
        if isinstance(other, DrtConcatenation):
            self_refs = self.get_refs()
            other_refs = other.get_refs()
            if len(self_refs) == len(other_refs):
                converted_other = other
                for (r1, r2) in zip(self_refs, other_refs):
                    varex = self.make_VariableExpression(r1)
                    converted_other = converted_other.replace(r2, varex, True)
                return (
                    self.first == converted_other.first
                    and self.second == converted_other.second
                    and self.consequent == converted_other.consequent
                )
        return False

    def __ne__(self, other):
        return not self == other

    __hash__ = DrtBooleanExpression.__hash__

    def fol(self):
        e = AndExpression(self.first.fol(), self.second.fol())
        if self.consequent:
            e = ImpExpression(e, self.consequent.fol())
        return e

    def _pretty(self):
        drs = DrtBinaryExpression._assemble_pretty(
            self._pretty_subex(self.first),
            self.getOp(),
            self._pretty_subex(self.second),
        )
        if self.consequent:
            drs = DrtBinaryExpression._assemble_pretty(
                drs, DrtTokens.IMP, self.consequent._pretty()
            )
        return drs

    def _pretty_subex(self, subex):
        if isinstance(subex, DrtConcatenation):
            return [line[1:-1] for line in subex._pretty()]
        return DrtBooleanExpression._pretty_subex(self, subex)

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        if self.consequent:
            return combinator(
                [function(self.first), function(self.second), function(self.consequent)]
            )
        else:
            return combinator([function(self.first), function(self.second)])

    def __str__(self):
        first = self._str_subex(self.first)
        second = self._str_subex(self.second)
        drs = Tokens.OPEN + first + " " + self.getOp() + " " + second + Tokens.CLOSE
        if self.consequent:
            return (
                DrtTokens.OPEN
                + drs
                + " "
                + DrtTokens.IMP
                + " "
                + "%s" % self.consequent
                + DrtTokens.CLOSE
            )
        return drs

    def _str_subex(self, subex):
        s = "%s" % subex
        if isinstance(subex, DrtConcatenation) and subex.consequent is None:
            return s[1:-1]
        return s


class DrtApplicationExpression(DrtExpression, ApplicationExpression):
    def fol(self):
        return ApplicationExpression(self.function.fol(), self.argument.fol())

    def get_refs(self, recursive=False):
        """:see: AbstractExpression.get_refs()"""
        return (
            self.function.get_refs(True) + self.argument.get_refs(True)
            if recursive
            else []
        )

    def _pretty(self):
        function, args = self.uncurry()
        function_lines = function._pretty()
        args_lines = [arg._pretty() for arg in args]
        max_lines = max(map(len, [function_lines] + args_lines))
        function_lines = _pad_vertically(function_lines, max_lines)
        args_lines = [_pad_vertically(arg_lines, max_lines) for arg_lines in args_lines]
        func_args_lines = list(zip(function_lines, list(zip(*args_lines))))
        return (
            [
                func_line + " " + " ".join(args_line) + " "
                for func_line, args_line in func_args_lines[:2]
            ]
            + [
                func_line + "(" + ",".join(args_line) + ")"
                for func_line, args_line in func_args_lines[2:3]
            ]
            + [
                func_line + " " + " ".join(args_line) + " "
                for func_line, args_line in func_args_lines[3:]
            ]
        )


def _pad_vertically(lines, max_lines):
    pad_line = [" " * len(lines[0])]
    return lines + pad_line * (max_lines - len(lines))


class PossibleAntecedents(list, DrtExpression, Expression):
    def free(self):
        """Set of free variables."""
        return set(self)

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """Replace all instances of variable v with expression E in self,
        where v is free in self."""
        result = PossibleAntecedents()
        for item in self:
            if item == variable:
                self.append(expression)
            else:
                self.append(item)
        return result

    def _pretty(self):
        s = "%s" % self
        blank = " " * len(s)
        return [blank, blank, s]

    def __str__(self):
        return "[" + ",".join("%s" % it for it in self) + "]"


class AnaphoraResolutionException(Exception):
    pass


def resolve_anaphora(expression, trail=[]):
    if isinstance(expression, ApplicationExpression):
        if expression.is_pronoun_function():
            possible_antecedents = PossibleAntecedents()
            for ancestor in trail:
                for ref in ancestor.get_refs():
                    refex = expression.make_VariableExpression(ref)

                    # ==========================================================
                    # Don't allow resolution to itself or other types
                    # ==========================================================
                    if refex.__class__ == expression.argument.__class__ and not (
                        refex == expression.argument
                    ):
                        possible_antecedents.append(refex)

            if len(possible_antecedents) == 1:
                resolution = possible_antecedents[0]
            else:
                resolution = possible_antecedents
            return expression.make_EqualityExpression(expression.argument, resolution)
        else:
            r_function = resolve_anaphora(expression.function, trail + [expression])
            r_argument = resolve_anaphora(expression.argument, trail + [expression])
            return expression.__class__(r_function, r_argument)

    elif isinstance(expression, DRS):
        r_conds = []
        for cond in expression.conds:
            r_cond = resolve_anaphora(cond, trail + [expression])

            # if the condition is of the form '(x = [])' then raise exception
            if isinstance(r_cond, EqualityExpression):
                if isinstance(r_cond.first, PossibleAntecedents):
                    # Reverse the order so that the variable is on the left
                    temp = r_cond.first
                    r_cond.first = r_cond.second
                    r_cond.second = temp
                if isinstance(r_cond.second, PossibleAntecedents):
                    if not r_cond.second:
                        raise AnaphoraResolutionException(
                            "Variable '%s' does not "
                            "resolve to anything." % r_cond.first
                        )

            r_conds.append(r_cond)
        if expression.consequent:
            consequent = resolve_anaphora(expression.consequent, trail + [expression])
        else:
            consequent = None
        return expression.__class__(expression.refs, r_conds, consequent)

    elif isinstance(expression, AbstractVariableExpression):
        return expression

    elif isinstance(expression, NegatedExpression):
        return expression.__class__(
            resolve_anaphora(expression.term, trail + [expression])
        )

    elif isinstance(expression, DrtConcatenation):
        if expression.consequent:
            consequent = resolve_anaphora(expression.consequent, trail + [expression])
        else:
            consequent = None
        return expression.__class__(
            resolve_anaphora(expression.first, trail + [expression]),
            resolve_anaphora(expression.second, trail + [expression]),
            consequent,
        )

    elif isinstance(expression, BinaryExpression):
        return expression.__class__(
            resolve_anaphora(expression.first, trail + [expression]),
            resolve_anaphora(expression.second, trail + [expression]),
        )

    elif isinstance(expression, LambdaExpression):
        return expression.__class__(
            expression.variable, resolve_anaphora(expression.term, trail + [expression])
        )


class DrsDrawer:
    BUFFER = 3  # Space between elements
    TOPSPACE = 10  # Space above whole DRS
    OUTERSPACE = 6  # Space to the left, right, and bottom of the while DRS

    def __init__(self, drs, size_canvas=True, canvas=None):
        """
        :param drs: ``DrtExpression``, The DRS to be drawn
        :param size_canvas: bool, True if the canvas size should be the exact size of the DRS
        :param canvas: ``Canvas`` The canvas on which to draw the DRS.  If none is given, create a new canvas.
        """
        master = None
        if not canvas:
            master = Tk()
            master.title("DRT")

            font = Font(family="helvetica", size=12)

            if size_canvas:
                canvas = Canvas(master, width=0, height=0)
                canvas.font = font
                self.canvas = canvas
                (right, bottom) = self._visit(drs, self.OUTERSPACE, self.TOPSPACE)

                width = max(right + self.OUTERSPACE, 100)
                height = bottom + self.OUTERSPACE
                canvas = Canvas(master, width=width, height=height)  # , bg='white')
            else:
                canvas = Canvas(master, width=300, height=300)

            canvas.pack()
            canvas.font = font

        self.canvas = canvas
        self.drs = drs
        self.master = master

    def _get_text_height(self):
        """Get the height of a line of text"""
        return self.canvas.font.metrics("linespace")

    def draw(self, x=OUTERSPACE, y=TOPSPACE):
        """Draw the DRS"""
        self._handle(self.drs, self._draw_command, x, y)

        if self.master and not in_idle():
            self.master.mainloop()
        else:
            return self._visit(self.drs, x, y)

    def _visit(self, expression, x, y):
        """
        Return the bottom-rightmost point without actually drawing the item

        :param expression: the item to visit
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
        return self._handle(expression, self._visit_command, x, y)

    def _draw_command(self, item, x, y):
        """
        Draw the given item at the given location

        :param item: the item to draw
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
        if isinstance(item, str):
            self.canvas.create_text(x, y, anchor="nw", font=self.canvas.font, text=item)
        elif isinstance(item, tuple):
            # item is the lower-right of a box
            (right, bottom) = item
            self.canvas.create_rectangle(x, y, right, bottom)
            horiz_line_y = (
                y + self._get_text_height() + (self.BUFFER * 2)
            )  # the line separating refs from conds
            self.canvas.create_line(x, horiz_line_y, right, horiz_line_y)

        return self._visit_command(item, x, y)

    def _visit_command(self, item, x, y):
        """
        Return the bottom-rightmost point without actually drawing the item

        :param item: the item to visit
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
        if isinstance(item, str):
            return (x + self.canvas.font.measure(item), y + self._get_text_height())
        elif isinstance(item, tuple):
            return item

    def _handle(self, expression, command, x=0, y=0):
        """
        :param expression: the expression to handle
        :param command: the function to apply, either _draw_command or _visit_command
        :param x: the top of the current drawing area
        :param y: the left side of the current drawing area
        :return: the bottom-rightmost point
        """
        if command == self._visit_command:
            # if we don't need to draw the item, then we can use the cached values
            try:
                # attempt to retrieve cached values
                right = expression._drawing_width + x
                bottom = expression._drawing_height + y
                return (right, bottom)
            except AttributeError:
                # the values have not been cached yet, so compute them
                pass

        if isinstance(expression, DrtAbstractVariableExpression):
            factory = self._handle_VariableExpression
        elif isinstance(expression, DRS):
            factory = self._handle_DRS
        elif isinstance(expression, DrtNegatedExpression):
            factory = self._handle_NegatedExpression
        elif isinstance(expression, DrtLambdaExpression):
            factory = self._handle_LambdaExpression
        elif isinstance(expression, BinaryExpression):
            factory = self._handle_BinaryExpression
        elif isinstance(expression, DrtApplicationExpression):
            factory = self._handle_ApplicationExpression
        elif isinstance(expression, PossibleAntecedents):
            factory = self._handle_VariableExpression
        elif isinstance(expression, DrtProposition):
            factory = self._handle_DrtProposition
        else:
            raise Exception(expression.__class__.__name__)

        (right, bottom) = factory(expression, command, x, y)

        # cache the values
        expression._drawing_width = right - x
        expression._drawing_height = bottom - y

        return (right, bottom)

    def _handle_VariableExpression(self, expression, command, x, y):
        return command("%s" % expression, x, y)

    def _handle_NegatedExpression(self, expression, command, x, y):
        # Find the width of the negation symbol
        right = self._visit_command(DrtTokens.NOT, x, y)[0]

        # Handle term
        (right, bottom) = self._handle(expression.term, command, right, y)

        # Handle variables now that we know the y-coordinate
        command(
            DrtTokens.NOT,
            x,
            self._get_centered_top(y, bottom - y, self._get_text_height()),
        )

        return (right, bottom)

    def _handle_DRS(self, expression, command, x, y):
        left = x + self.BUFFER  # indent the left side
        bottom = y + self.BUFFER  # indent the top

        # Handle Discourse Referents
        if expression.refs:
            refs = " ".join("%s" % r for r in expression.refs)
        else:
            refs = "     "
        (max_right, bottom) = command(refs, left, bottom)
        bottom += self.BUFFER * 2

        # Handle Conditions
        if expression.conds:
            for cond in expression.conds:
                (right, bottom) = self._handle(cond, command, left, bottom)
                max_right = max(max_right, right)
                bottom += self.BUFFER
        else:
            bottom += self._get_text_height() + self.BUFFER

        # Handle Box
        max_right += self.BUFFER
        return command((max_right, bottom), x, y)

    def _handle_ApplicationExpression(self, expression, command, x, y):
        function, args = expression.uncurry()
        if not isinstance(function, DrtAbstractVariableExpression):
            # It's not a predicate expression ("P(x,y)"), so leave arguments curried
            function = expression.function
            args = [expression.argument]

        # Get the max bottom of any element on the line
        function_bottom = self._visit(function, x, y)[1]
        max_bottom = max(
            [function_bottom] + [self._visit(arg, x, y)[1] for arg in args]
        )

        line_height = max_bottom - y

        # Handle 'function'
        function_drawing_top = self._get_centered_top(
            y, line_height, function._drawing_height
        )
        right = self._handle(function, command, x, function_drawing_top)[0]

        # Handle open paren
        centred_string_top = self._get_centered_top(
            y, line_height, self._get_text_height()
        )
        right = command(DrtTokens.OPEN, right, centred_string_top)[0]

        # Handle each arg
        for (i, arg) in enumerate(args):
            arg_drawing_top = self._get_centered_top(
                y, line_height, arg._drawing_height
            )
            right = self._handle(arg, command, right, arg_drawing_top)[0]

            if i + 1 < len(args):
                # since it's not the last arg, add a comma
                right = command(DrtTokens.COMMA + " ", right, centred_string_top)[0]

        # Handle close paren
        right = command(DrtTokens.CLOSE, right, centred_string_top)[0]

        return (right, max_bottom)

    def _handle_LambdaExpression(self, expression, command, x, y):
        # Find the width of the lambda symbol and abstracted variables
        variables = DrtTokens.LAMBDA + "%s" % expression.variable + DrtTokens.DOT
        right = self._visit_command(variables, x, y)[0]

        # Handle term
        (right, bottom) = self._handle(expression.term, command, right, y)

        # Handle variables now that we know the y-coordinate
        command(
            variables, x, self._get_centered_top(y, bottom - y, self._get_text_height())
        )

        return (right, bottom)

    def _handle_BinaryExpression(self, expression, command, x, y):
        # Get the full height of the line, based on the operands
        first_height = self._visit(expression.first, 0, 0)[1]
        second_height = self._visit(expression.second, 0, 0)[1]
        line_height = max(first_height, second_height)

        # Handle open paren
        centred_string_top = self._get_centered_top(
            y, line_height, self._get_text_height()
        )
        right = command(DrtTokens.OPEN, x, centred_string_top)[0]

        # Handle the first operand
        first_height = expression.first._drawing_height
        (right, first_bottom) = self._handle(
            expression.first,
            command,
            right,
            self._get_centered_top(y, line_height, first_height),
        )

        # Handle the operator
        right = command(" %s " % expression.getOp(), right, centred_string_top)[0]

        # Handle the second operand
        second_height = expression.second._drawing_height
        (right, second_bottom) = self._handle(
            expression.second,
            command,
            right,
            self._get_centered_top(y, line_height, second_height),
        )

        # Handle close paren
        right = command(DrtTokens.CLOSE, right, centred_string_top)[0]

        return (right, max(first_bottom, second_bottom))

    def _handle_DrtProposition(self, expression, command, x, y):
        # Find the width of the negation symbol
        right = command(expression.variable, x, y)[0]

        # Handle term
        (right, bottom) = self._handle(expression.term, command, right, y)

        return (right, bottom)

    def _get_centered_top(self, top, full_height, item_height):
        """Get the y-coordinate of the point that a figure should start at if
        its height is 'item_height' and it needs to be centered in an area that
        starts at 'top' and is 'full_height' tall."""
        return top + (full_height - item_height) / 2


def demo():
    print("=" * 20 + "TEST PARSE" + "=" * 20)
    dexpr = DrtExpression.fromstring
    print(dexpr(r"([x,y],[sees(x,y)])"))
    print(dexpr(r"([x],[man(x), walks(x)])"))
    print(dexpr(r"\x.\y.([],[sees(x,y)])"))
    print(dexpr(r"\x.([],[walks(x)])(john)"))
    print(dexpr(r"(([x],[walks(x)]) + ([y],[runs(y)]))"))
    print(dexpr(r"(([],[walks(x)]) -> ([],[runs(x)]))"))
    print(dexpr(r"([x],[PRO(x), sees(John,x)])"))
    print(dexpr(r"([x],[man(x), -([],[walks(x)])])"))
    print(dexpr(r"([],[(([x],[man(x)]) -> ([],[walks(x)]))])"))

    print("=" * 20 + "Test fol()" + "=" * 20)
    print(dexpr(r"([x,y],[sees(x,y)])").fol())

    print("=" * 20 + "Test alpha conversion and lambda expression equality" + "=" * 20)
    e1 = dexpr(r"\x.([],[P(x)])")
    print(e1)
    e2 = e1.alpha_convert(Variable("z"))
    print(e2)
    print(e1 == e2)

    print("=" * 20 + "Test resolve_anaphora()" + "=" * 20)
    print(resolve_anaphora(dexpr(r"([x,y,z],[dog(x), cat(y), walks(z), PRO(z)])")))
    print(
        resolve_anaphora(dexpr(r"([],[(([x],[dog(x)]) -> ([y],[walks(y), PRO(y)]))])"))
    )
    print(resolve_anaphora(dexpr(r"(([x,y],[]) + ([],[PRO(x)]))")))

    print("=" * 20 + "Test pretty_print()" + "=" * 20)
    dexpr(r"([],[])").pretty_print()
    dexpr(
        r"([],[([x],[big(x), dog(x)]) -> ([],[bark(x)]) -([x],[walk(x)])])"
    ).pretty_print()
    dexpr(r"([x,y],[x=y]) + ([z],[dog(z), walk(z)])").pretty_print()
    dexpr(r"([],[([x],[]) | ([y],[]) | ([z],[dog(z), walk(z)])])").pretty_print()
    dexpr(r"\P.\Q.(([x],[]) + P(x) + Q(x))(\x.([],[dog(x)]))").pretty_print()


def test_draw():
    try:
        from tkinter import Tk
    except ImportError as e:
        raise ValueError("tkinter is required, but it's not available.")

    expressions = [
        r"x",
        r"([],[])",
        r"([x],[])",
        r"([x],[man(x)])",
        r"([x,y],[sees(x,y)])",
        r"([x],[man(x), walks(x)])",
        r"\x.([],[man(x), walks(x)])",
        r"\x y.([],[sees(x,y)])",
        r"([],[(([],[walks(x)]) + ([],[runs(x)]))])",
        r"([x],[man(x), -([],[walks(x)])])",
        r"([],[(([x],[man(x)]) -> ([],[walks(x)]))])",
    ]

    for e in expressions:
        d = DrtExpression.fromstring(e)
        d.draw()


if __name__ == "__main__":
    demo()
