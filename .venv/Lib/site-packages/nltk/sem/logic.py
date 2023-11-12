# Natural Language Toolkit: Logic
#
# Author: Dan Garrette <dhgarrette@gmail.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A version of first order predicate logic, built on
top of the typed lambda calculus.
"""

import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering

from nltk.internals import Counter
from nltk.util import Trie

APP = "APP"

_counter = Counter()


class Tokens:
    LAMBDA = "\\"
    LAMBDA_LIST = ["\\"]

    # Quantifiers
    EXISTS = "exists"
    EXISTS_LIST = ["some", "exists", "exist"]
    ALL = "all"
    ALL_LIST = ["all", "forall"]
    IOTA = "iota"
    IOTA_LIST = ["iota"]

    # Punctuation
    DOT = "."
    OPEN = "("
    CLOSE = ")"
    COMMA = ","

    # Operations
    NOT = "-"
    NOT_LIST = ["not", "-", "!"]
    AND = "&"
    AND_LIST = ["and", "&", "^"]
    OR = "|"
    OR_LIST = ["or", "|"]
    IMP = "->"
    IMP_LIST = ["implies", "->", "=>"]
    IFF = "<->"
    IFF_LIST = ["iff", "<->", "<=>"]
    EQ = "="
    EQ_LIST = ["=", "=="]
    NEQ = "!="
    NEQ_LIST = ["!="]

    # Collections of tokens
    BINOPS = AND_LIST + OR_LIST + IMP_LIST + IFF_LIST
    QUANTS = EXISTS_LIST + ALL_LIST + IOTA_LIST
    PUNCT = [DOT, OPEN, CLOSE, COMMA]

    TOKENS = BINOPS + EQ_LIST + NEQ_LIST + QUANTS + LAMBDA_LIST + PUNCT + NOT_LIST

    # Special
    SYMBOLS = [x for x in TOKENS if re.match(r"^[-\\.(),!&^|>=<]*$", x)]


def boolean_ops():
    """
    Boolean operators
    """
    names = ["negation", "conjunction", "disjunction", "implication", "equivalence"]
    for pair in zip(names, [Tokens.NOT, Tokens.AND, Tokens.OR, Tokens.IMP, Tokens.IFF]):
        print("%-15s\t%s" % pair)


def equality_preds():
    """
    Equality predicates
    """
    names = ["equality", "inequality"]
    for pair in zip(names, [Tokens.EQ, Tokens.NEQ]):
        print("%-15s\t%s" % pair)


def binding_ops():
    """
    Binding operators
    """
    names = ["existential", "universal", "lambda"]
    for pair in zip(names, [Tokens.EXISTS, Tokens.ALL, Tokens.LAMBDA, Tokens.IOTA]):
        print("%-15s\t%s" % pair)


class LogicParser:
    """A lambda calculus expression parser."""

    def __init__(self, type_check=False):
        """
        :param type_check: should type checking be performed
            to their types?
        :type type_check: bool
        """
        assert isinstance(type_check, bool)

        self._currentIndex = 0
        self._buffer = []
        self.type_check = type_check

        """A list of tuples of quote characters.  The 4-tuple is comprised
        of the start character, the end character, the escape character, and
        a boolean indicating whether the quotes should be included in the
        result. Quotes are used to signify that a token should be treated as
        atomic, ignoring any special characters within the token.  The escape
        character allows the quote end character to be used within the quote.
        If True, the boolean indicates that the final token should contain the
        quote and escape characters.
        This method exists to be overridden"""
        self.quote_chars = []

        self.operator_precedence = dict(
            [(x, 1) for x in Tokens.LAMBDA_LIST]
            + [(x, 2) for x in Tokens.NOT_LIST]
            + [(APP, 3)]
            + [(x, 4) for x in Tokens.EQ_LIST + Tokens.NEQ_LIST]
            + [(x, 5) for x in Tokens.QUANTS]
            + [(x, 6) for x in Tokens.AND_LIST]
            + [(x, 7) for x in Tokens.OR_LIST]
            + [(x, 8) for x in Tokens.IMP_LIST]
            + [(x, 9) for x in Tokens.IFF_LIST]
            + [(None, 10)]
        )
        self.right_associated_operations = [APP]

    def parse(self, data, signature=None):
        """
        Parse the expression.

        :param data: str for the input to be parsed
        :param signature: ``dict<str, str>`` that maps variable names to type
            strings
        :returns: a parsed Expression
        """
        data = data.rstrip()

        self._currentIndex = 0
        self._buffer, mapping = self.process(data)

        try:
            result = self.process_next_expression(None)
            if self.inRange(0):
                raise UnexpectedTokenException(self._currentIndex + 1, self.token(0))
        except LogicalExpressionException as e:
            msg = "{}\n{}\n{}^".format(e, data, " " * mapping[e.index - 1])
            raise LogicalExpressionException(None, msg) from e

        if self.type_check:
            result.typecheck(signature)

        return result

    def process(self, data):
        """Split the data into tokens"""
        out = []
        mapping = {}
        tokenTrie = Trie(self.get_all_symbols())
        token = ""
        data_idx = 0
        token_start_idx = data_idx
        while data_idx < len(data):
            cur_data_idx = data_idx
            quoted_token, data_idx = self.process_quoted_token(data_idx, data)
            if quoted_token:
                if not token:
                    token_start_idx = cur_data_idx
                token += quoted_token
                continue

            st = tokenTrie
            c = data[data_idx]
            symbol = ""
            while c in st:
                symbol += c
                st = st[c]
                if len(data) - data_idx > len(symbol):
                    c = data[data_idx + len(symbol)]
                else:
                    break
            if Trie.LEAF in st:
                # token is a complete symbol
                if token:
                    mapping[len(out)] = token_start_idx
                    out.append(token)
                    token = ""
                mapping[len(out)] = data_idx
                out.append(symbol)
                data_idx += len(symbol)
            else:
                if data[data_idx] in " \t\n":  # any whitespace
                    if token:
                        mapping[len(out)] = token_start_idx
                        out.append(token)
                        token = ""
                else:
                    if not token:
                        token_start_idx = data_idx
                    token += data[data_idx]
                data_idx += 1
        if token:
            mapping[len(out)] = token_start_idx
            out.append(token)
        mapping[len(out)] = len(data)
        mapping[len(out) + 1] = len(data) + 1
        return out, mapping

    def process_quoted_token(self, data_idx, data):
        token = ""
        c = data[data_idx]
        i = data_idx
        for start, end, escape, incl_quotes in self.quote_chars:
            if c == start:
                if incl_quotes:
                    token += c
                i += 1
                while data[i] != end:
                    if data[i] == escape:
                        if incl_quotes:
                            token += data[i]
                        i += 1
                        if len(data) == i:  # if there are no more chars
                            raise LogicalExpressionException(
                                None,
                                "End of input reached.  "
                                "Escape character [%s] found at end." % escape,
                            )
                        token += data[i]
                    else:
                        token += data[i]
                    i += 1
                    if len(data) == i:
                        raise LogicalExpressionException(
                            None, "End of input reached.  " "Expected: [%s]" % end
                        )
                if incl_quotes:
                    token += data[i]
                i += 1
                if not token:
                    raise LogicalExpressionException(None, "Empty quoted token found")
                break
        return token, i

    def get_all_symbols(self):
        """This method exists to be overridden"""
        return Tokens.SYMBOLS

    def inRange(self, location):
        """Return TRUE if the given location is within the buffer"""
        return self._currentIndex + location < len(self._buffer)

    def token(self, location=None):
        """Get the next waiting token.  If a location is given, then
        return the token at currentIndex+location without advancing
        currentIndex; setting it gives lookahead/lookback capability."""
        try:
            if location is None:
                tok = self._buffer[self._currentIndex]
                self._currentIndex += 1
            else:
                tok = self._buffer[self._currentIndex + location]
            return tok
        except IndexError as e:
            raise ExpectedMoreTokensException(self._currentIndex + 1) from e

    def isvariable(self, tok):
        return tok not in Tokens.TOKENS

    def process_next_expression(self, context):
        """Parse the next complete expression from the stream and return it."""
        try:
            tok = self.token()
        except ExpectedMoreTokensException as e:
            raise ExpectedMoreTokensException(
                self._currentIndex + 1, message="Expression expected."
            ) from e

        accum = self.handle(tok, context)

        if not accum:
            raise UnexpectedTokenException(
                self._currentIndex, tok, message="Expression expected."
            )

        return self.attempt_adjuncts(accum, context)

    def handle(self, tok, context):
        """This method is intended to be overridden for logics that
        use different operators or expressions"""
        if self.isvariable(tok):
            return self.handle_variable(tok, context)

        elif tok in Tokens.NOT_LIST:
            return self.handle_negation(tok, context)

        elif tok in Tokens.LAMBDA_LIST:
            return self.handle_lambda(tok, context)

        elif tok in Tokens.QUANTS:
            return self.handle_quant(tok, context)

        elif tok == Tokens.OPEN:
            return self.handle_open(tok, context)

    def attempt_adjuncts(self, expression, context):
        cur_idx = None
        while cur_idx != self._currentIndex:  # while adjuncts are added
            cur_idx = self._currentIndex
            expression = self.attempt_EqualityExpression(expression, context)
            expression = self.attempt_ApplicationExpression(expression, context)
            expression = self.attempt_BooleanExpression(expression, context)
        return expression

    def handle_negation(self, tok, context):
        return self.make_NegatedExpression(self.process_next_expression(Tokens.NOT))

    def make_NegatedExpression(self, expression):
        return NegatedExpression(expression)

    def handle_variable(self, tok, context):
        # It's either: 1) a predicate expression: sees(x,y)
        #             2) an application expression: P(x)
        #             3) a solo variable: john OR x
        accum = self.make_VariableExpression(tok)
        if self.inRange(0) and self.token(0) == Tokens.OPEN:
            # The predicate has arguments
            if not isinstance(accum, FunctionVariableExpression) and not isinstance(
                accum, ConstantExpression
            ):
                raise LogicalExpressionException(
                    self._currentIndex,
                    "'%s' is an illegal predicate name.  "
                    "Individual variables may not be used as "
                    "predicates." % tok,
                )
            self.token()  # swallow the Open Paren

            # curry the arguments
            accum = self.make_ApplicationExpression(
                accum, self.process_next_expression(APP)
            )
            while self.inRange(0) and self.token(0) == Tokens.COMMA:
                self.token()  # swallow the comma
                accum = self.make_ApplicationExpression(
                    accum, self.process_next_expression(APP)
                )
            self.assertNextToken(Tokens.CLOSE)
        return accum

    def get_next_token_variable(self, description):
        try:
            tok = self.token()
        except ExpectedMoreTokensException as e:
            raise ExpectedMoreTokensException(e.index, "Variable expected.") from e
        if isinstance(self.make_VariableExpression(tok), ConstantExpression):
            raise LogicalExpressionException(
                self._currentIndex,
                "'%s' is an illegal variable name.  "
                "Constants may not be %s." % (tok, description),
            )
        return Variable(tok)

    def handle_lambda(self, tok, context):
        # Expression is a lambda expression
        if not self.inRange(0):
            raise ExpectedMoreTokensException(
                self._currentIndex + 2,
                message="Variable and Expression expected following lambda operator.",
            )
        vars = [self.get_next_token_variable("abstracted")]
        while True:
            if not self.inRange(0) or (
                self.token(0) == Tokens.DOT and not self.inRange(1)
            ):
                raise ExpectedMoreTokensException(
                    self._currentIndex + 2, message="Expression expected."
                )
            if not self.isvariable(self.token(0)):
                break
            # Support expressions like: \x y.M == \x.\y.M
            vars.append(self.get_next_token_variable("abstracted"))
        if self.inRange(0) and self.token(0) == Tokens.DOT:
            self.token()  # swallow the dot

        accum = self.process_next_expression(tok)
        while vars:
            accum = self.make_LambdaExpression(vars.pop(), accum)
        return accum

    def handle_quant(self, tok, context):
        # Expression is a quantified expression: some x.M
        factory = self.get_QuantifiedExpression_factory(tok)

        if not self.inRange(0):
            raise ExpectedMoreTokensException(
                self._currentIndex + 2,
                message="Variable and Expression expected following quantifier '%s'."
                % tok,
            )
        vars = [self.get_next_token_variable("quantified")]
        while True:
            if not self.inRange(0) or (
                self.token(0) == Tokens.DOT and not self.inRange(1)
            ):
                raise ExpectedMoreTokensException(
                    self._currentIndex + 2, message="Expression expected."
                )
            if not self.isvariable(self.token(0)):
                break
            # Support expressions like: some x y.M == some x.some y.M
            vars.append(self.get_next_token_variable("quantified"))
        if self.inRange(0) and self.token(0) == Tokens.DOT:
            self.token()  # swallow the dot

        accum = self.process_next_expression(tok)
        while vars:
            accum = self.make_QuanifiedExpression(factory, vars.pop(), accum)
        return accum

    def get_QuantifiedExpression_factory(self, tok):
        """This method serves as a hook for other logic parsers that
        have different quantifiers"""
        if tok in Tokens.EXISTS_LIST:
            return ExistsExpression
        elif tok in Tokens.ALL_LIST:
            return AllExpression
        elif tok in Tokens.IOTA_LIST:
            return IotaExpression
        else:
            self.assertToken(tok, Tokens.QUANTS)

    def make_QuanifiedExpression(self, factory, variable, term):
        return factory(variable, term)

    def handle_open(self, tok, context):
        # Expression is in parens
        accum = self.process_next_expression(None)
        self.assertNextToken(Tokens.CLOSE)
        return accum

    def attempt_EqualityExpression(self, expression, context):
        """Attempt to make an equality expression.  If the next token is an
        equality operator, then an EqualityExpression will be returned.
        Otherwise, the parameter will be returned."""
        if self.inRange(0):
            tok = self.token(0)
            if tok in Tokens.EQ_LIST + Tokens.NEQ_LIST and self.has_priority(
                tok, context
            ):
                self.token()  # swallow the "=" or "!="
                expression = self.make_EqualityExpression(
                    expression, self.process_next_expression(tok)
                )
                if tok in Tokens.NEQ_LIST:
                    expression = self.make_NegatedExpression(expression)
        return expression

    def make_EqualityExpression(self, first, second):
        """This method serves as a hook for other logic parsers that
        have different equality expression classes"""
        return EqualityExpression(first, second)

    def attempt_BooleanExpression(self, expression, context):
        """Attempt to make a boolean expression.  If the next token is a boolean
        operator, then a BooleanExpression will be returned.  Otherwise, the
        parameter will be returned."""
        while self.inRange(0):
            tok = self.token(0)
            factory = self.get_BooleanExpression_factory(tok)
            if factory and self.has_priority(tok, context):
                self.token()  # swallow the operator
                expression = self.make_BooleanExpression(
                    factory, expression, self.process_next_expression(tok)
                )
            else:
                break
        return expression

    def get_BooleanExpression_factory(self, tok):
        """This method serves as a hook for other logic parsers that
        have different boolean operators"""
        if tok in Tokens.AND_LIST:
            return AndExpression
        elif tok in Tokens.OR_LIST:
            return OrExpression
        elif tok in Tokens.IMP_LIST:
            return ImpExpression
        elif tok in Tokens.IFF_LIST:
            return IffExpression
        else:
            return None

    def make_BooleanExpression(self, factory, first, second):
        return factory(first, second)

    def attempt_ApplicationExpression(self, expression, context):
        """Attempt to make an application expression.  The next tokens are
        a list of arguments in parens, then the argument expression is a
        function being applied to the arguments.  Otherwise, return the
        argument expression."""
        if self.has_priority(APP, context):
            if self.inRange(0) and self.token(0) == Tokens.OPEN:
                if (
                    not isinstance(expression, LambdaExpression)
                    and not isinstance(expression, ApplicationExpression)
                    and not isinstance(expression, FunctionVariableExpression)
                    and not isinstance(expression, ConstantExpression)
                ):
                    raise LogicalExpressionException(
                        self._currentIndex,
                        ("The function '%s" % expression)
                        + "' is not a Lambda Expression, an "
                        "Application Expression, or a "
                        "functional predicate, so it may "
                        "not take arguments.",
                    )
                self.token()  # swallow then open paren
                # curry the arguments
                accum = self.make_ApplicationExpression(
                    expression, self.process_next_expression(APP)
                )
                while self.inRange(0) and self.token(0) == Tokens.COMMA:
                    self.token()  # swallow the comma
                    accum = self.make_ApplicationExpression(
                        accum, self.process_next_expression(APP)
                    )
                self.assertNextToken(Tokens.CLOSE)
                return accum
        return expression

    def make_ApplicationExpression(self, function, argument):
        return ApplicationExpression(function, argument)

    def make_VariableExpression(self, name):
        return VariableExpression(Variable(name))

    def make_LambdaExpression(self, variable, term):
        return LambdaExpression(variable, term)

    def has_priority(self, operation, context):
        return self.operator_precedence[operation] < self.operator_precedence[
            context
        ] or (
            operation in self.right_associated_operations
            and self.operator_precedence[operation] == self.operator_precedence[context]
        )

    def assertNextToken(self, expected):
        try:
            tok = self.token()
        except ExpectedMoreTokensException as e:
            raise ExpectedMoreTokensException(
                e.index, message="Expected token '%s'." % expected
            ) from e

        if isinstance(expected, list):
            if tok not in expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)
        else:
            if tok != expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)

    def assertToken(self, tok, expected):
        if isinstance(expected, list):
            if tok not in expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)
        else:
            if tok != expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)

    def __repr__(self):
        if self.inRange(0):
            msg = "Next token: " + self.token(0)
        else:
            msg = "No more tokens"
        return "<" + self.__class__.__name__ + ": " + msg + ">"


def read_logic(s, logic_parser=None, encoding=None):
    """
    Convert a file of First Order Formulas into a list of {Expression}s.

    :param s: the contents of the file
    :type s: str
    :param logic_parser: The parser to be used to parse the logical expression
    :type logic_parser: LogicParser
    :param encoding: the encoding of the input string, if it is binary
    :type encoding: str
    :return: a list of parsed formulas.
    :rtype: list(Expression)
    """
    if encoding is not None:
        s = s.decode(encoding)
    if logic_parser is None:
        logic_parser = LogicParser()

    statements = []
    for linenum, line in enumerate(s.splitlines()):
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        try:
            statements.append(logic_parser.parse(line))
        except LogicalExpressionException as e:
            raise ValueError(f"Unable to parse line {linenum}: {line}") from e
    return statements


@total_ordering
class Variable:
    def __init__(self, name):
        """
        :param name: the name of the variable
        """
        assert isinstance(name, str), "%s is not a string" % name
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, Variable):
            raise TypeError
        return self.name < other.name

    def substitute_bindings(self, bindings):
        return bindings.get(self, self)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Variable('%s')" % self.name


def unique_variable(pattern=None, ignore=None):
    """
    Return a new, unique variable.

    :param pattern: ``Variable`` that is being replaced.  The new variable must
        be the same type.
    :param term: a set of ``Variable`` objects that should not be returned from
        this function.
    :rtype: Variable
    """
    if pattern is not None:
        if is_indvar(pattern.name):
            prefix = "z"
        elif is_funcvar(pattern.name):
            prefix = "F"
        elif is_eventvar(pattern.name):
            prefix = "e0"
        else:
            assert False, "Cannot generate a unique constant"
    else:
        prefix = "z"

    v = Variable(f"{prefix}{_counter.get()}")
    while ignore is not None and v in ignore:
        v = Variable(f"{prefix}{_counter.get()}")
    return v


def skolem_function(univ_scope=None):
    """
    Return a skolem function over the variables in univ_scope
    param univ_scope
    """
    skolem = VariableExpression(Variable("F%s" % _counter.get()))
    if univ_scope:
        for v in list(univ_scope):
            skolem = skolem(VariableExpression(v))
    return skolem


class Type:
    def __repr__(self):
        return "%s" % self

    def __hash__(self):
        return hash("%s" % self)

    @classmethod
    def fromstring(cls, s):
        return read_type(s)


class ComplexType(Type):
    def __init__(self, first, second):
        assert isinstance(first, Type), "%s is not a Type" % first
        assert isinstance(second, Type), "%s is not a Type" % second
        self.first = first
        self.second = second

    def __eq__(self, other):
        return (
            isinstance(other, ComplexType)
            and self.first == other.first
            and self.second == other.second
        )

    def __ne__(self, other):
        return not self == other

    __hash__ = Type.__hash__

    def matches(self, other):
        if isinstance(other, ComplexType):
            return self.first.matches(other.first) and self.second.matches(other.second)
        else:
            return self == ANY_TYPE

    def resolve(self, other):
        if other == ANY_TYPE:
            return self
        elif isinstance(other, ComplexType):
            f = self.first.resolve(other.first)
            s = self.second.resolve(other.second)
            if f and s:
                return ComplexType(f, s)
            else:
                return None
        elif self == ANY_TYPE:
            return other
        else:
            return None

    def __str__(self):
        if self == ANY_TYPE:
            return "%s" % ANY_TYPE
        else:
            return f"<{self.first},{self.second}>"

    def str(self):
        if self == ANY_TYPE:
            return ANY_TYPE.str()
        else:
            return f"({self.first.str()} -> {self.second.str()})"


class BasicType(Type):
    def __eq__(self, other):
        return isinstance(other, BasicType) and ("%s" % self) == ("%s" % other)

    def __ne__(self, other):
        return not self == other

    __hash__ = Type.__hash__

    def matches(self, other):
        return other == ANY_TYPE or self == other

    def resolve(self, other):
        if self.matches(other):
            return self
        else:
            return None


class EntityType(BasicType):
    def __str__(self):
        return "e"

    def str(self):
        return "IND"


class TruthValueType(BasicType):
    def __str__(self):
        return "t"

    def str(self):
        return "BOOL"


class EventType(BasicType):
    def __str__(self):
        return "v"

    def str(self):
        return "EVENT"


class AnyType(BasicType, ComplexType):
    def __init__(self):
        pass

    @property
    def first(self):
        return self

    @property
    def second(self):
        return self

    def __eq__(self, other):
        return isinstance(other, AnyType) or other.__eq__(self)

    def __ne__(self, other):
        return not self == other

    __hash__ = Type.__hash__

    def matches(self, other):
        return True

    def resolve(self, other):
        return other

    def __str__(self):
        return "?"

    def str(self):
        return "ANY"


TRUTH_TYPE = TruthValueType()
ENTITY_TYPE = EntityType()
EVENT_TYPE = EventType()
ANY_TYPE = AnyType()


def read_type(type_string):
    assert isinstance(type_string, str)
    type_string = type_string.replace(" ", "")  # remove spaces

    if type_string[0] == "<":
        assert type_string[-1] == ">"
        paren_count = 0
        for i, char in enumerate(type_string):
            if char == "<":
                paren_count += 1
            elif char == ">":
                paren_count -= 1
                assert paren_count > 0
            elif char == ",":
                if paren_count == 1:
                    break
        return ComplexType(
            read_type(type_string[1:i]), read_type(type_string[i + 1 : -1])
        )
    elif type_string[0] == "%s" % ENTITY_TYPE:
        return ENTITY_TYPE
    elif type_string[0] == "%s" % TRUTH_TYPE:
        return TRUTH_TYPE
    elif type_string[0] == "%s" % ANY_TYPE:
        return ANY_TYPE
    else:
        raise LogicalExpressionException(
            None, "Unexpected character: '%s'." % type_string[0]
        )


class TypeException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class InconsistentTypeHierarchyException(TypeException):
    def __init__(self, variable, expression=None):
        if expression:
            msg = (
                "The variable '%s' was found in multiple places with different"
                " types in '%s'." % (variable, expression)
            )
        else:
            msg = (
                "The variable '%s' was found in multiple places with different"
                " types." % (variable)
            )
        super().__init__(msg)


class TypeResolutionException(TypeException):
    def __init__(self, expression, other_type):
        super().__init__(
            "The type of '%s', '%s', cannot be resolved with type '%s'"
            % (expression, expression.type, other_type)
        )


class IllegalTypeException(TypeException):
    def __init__(self, expression, other_type, allowed_type):
        super().__init__(
            "Cannot set type of %s '%s' to '%s'; must match type '%s'."
            % (expression.__class__.__name__, expression, other_type, allowed_type)
        )


def typecheck(expressions, signature=None):
    """
    Ensure correct typing across a collection of ``Expression`` objects.
    :param expressions: a collection of expressions
    :param signature: dict that maps variable names to types (or string
    representations of types)
    """
    # typecheck and create master signature
    for expression in expressions:
        signature = expression.typecheck(signature)
    # apply master signature to all expressions
    for expression in expressions[:-1]:
        expression.typecheck(signature)
    return signature


class SubstituteBindingsI:
    """
    An interface for classes that can perform substitutions for
    variables.
    """

    def substitute_bindings(self, bindings):
        """
        :return: The object that is obtained by replacing
            each variable bound by ``bindings`` with its values.
            Aliases are already resolved. (maybe?)
        :rtype: (any)
        """
        raise NotImplementedError()

    def variables(self):
        """
        :return: A list of all variables in this object.
        """
        raise NotImplementedError()


class Expression(SubstituteBindingsI):
    """This is the base abstract object for all logical expressions"""

    _logic_parser = LogicParser()
    _type_checking_logic_parser = LogicParser(type_check=True)

    @classmethod
    def fromstring(cls, s, type_check=False, signature=None):
        if type_check:
            return cls._type_checking_logic_parser.parse(s, signature)
        else:
            return cls._logic_parser.parse(s, signature)

    def __call__(self, other, *additional):
        accum = self.applyto(other)
        for a in additional:
            accum = accum(a)
        return accum

    def applyto(self, other):
        assert isinstance(other, Expression), "%s is not an Expression" % other
        return ApplicationExpression(self, other)

    def __neg__(self):
        return NegatedExpression(self)

    def negate(self):
        """If this is a negated expression, remove the negation.
        Otherwise add a negation."""
        return -self

    def __and__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("%s is not an Expression" % other)
        return AndExpression(self, other)

    def __or__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("%s is not an Expression" % other)
        return OrExpression(self, other)

    def __gt__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("%s is not an Expression" % other)
        return ImpExpression(self, other)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("%s is not an Expression" % other)
        return IffExpression(self, other)

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def equiv(self, other, prover=None):
        """
        Check for logical equivalence.
        Pass the expression (self <-> other) to the theorem prover.
        If the prover says it is valid, then the self and other are equal.

        :param other: an ``Expression`` to check equality against
        :param prover: a ``nltk.inference.api.Prover``
        """
        assert isinstance(other, Expression), "%s is not an Expression" % other

        if prover is None:
            from nltk.inference import Prover9

            prover = Prover9()
        bicond = IffExpression(self.simplify(), other.simplify())
        return prover.prove(bicond)

    def __hash__(self):
        return hash(repr(self))

    def substitute_bindings(self, bindings):
        expr = self
        for var in expr.variables():
            if var in bindings:
                val = bindings[var]
                if isinstance(val, Variable):
                    val = self.make_VariableExpression(val)
                elif not isinstance(val, Expression):
                    raise ValueError(
                        "Can not substitute a non-expression "
                        "value into an expression: %r" % (val,)
                    )
                # Substitute bindings in the target value.
                val = val.substitute_bindings(bindings)
                # Replace var w/ the target value.
                expr = expr.replace(var, val)
        return expr.simplify()

    def typecheck(self, signature=None):
        """
        Infer and check types.  Raise exceptions if necessary.

        :param signature: dict that maps variable names to types (or string
            representations of types)
        :return: the signature, plus any additional type mappings
        """
        sig = defaultdict(list)
        if signature:
            for key in signature:
                val = signature[key]
                varEx = VariableExpression(Variable(key))
                if isinstance(val, Type):
                    varEx.type = val
                else:
                    varEx.type = read_type(val)
                sig[key].append(varEx)

        self._set_type(signature=sig)

        return {key: sig[key][0].type for key in sig}

    def findtype(self, variable):
        """
        Find the type of the given variable as it is used in this expression.
        For example, finding the type of "P" in "P(x) & Q(x,y)" yields "<e,t>"

        :param variable: Variable
        """
        raise NotImplementedError()

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """
        Set the type of this expression to be the given type.  Raise type
        exceptions where applicable.

        :param other_type: Type
        :param signature: dict(str -> list(AbstractVariableExpression))
        """
        raise NotImplementedError()

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """
        Replace every instance of 'variable' with 'expression'
        :param variable: ``Variable`` The variable to replace
        :param expression: ``Expression`` The expression with which to replace it
        :param replace_bound: bool Should bound variables be replaced?
        :param alpha_convert: bool Alpha convert automatically to avoid name clashes?
        """
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        assert isinstance(expression, Expression), (
            "%s is not an Expression" % expression
        )

        return self.visit_structured(
            lambda e: e.replace(variable, expression, replace_bound, alpha_convert),
            self.__class__,
        )

    def normalize(self, newvars=None):
        """Rename auto-generated unique variables"""

        def get_indiv_vars(e):
            if isinstance(e, IndividualVariableExpression):
                return {e}
            elif isinstance(e, AbstractVariableExpression):
                return set()
            else:
                return e.visit(
                    get_indiv_vars, lambda parts: reduce(operator.or_, parts, set())
                )

        result = self
        for i, e in enumerate(sorted(get_indiv_vars(self), key=lambda e: e.variable)):
            if isinstance(e, EventVariableExpression):
                newVar = e.__class__(Variable("e0%s" % (i + 1)))
            elif isinstance(e, IndividualVariableExpression):
                newVar = e.__class__(Variable("z%s" % (i + 1)))
            else:
                newVar = e
            result = result.replace(e.variable, newVar, True)
        return result

    def visit(self, function, combinator):
        """
        Recursively visit subexpressions.  Apply 'function' to each
        subexpression and pass the result of each function application
        to the 'combinator' for aggregation:

            return combinator(map(function, self.subexpressions))

        Bound variables are neither applied upon by the function nor given to
        the combinator.
        :param function: ``Function<Expression,T>`` to call on each subexpression
        :param combinator: ``Function<list<T>,R>`` to combine the results of the
        function calls
        :return: result of combination ``R``
        """
        raise NotImplementedError()

    def visit_structured(self, function, combinator):
        """
        Recursively visit subexpressions.  Apply 'function' to each
        subexpression and pass the result of each function application
        to the 'combinator' for aggregation.  The combinator must have
        the same signature as the constructor.  The function is not
        applied to bound variables, but they are passed to the
        combinator.
        :param function: ``Function`` to call on each subexpression
        :param combinator: ``Function`` with the same signature as the
        constructor, to combine the results of the function calls
        :return: result of combination
        """
        return self.visit(function, lambda parts: combinator(*parts))

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"

    def __str__(self):
        return self.str()

    def variables(self):
        """
        Return a set of all the variables for binding substitution.
        The variables returned include all free (non-bound) individual
        variables and any variable starting with '?' or '@'.
        :return: set of ``Variable`` objects
        """
        return self.free() | {
            p for p in self.predicates() | self.constants() if re.match("^[?@]", p.name)
        }

    def free(self):
        """
        Return a set of all the free (non-bound) variables.  This includes
        both individual and predicate variables, but not constants.
        :return: set of ``Variable`` objects
        """
        return self.visit(
            lambda e: e.free(), lambda parts: reduce(operator.or_, parts, set())
        )

    def constants(self):
        """
        Return a set of individual constants (non-predicates).
        :return: set of ``Variable`` objects
        """
        return self.visit(
            lambda e: e.constants(), lambda parts: reduce(operator.or_, parts, set())
        )

    def predicates(self):
        """
        Return a set of predicates (constants, not variables).
        :return: set of ``Variable`` objects
        """
        return self.visit(
            lambda e: e.predicates(), lambda parts: reduce(operator.or_, parts, set())
        )

    def simplify(self):
        """
        :return: beta-converted version of this expression
        """
        return self.visit_structured(lambda e: e.simplify(), self.__class__)

    def make_VariableExpression(self, variable):
        return VariableExpression(variable)


class ApplicationExpression(Expression):
    r"""
    This class is used to represent two related types of logical expressions.

    The first is a Predicate Expression, such as "P(x,y)".  A predicate
    expression is comprised of a ``FunctionVariableExpression`` or
    ``ConstantExpression`` as the predicate and a list of Expressions as the
    arguments.

    The second is a an application of one expression to another, such as
    "(\x.dog(x))(fido)".

    The reason Predicate Expressions are treated as Application Expressions is
    that the Variable Expression predicate of the expression may be replaced
    with another Expression, such as a LambdaExpression, which would mean that
    the Predicate should be thought of as being applied to the arguments.

    The logical expression reader will always curry arguments in a application expression.
    So, "\x y.see(x,y)(john,mary)" will be represented internally as
    "((\x y.(see(x))(y))(john))(mary)".  This simplifies the internals since
    there will always be exactly one argument in an application.

    The str() method will usually print the curried forms of application
    expressions.  The one exception is when the the application expression is
    really a predicate expression (ie, underlying function is an
    ``AbstractVariableExpression``).  This means that the example from above
    will be returned as "(\x y.see(x,y)(john))(mary)".
    """

    def __init__(self, function, argument):
        """
        :param function: ``Expression``, for the function expression
        :param argument: ``Expression``, for the argument
        """
        assert isinstance(function, Expression), "%s is not an Expression" % function
        assert isinstance(argument, Expression), "%s is not an Expression" % argument
        self.function = function
        self.argument = argument

    def simplify(self):
        function = self.function.simplify()
        argument = self.argument.simplify()
        if isinstance(function, LambdaExpression):
            return function.term.replace(function.variable, argument).simplify()
        else:
            return self.__class__(function, argument)

    @property
    def type(self):
        if isinstance(self.function.type, ComplexType):
            return self.function.type.second
        else:
            return ANY_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        self.argument._set_type(ANY_TYPE, signature)
        try:
            self.function._set_type(
                ComplexType(self.argument.type, other_type), signature
            )
        except TypeResolutionException as e:
            raise TypeException(
                "The function '%s' is of type '%s' and cannot be applied "
                "to '%s' of type '%s'.  Its argument must match type '%s'."
                % (
                    self.function,
                    self.function.type,
                    self.argument,
                    self.argument.type,
                    self.function.type.first,
                )
            ) from e

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        if self.is_atom():
            function, args = self.uncurry()
        else:
            # It's not a predicate expression ("P(x,y)"), so leave args curried
            function = self.function
            args = [self.argument]

        found = [arg.findtype(variable) for arg in [function] + args]

        unique = []
        for f in found:
            if f != ANY_TYPE:
                if unique:
                    for u in unique:
                        if f.matches(u):
                            break
                else:
                    unique.append(f)

        if len(unique) == 1:
            return list(unique)[0]
        else:
            return ANY_TYPE

    def constants(self):
        """:see: Expression.constants()"""
        if isinstance(self.function, AbstractVariableExpression):
            function_constants = set()
        else:
            function_constants = self.function.constants()
        return function_constants | self.argument.constants()

    def predicates(self):
        """:see: Expression.predicates()"""
        if isinstance(self.function, ConstantExpression):
            function_preds = {self.function.variable}
        else:
            function_preds = self.function.predicates()
        return function_preds | self.argument.predicates()

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.function), function(self.argument)])

    def __eq__(self, other):
        return (
            isinstance(other, ApplicationExpression)
            and self.function == other.function
            and self.argument == other.argument
        )

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__

    def __str__(self):
        # uncurry the arguments and find the base function
        if self.is_atom():
            function, args = self.uncurry()
            arg_str = ",".join("%s" % arg for arg in args)
        else:
            # Leave arguments curried
            function = self.function
            arg_str = "%s" % self.argument

        function_str = "%s" % function
        parenthesize_function = False
        if isinstance(function, LambdaExpression):
            if isinstance(function.term, ApplicationExpression):
                if not isinstance(function.term.function, AbstractVariableExpression):
                    parenthesize_function = True
            elif not isinstance(function.term, BooleanExpression):
                parenthesize_function = True
        elif isinstance(function, ApplicationExpression):
            parenthesize_function = True

        if parenthesize_function:
            function_str = Tokens.OPEN + function_str + Tokens.CLOSE

        return function_str + Tokens.OPEN + arg_str + Tokens.CLOSE

    def uncurry(self):
        """
        Uncurry this application expression

        return: A tuple (base-function, arg-list)
        """
        function = self.function
        args = [self.argument]
        while isinstance(function, ApplicationExpression):
            # (\x.\y.sees(x,y)(john))(mary)
            args.insert(0, function.argument)
            function = function.function
        return (function, args)

    @property
    def pred(self):
        """
        Return uncurried base-function.
        If this is an atom, then the result will be a variable expression.
        Otherwise, it will be a lambda expression.
        """
        return self.uncurry()[0]

    @property
    def args(self):
        """
        Return uncurried arg-list
        """
        return self.uncurry()[1]

    def is_atom(self):
        """
        Is this expression an atom (as opposed to a lambda expression applied
        to a term)?
        """
        return isinstance(self.pred, AbstractVariableExpression)


@total_ordering
class AbstractVariableExpression(Expression):
    """This class represents a variable to be used as a predicate or entity"""

    def __init__(self, variable):
        """
        :param variable: ``Variable``, for the variable
        """
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        self.variable = variable

    def simplify(self):
        return self

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """:see: Expression.replace()"""
        assert isinstance(variable, Variable), "%s is not an Variable" % variable
        assert isinstance(expression, Expression), (
            "%s is not an Expression" % expression
        )
        if self.variable == variable:
            return expression
        else:
            return self

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        resolution = other_type
        for varEx in signature[self.variable.name]:
            resolution = varEx.type.resolve(resolution)
            if not resolution:
                raise InconsistentTypeHierarchyException(self)

        signature[self.variable.name].append(self)
        for varEx in signature[self.variable.name]:
            varEx.type = resolution

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        if self.variable == variable:
            return self.type
        else:
            return ANY_TYPE

    def predicates(self):
        """:see: Expression.predicates()"""
        return set()

    def __eq__(self, other):
        """Allow equality between instances of ``AbstractVariableExpression``
        subtypes."""
        return (
            isinstance(other, AbstractVariableExpression)
            and self.variable == other.variable
        )

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, AbstractVariableExpression):
            raise TypeError
        return self.variable < other.variable

    __hash__ = Expression.__hash__

    def __str__(self):
        return "%s" % self.variable


class IndividualVariableExpression(AbstractVariableExpression):
    """This class represents variables that take the form of a single lowercase
    character (other than 'e') followed by zero or more digits."""

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if not other_type.matches(ENTITY_TYPE):
            raise IllegalTypeException(self, other_type, ENTITY_TYPE)

        signature[self.variable.name].append(self)

    def _get_type(self):
        return ENTITY_TYPE

    type = property(_get_type, _set_type)

    def free(self):
        """:see: Expression.free()"""
        return {self.variable}

    def constants(self):
        """:see: Expression.constants()"""
        return set()


class FunctionVariableExpression(AbstractVariableExpression):
    """This class represents variables that take the form of a single uppercase
    character followed by zero or more digits."""

    type = ANY_TYPE

    def free(self):
        """:see: Expression.free()"""
        return {self.variable}

    def constants(self):
        """:see: Expression.constants()"""
        return set()


class EventVariableExpression(IndividualVariableExpression):
    """This class represents variables that take the form of a single lowercase
    'e' character followed by zero or more digits."""

    type = EVENT_TYPE


class ConstantExpression(AbstractVariableExpression):
    """This class represents variables that do not take the form of a single
    character followed by zero or more digits."""

    type = ENTITY_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if other_type == ANY_TYPE:
            # entity type by default, for individuals
            resolution = ENTITY_TYPE
        else:
            resolution = other_type
            if self.type != ENTITY_TYPE:
                resolution = resolution.resolve(self.type)

        for varEx in signature[self.variable.name]:
            resolution = varEx.type.resolve(resolution)
            if not resolution:
                raise InconsistentTypeHierarchyException(self)

        signature[self.variable.name].append(self)
        for varEx in signature[self.variable.name]:
            varEx.type = resolution

    def free(self):
        """:see: Expression.free()"""
        return set()

    def constants(self):
        """:see: Expression.constants()"""
        return {self.variable}


def VariableExpression(variable):
    """
    This is a factory method that instantiates and returns a subtype of
    ``AbstractVariableExpression`` appropriate for the given variable.
    """
    assert isinstance(variable, Variable), "%s is not a Variable" % variable
    if is_indvar(variable.name):
        return IndividualVariableExpression(variable)
    elif is_funcvar(variable.name):
        return FunctionVariableExpression(variable)
    elif is_eventvar(variable.name):
        return EventVariableExpression(variable)
    else:
        return ConstantExpression(variable)


class VariableBinderExpression(Expression):
    """This an abstract class for any Expression that binds a variable in an
    Expression.  This includes LambdaExpressions and Quantified Expressions"""

    def __init__(self, variable, term):
        """
        :param variable: ``Variable``, for the variable
        :param term: ``Expression``, for the term
        """
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        assert isinstance(term, Expression), "%s is not an Expression" % term
        self.variable = variable
        self.term = term

    def replace(self, variable, expression, replace_bound=False, alpha_convert=True):
        """:see: Expression.replace()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        assert isinstance(expression, Expression), (
            "%s is not an Expression" % expression
        )
        # if the bound variable is the thing being replaced
        if self.variable == variable:
            if replace_bound:
                assert isinstance(expression, AbstractVariableExpression), (
                    "%s is not a AbstractVariableExpression" % expression
                )
                return self.__class__(
                    expression.variable,
                    self.term.replace(variable, expression, True, alpha_convert),
                )
            else:
                return self
        else:
            # if the bound variable appears in the expression, then it must
            # be alpha converted to avoid a conflict
            if alpha_convert and self.variable in expression.free():
                self = self.alpha_convert(unique_variable(pattern=self.variable))

            # replace in the term
            return self.__class__(
                self.variable,
                self.term.replace(variable, expression, replace_bound, alpha_convert),
            )

    def alpha_convert(self, newvar):
        """Rename all occurrences of the variable introduced by this variable
        binder in the expression to ``newvar``.
        :param newvar: ``Variable``, for the new variable
        """
        assert isinstance(newvar, Variable), "%s is not a Variable" % newvar
        return self.__class__(
            newvar, self.term.replace(self.variable, VariableExpression(newvar), True)
        )

    def free(self):
        """:see: Expression.free()"""
        return self.term.free() - {self.variable}

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        if variable == self.variable:
            return ANY_TYPE
        else:
            return self.term.findtype(variable)

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.term)])

    def visit_structured(self, function, combinator):
        """:see: Expression.visit_structured()"""
        return combinator(self.variable, function(self.term))

    def __eq__(self, other):
        r"""Defines equality modulo alphabetic variance.  If we are comparing
        \x.M  and \y.N, then check equality of M and N[x/y]."""
        if isinstance(self, other.__class__) or isinstance(other, self.__class__):
            if self.variable == other.variable:
                return self.term == other.term
            else:
                # Comparing \x.M  and \y.N.  Relabel y in N with x and continue.
                varex = VariableExpression(self.variable)
                return self.term == other.term.replace(other.variable, varex)
        else:
            return False

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__


class LambdaExpression(VariableBinderExpression):
    @property
    def type(self):
        return ComplexType(self.term.findtype(self.variable), self.term.type)

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        self.term._set_type(other_type.second, signature)
        if not self.type.resolve(other_type):
            raise TypeResolutionException(self, other_type)

    def __str__(self):
        variables = [self.variable]
        term = self.term
        while term.__class__ == self.__class__:
            variables.append(term.variable)
            term = term.term
        return (
            Tokens.LAMBDA
            + " ".join("%s" % v for v in variables)
            + Tokens.DOT
            + "%s" % term
        )


class QuantifiedExpression(VariableBinderExpression):
    @property
    def type(self):
        return TRUTH_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.term._set_type(TRUTH_TYPE, signature)

    def __str__(self):
        variables = [self.variable]
        term = self.term
        while term.__class__ == self.__class__:
            variables.append(term.variable)
            term = term.term
        return (
            self.getQuantifier()
            + " "
            + " ".join("%s" % v for v in variables)
            + Tokens.DOT
            + "%s" % term
        )


class ExistsExpression(QuantifiedExpression):
    def getQuantifier(self):
        return Tokens.EXISTS


class AllExpression(QuantifiedExpression):
    def getQuantifier(self):
        return Tokens.ALL


class IotaExpression(QuantifiedExpression):
    def getQuantifier(self):
        return Tokens.IOTA


class NegatedExpression(Expression):
    def __init__(self, term):
        assert isinstance(term, Expression), "%s is not an Expression" % term
        self.term = term

    @property
    def type(self):
        return TRUTH_TYPE

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.term._set_type(TRUTH_TYPE, signature)

    def findtype(self, variable):
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        return self.term.findtype(variable)

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.term)])

    def negate(self):
        """:see: Expression.negate()"""
        return self.term

    def __eq__(self, other):
        return isinstance(other, NegatedExpression) and self.term == other.term

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__

    def __str__(self):
        return Tokens.NOT + "%s" % self.term


class BinaryExpression(Expression):
    def __init__(self, first, second):
        assert isinstance(first, Expression), "%s is not an Expression" % first
        assert isinstance(second, Expression), "%s is not an Expression" % second
        self.first = first
        self.second = second

    @property
    def type(self):
        return TRUTH_TYPE

    def findtype(self, variable):
        """:see Expression.findtype()"""
        assert isinstance(variable, Variable), "%s is not a Variable" % variable
        f = self.first.findtype(variable)
        s = self.second.findtype(variable)
        if f == s or s == ANY_TYPE:
            return f
        elif f == ANY_TYPE:
            return s
        else:
            return ANY_TYPE

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        return combinator([function(self.first), function(self.second)])

    def __eq__(self, other):
        return (
            (isinstance(self, other.__class__) or isinstance(other, self.__class__))
            and self.first == other.first
            and self.second == other.second
        )

    def __ne__(self, other):
        return not self == other

    __hash__ = Expression.__hash__

    def __str__(self):
        first = self._str_subex(self.first)
        second = self._str_subex(self.second)
        return Tokens.OPEN + first + " " + self.getOp() + " " + second + Tokens.CLOSE

    def _str_subex(self, subex):
        return "%s" % subex


class BooleanExpression(BinaryExpression):
    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.first._set_type(TRUTH_TYPE, signature)
        self.second._set_type(TRUTH_TYPE, signature)


class AndExpression(BooleanExpression):
    """This class represents conjunctions"""

    def getOp(self):
        return Tokens.AND

    def _str_subex(self, subex):
        s = "%s" % subex
        if isinstance(subex, AndExpression):
            return s[1:-1]
        return s


class OrExpression(BooleanExpression):
    """This class represents disjunctions"""

    def getOp(self):
        return Tokens.OR

    def _str_subex(self, subex):
        s = "%s" % subex
        if isinstance(subex, OrExpression):
            return s[1:-1]
        return s


class ImpExpression(BooleanExpression):
    """This class represents implications"""

    def getOp(self):
        return Tokens.IMP


class IffExpression(BooleanExpression):
    """This class represents biconditionals"""

    def getOp(self):
        return Tokens.IFF


class EqualityExpression(BinaryExpression):
    """This class represents equality expressions like "(x = y)"."""

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """:see Expression._set_type()"""
        assert isinstance(other_type, Type)

        if signature is None:
            signature = defaultdict(list)

        if not other_type.matches(TRUTH_TYPE):
            raise IllegalTypeException(self, other_type, TRUTH_TYPE)
        self.first._set_type(ENTITY_TYPE, signature)
        self.second._set_type(ENTITY_TYPE, signature)

    def getOp(self):
        return Tokens.EQ


### Utilities


class LogicalExpressionException(Exception):
    def __init__(self, index, message):
        self.index = index
        Exception.__init__(self, message)


class UnexpectedTokenException(LogicalExpressionException):
    def __init__(self, index, unexpected=None, expected=None, message=None):
        if unexpected and expected:
            msg = "Unexpected token: '%s'.  " "Expected token '%s'." % (
                unexpected,
                expected,
            )
        elif unexpected:
            msg = "Unexpected token: '%s'." % unexpected
            if message:
                msg += "  " + message
        else:
            msg = "Expected token '%s'." % expected
        LogicalExpressionException.__init__(self, index, msg)


class ExpectedMoreTokensException(LogicalExpressionException):
    def __init__(self, index, message=None):
        if not message:
            message = "More tokens expected."
        LogicalExpressionException.__init__(
            self, index, "End of input found.  " + message
        )


def is_indvar(expr):
    """
    An individual variable must be a single lowercase character other than 'e',
    followed by zero or more digits.

    :param expr: str
    :return: bool True if expr is of the correct form
    """
    assert isinstance(expr, str), "%s is not a string" % expr
    return re.match(r"^[a-df-z]\d*$", expr) is not None


def is_funcvar(expr):
    """
    A function variable must be a single uppercase character followed by
    zero or more digits.

    :param expr: str
    :return: bool True if expr is of the correct form
    """
    assert isinstance(expr, str), "%s is not a string" % expr
    return re.match(r"^[A-Z]\d*$", expr) is not None


def is_eventvar(expr):
    """
    An event variable must be a single lowercase 'e' character followed by
    zero or more digits.

    :param expr: str
    :return: bool True if expr is of the correct form
    """
    assert isinstance(expr, str), "%s is not a string" % expr
    return re.match(r"^e\d*$", expr) is not None


def demo():
    lexpr = Expression.fromstring
    print("=" * 20 + "Test reader" + "=" * 20)
    print(lexpr(r"john"))
    print(lexpr(r"man(x)"))
    print(lexpr(r"-man(x)"))
    print(lexpr(r"(man(x) & tall(x) & walks(x))"))
    print(lexpr(r"exists x.(man(x) & tall(x) & walks(x))"))
    print(lexpr(r"\x.man(x)"))
    print(lexpr(r"\x.man(x)(john)"))
    print(lexpr(r"\x y.sees(x,y)"))
    print(lexpr(r"\x y.sees(x,y)(a,b)"))
    print(lexpr(r"(\x.exists y.walks(x,y))(x)"))
    print(lexpr(r"exists x.x = y"))
    print(lexpr(r"exists x.(x = y)"))
    print(lexpr("P(x) & x=y & P(y)"))
    print(lexpr(r"\P Q.exists x.(P(x) & Q(x))"))
    print(lexpr(r"man(x) <-> tall(x)"))

    print("=" * 20 + "Test simplify" + "=" * 20)
    print(lexpr(r"\x.\y.sees(x,y)(john)(mary)").simplify())
    print(lexpr(r"\x.\y.sees(x,y)(john, mary)").simplify())
    print(lexpr(r"all x.(man(x) & (\x.exists y.walks(x,y))(x))").simplify())
    print(lexpr(r"(\P.\Q.exists x.(P(x) & Q(x)))(\x.dog(x))(\x.bark(x))").simplify())

    print("=" * 20 + "Test alpha conversion and binder expression equality" + "=" * 20)
    e1 = lexpr("exists x.P(x)")
    print(e1)
    e2 = e1.alpha_convert(Variable("z"))
    print(e2)
    print(e1 == e2)


def demo_errors():
    print("=" * 20 + "Test reader errors" + "=" * 20)
    demoException("(P(x) & Q(x)")
    demoException("((P(x) &) & Q(x))")
    demoException("P(x) -> ")
    demoException("P(x")
    demoException("P(x,")
    demoException("P(x,)")
    demoException("exists")
    demoException("exists x.")
    demoException("\\")
    demoException("\\ x y.")
    demoException("P(x)Q(x)")
    demoException("(P(x)Q(x)")
    demoException("exists x -> y")


def demoException(s):
    try:
        Expression.fromstring(s)
    except LogicalExpressionException as e:
        print(f"{e.__class__.__name__}: {e}")


def printtype(ex):
    print(f"{ex.str()} : {ex.type}")


if __name__ == "__main__":
    demo()
#    demo_errors()
