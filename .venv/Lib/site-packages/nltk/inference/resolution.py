# Natural Language Toolkit: First-order Resolution-based Theorem Prover
#
# Author: Dan Garrette <dhgarrette@gmail.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Module for a resolution-based First Order theorem prover.
"""

import operator
from collections import defaultdict
from functools import reduce

from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
    AndExpression,
    ApplicationExpression,
    EqualityExpression,
    Expression,
    IndividualVariableExpression,
    NegatedExpression,
    OrExpression,
    Variable,
    VariableExpression,
    is_indvar,
    unique_variable,
)


class ProverParseError(Exception):
    pass


class ResolutionProver(Prover):
    ANSWER_KEY = "ANSWER"
    _assume_false = True

    def _prove(self, goal=None, assumptions=None, verbose=False):
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in the proof
        :type assumptions: list(sem.Expression)
        """
        if not assumptions:
            assumptions = []

        result = None
        try:
            clauses = []
            if goal:
                clauses.extend(clausify(-goal))
            for a in assumptions:
                clauses.extend(clausify(a))
            result, clauses = self._attempt_proof(clauses)
            if verbose:
                print(ResolutionProverCommand._decorate_clauses(clauses))
        except RuntimeError as e:
            if self._assume_false and str(e).startswith(
                "maximum recursion depth exceeded"
            ):
                result = False
                clauses = []
            else:
                if verbose:
                    print(e)
                else:
                    raise e
        return (result, clauses)

    def _attempt_proof(self, clauses):
        # map indices to lists of indices, to store attempted unifications
        tried = defaultdict(list)

        i = 0
        while i < len(clauses):
            if not clauses[i].is_tautology():
                # since we try clauses in order, we should start after the last
                # index tried
                if tried[i]:
                    j = tried[i][-1] + 1
                else:
                    j = i + 1  # nothing tried yet for 'i', so start with the next

                while j < len(clauses):
                    # don't: 1) unify a clause with itself,
                    #       2) use tautologies
                    if i != j and j and not clauses[j].is_tautology():
                        tried[i].append(j)
                        newclauses = clauses[i].unify(clauses[j])
                        if newclauses:
                            for newclause in newclauses:
                                newclause._parents = (i + 1, j + 1)
                                clauses.append(newclause)
                                if not len(newclause):  # if there's an empty clause
                                    return (True, clauses)
                            i = -1  # since we added a new clause, restart from the top
                            break
                    j += 1
            i += 1
        return (False, clauses)


class ResolutionProverCommand(BaseProverCommand):
    def __init__(self, goal=None, assumptions=None, prover=None):
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        """
        if prover is not None:
            assert isinstance(prover, ResolutionProver)
        else:
            prover = ResolutionProver()

        BaseProverCommand.__init__(self, prover, goal, assumptions)
        self._clauses = None

    def prove(self, verbose=False):
        """
        Perform the actual proof.  Store the result to prevent unnecessary
        re-proving.
        """
        if self._result is None:
            self._result, clauses = self._prover._prove(
                self.goal(), self.assumptions(), verbose
            )
            self._clauses = clauses
            self._proof = ResolutionProverCommand._decorate_clauses(clauses)
        return self._result

    def find_answers(self, verbose=False):
        self.prove(verbose)

        answers = set()
        answer_ex = VariableExpression(Variable(ResolutionProver.ANSWER_KEY))
        for clause in self._clauses:
            for term in clause:
                if (
                    isinstance(term, ApplicationExpression)
                    and term.function == answer_ex
                    and not isinstance(term.argument, IndividualVariableExpression)
                ):
                    answers.add(term.argument)
        return answers

    @staticmethod
    def _decorate_clauses(clauses):
        """
        Decorate the proof output.
        """
        out = ""
        max_clause_len = max(len(str(clause)) for clause in clauses)
        max_seq_len = len(str(len(clauses)))
        for i in range(len(clauses)):
            parents = "A"
            taut = ""
            if clauses[i].is_tautology():
                taut = "Tautology"
            if clauses[i]._parents:
                parents = str(clauses[i]._parents)
            parents = " " * (max_clause_len - len(str(clauses[i])) + 1) + parents
            seq = " " * (max_seq_len - len(str(i + 1))) + str(i + 1)
            out += f"[{seq}] {clauses[i]} {parents} {taut}\n"
        return out


class Clause(list):
    def __init__(self, data):
        list.__init__(self, data)
        self._is_tautology = None
        self._parents = None

    def unify(self, other, bindings=None, used=None, skipped=None, debug=False):
        """
        Attempt to unify this Clause with the other, returning a list of
        resulting, unified, Clauses.

        :param other: ``Clause`` with which to unify
        :param bindings: ``BindingDict`` containing bindings that should be used
            during the unification
        :param used: tuple of two lists of atoms.  The first lists the
            atoms from 'self' that were successfully unified with atoms from
            'other'.  The second lists the atoms from 'other' that were successfully
            unified with atoms from 'self'.
        :param skipped: tuple of two ``Clause`` objects.  The first is a list of all
            the atoms from the 'self' Clause that have not been unified with
            anything on the path.  The second is same thing for the 'other' Clause.
        :param debug: bool indicating whether debug statements should print
        :return: list containing all the resulting ``Clause`` objects that could be
            obtained by unification
        """
        if bindings is None:
            bindings = BindingDict()
        if used is None:
            used = ([], [])
        if skipped is None:
            skipped = ([], [])
        if isinstance(debug, bool):
            debug = DebugObject(debug)

        newclauses = _iterate_first(
            self, other, bindings, used, skipped, _complete_unify_path, debug
        )

        # remove subsumed clauses.  make a list of all indices of subsumed
        # clauses, and then remove them from the list
        subsumed = []
        for i, c1 in enumerate(newclauses):
            if i not in subsumed:
                for j, c2 in enumerate(newclauses):
                    if i != j and j not in subsumed and c1.subsumes(c2):
                        subsumed.append(j)
        result = []
        for i in range(len(newclauses)):
            if i not in subsumed:
                result.append(newclauses[i])

        return result

    def isSubsetOf(self, other):
        """
        Return True iff every term in 'self' is a term in 'other'.

        :param other: ``Clause``
        :return: bool
        """
        for a in self:
            if a not in other:
                return False
        return True

    def subsumes(self, other):
        """
        Return True iff 'self' subsumes 'other', this is, if there is a
        substitution such that every term in 'self' can be unified with a term
        in 'other'.

        :param other: ``Clause``
        :return: bool
        """
        negatedother = []
        for atom in other:
            if isinstance(atom, NegatedExpression):
                negatedother.append(atom.term)
            else:
                negatedother.append(-atom)

        negatedotherClause = Clause(negatedother)

        bindings = BindingDict()
        used = ([], [])
        skipped = ([], [])
        debug = DebugObject(False)

        return (
            len(
                _iterate_first(
                    self,
                    negatedotherClause,
                    bindings,
                    used,
                    skipped,
                    _subsumes_finalize,
                    debug,
                )
            )
            > 0
        )

    def __getslice__(self, start, end):
        return Clause(list.__getslice__(self, start, end))

    def __sub__(self, other):
        return Clause([a for a in self if a not in other])

    def __add__(self, other):
        return Clause(list.__add__(self, other))

    def is_tautology(self):
        """
        Self is a tautology if it contains ground terms P and -P.  The ground
        term, P, must be an exact match, ie, not using unification.
        """
        if self._is_tautology is not None:
            return self._is_tautology
        for i, a in enumerate(self):
            if not isinstance(a, EqualityExpression):
                j = len(self) - 1
                while j > i:
                    b = self[j]
                    if isinstance(a, NegatedExpression):
                        if a.term == b:
                            self._is_tautology = True
                            return True
                    elif isinstance(b, NegatedExpression):
                        if a == b.term:
                            self._is_tautology = True
                            return True
                    j -= 1
        self._is_tautology = False
        return False

    def free(self):
        return reduce(operator.or_, ((atom.free() | atom.constants()) for atom in self))

    def replace(self, variable, expression):
        """
        Replace every instance of variable with expression across every atom
        in the clause

        :param variable: ``Variable``
        :param expression: ``Expression``
        """
        return Clause([atom.replace(variable, expression) for atom in self])

    def substitute_bindings(self, bindings):
        """
        Replace every binding

        :param bindings: A list of tuples mapping Variable Expressions to the
            Expressions to which they are bound.
        :return: ``Clause``
        """
        return Clause([atom.substitute_bindings(bindings) for atom in self])

    def __str__(self):
        return "{" + ", ".join("%s" % item for item in self) + "}"

    def __repr__(self):
        return "%s" % self


def _iterate_first(first, second, bindings, used, skipped, finalize_method, debug):
    """
    This method facilitates movement through the terms of 'self'
    """
    debug.line(f"unify({first},{second}) {bindings}")

    if not len(first) or not len(second):  # if no more recursions can be performed
        return finalize_method(first, second, bindings, used, skipped, debug)
    else:
        # explore this 'self' atom
        result = _iterate_second(
            first, second, bindings, used, skipped, finalize_method, debug + 1
        )

        # skip this possible 'self' atom
        newskipped = (skipped[0] + [first[0]], skipped[1])
        result += _iterate_first(
            first[1:], second, bindings, used, newskipped, finalize_method, debug + 1
        )

        try:
            newbindings, newused, unused = _unify_terms(
                first[0], second[0], bindings, used
            )
            # Unification found, so progress with this line of unification
            # put skipped and unused terms back into play for later unification.
            newfirst = first[1:] + skipped[0] + unused[0]
            newsecond = second[1:] + skipped[1] + unused[1]
            result += _iterate_first(
                newfirst,
                newsecond,
                newbindings,
                newused,
                ([], []),
                finalize_method,
                debug + 1,
            )
        except BindingException:
            # the atoms could not be unified,
            pass

        return result


def _iterate_second(first, second, bindings, used, skipped, finalize_method, debug):
    """
    This method facilitates movement through the terms of 'other'
    """
    debug.line(f"unify({first},{second}) {bindings}")

    if not len(first) or not len(second):  # if no more recursions can be performed
        return finalize_method(first, second, bindings, used, skipped, debug)
    else:
        # skip this possible pairing and move to the next
        newskipped = (skipped[0], skipped[1] + [second[0]])
        result = _iterate_second(
            first, second[1:], bindings, used, newskipped, finalize_method, debug + 1
        )

        try:
            newbindings, newused, unused = _unify_terms(
                first[0], second[0], bindings, used
            )
            # Unification found, so progress with this line of unification
            # put skipped and unused terms back into play for later unification.
            newfirst = first[1:] + skipped[0] + unused[0]
            newsecond = second[1:] + skipped[1] + unused[1]
            result += _iterate_second(
                newfirst,
                newsecond,
                newbindings,
                newused,
                ([], []),
                finalize_method,
                debug + 1,
            )
        except BindingException:
            # the atoms could not be unified,
            pass

        return result


def _unify_terms(a, b, bindings=None, used=None):
    """
    This method attempts to unify two terms.  Two expressions are unifiable
    if there exists a substitution function S such that S(a) == S(-b).

    :param a: ``Expression``
    :param b: ``Expression``
    :param bindings: ``BindingDict`` a starting set of bindings with which
    the unification must be consistent
    :return: ``BindingDict`` A dictionary of the bindings required to unify
    :raise ``BindingException``: If the terms cannot be unified
    """
    assert isinstance(a, Expression)
    assert isinstance(b, Expression)

    if bindings is None:
        bindings = BindingDict()
    if used is None:
        used = ([], [])

    # Use resolution
    if isinstance(a, NegatedExpression) and isinstance(b, ApplicationExpression):
        newbindings = most_general_unification(a.term, b, bindings)
        newused = (used[0] + [a], used[1] + [b])
        unused = ([], [])
    elif isinstance(a, ApplicationExpression) and isinstance(b, NegatedExpression):
        newbindings = most_general_unification(a, b.term, bindings)
        newused = (used[0] + [a], used[1] + [b])
        unused = ([], [])

    # Use demodulation
    elif isinstance(a, EqualityExpression):
        newbindings = BindingDict([(a.first.variable, a.second)])
        newused = (used[0] + [a], used[1])
        unused = ([], [b])
    elif isinstance(b, EqualityExpression):
        newbindings = BindingDict([(b.first.variable, b.second)])
        newused = (used[0], used[1] + [b])
        unused = ([a], [])

    else:
        raise BindingException((a, b))

    return newbindings, newused, unused


def _complete_unify_path(first, second, bindings, used, skipped, debug):
    if used[0] or used[1]:  # if bindings were made along the path
        newclause = Clause(skipped[0] + skipped[1] + first + second)
        debug.line("  -> New Clause: %s" % newclause)
        return [newclause.substitute_bindings(bindings)]
    else:  # no bindings made means no unification occurred.  so no result
        debug.line("  -> End")
        return []


def _subsumes_finalize(first, second, bindings, used, skipped, debug):
    if not len(skipped[0]) and not len(first):
        # If there are no skipped terms and no terms left in 'first', then
        # all of the terms in the original 'self' were unified with terms
        # in 'other'.  Therefore, there exists a binding (this one) such that
        # every term in self can be unified with a term in other, which
        # is the definition of subsumption.
        return [True]
    else:
        return []


def clausify(expression):
    """
    Skolemize, clausify, and standardize the variables apart.
    """
    clause_list = []
    for clause in _clausify(skolemize(expression)):
        for free in clause.free():
            if is_indvar(free.name):
                newvar = VariableExpression(unique_variable())
                clause = clause.replace(free, newvar)
        clause_list.append(clause)
    return clause_list


def _clausify(expression):
    """
    :param expression: a skolemized expression in CNF
    """
    if isinstance(expression, AndExpression):
        return _clausify(expression.first) + _clausify(expression.second)
    elif isinstance(expression, OrExpression):
        first = _clausify(expression.first)
        second = _clausify(expression.second)
        assert len(first) == 1
        assert len(second) == 1
        return [first[0] + second[0]]
    elif isinstance(expression, EqualityExpression):
        return [Clause([expression])]
    elif isinstance(expression, ApplicationExpression):
        return [Clause([expression])]
    elif isinstance(expression, NegatedExpression):
        if isinstance(expression.term, ApplicationExpression):
            return [Clause([expression])]
        elif isinstance(expression.term, EqualityExpression):
            return [Clause([expression])]
    raise ProverParseError()


class BindingDict:
    def __init__(self, binding_list=None):
        """
        :param binding_list: list of (``AbstractVariableExpression``, ``AtomicExpression``) to initialize the dictionary
        """
        self.d = {}

        if binding_list:
            for (v, b) in binding_list:
                self[v] = b

    def __setitem__(self, variable, binding):
        """
        A binding is consistent with the dict if its variable is not already bound, OR if its
        variable is already bound to its argument.

        :param variable: ``Variable`` The variable to bind
        :param binding: ``Expression`` The atomic to which 'variable' should be bound
        :raise BindingException: If the variable cannot be bound in this dictionary
        """
        assert isinstance(variable, Variable)
        assert isinstance(binding, Expression)

        try:
            existing = self[variable]
        except KeyError:
            existing = None

        if not existing or binding == existing:
            self.d[variable] = binding
        elif isinstance(binding, IndividualVariableExpression):
            # Since variable is already bound, try to bind binding to variable
            try:
                existing = self[binding.variable]
            except KeyError:
                existing = None

            binding2 = VariableExpression(variable)

            if not existing or binding2 == existing:
                self.d[binding.variable] = binding2
            else:
                raise BindingException(
                    "Variable %s already bound to another " "value" % (variable)
                )
        else:
            raise BindingException(
                "Variable %s already bound to another " "value" % (variable)
            )

    def __getitem__(self, variable):
        """
        Return the expression to which 'variable' is bound
        """
        assert isinstance(variable, Variable)

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
        :raise BindingException: If the parameter dictionaries are not consistent with each other
        """
        try:
            combined = BindingDict()
            for v in self.d:
                combined[v] = self.d[v]
            for v in other.d:
                combined[v] = other.d[v]
            return combined
        except BindingException as e:
            raise BindingException(
                "Attempting to add two contradicting "
                "BindingDicts: '%s' and '%s'" % (self, other)
            ) from e

    def __len__(self):
        return len(self.d)

    def __str__(self):
        data_str = ", ".join(f"{v}: {self.d[v]}" for v in sorted(self.d.keys()))
        return "{" + data_str + "}"

    def __repr__(self):
        return "%s" % self


def most_general_unification(a, b, bindings=None):
    """
    Find the most general unification of the two given expressions

    :param a: ``Expression``
    :param b: ``Expression``
    :param bindings: ``BindingDict`` a starting set of bindings with which the
                     unification must be consistent
    :return: a list of bindings
    :raise BindingException: if the Expressions cannot be unified
    """
    if bindings is None:
        bindings = BindingDict()

    if a == b:
        return bindings
    elif isinstance(a, IndividualVariableExpression):
        return _mgu_var(a, b, bindings)
    elif isinstance(b, IndividualVariableExpression):
        return _mgu_var(b, a, bindings)
    elif isinstance(a, ApplicationExpression) and isinstance(b, ApplicationExpression):
        return most_general_unification(
            a.function, b.function, bindings
        ) + most_general_unification(a.argument, b.argument, bindings)
    raise BindingException((a, b))


def _mgu_var(var, expression, bindings):
    if var.variable in expression.free() | expression.constants():
        raise BindingException((var, expression))
    else:
        return BindingDict([(var.variable, expression)]) + bindings


class BindingException(Exception):
    def __init__(self, arg):
        if isinstance(arg, tuple):
            Exception.__init__(self, "'%s' cannot be bound to '%s'" % arg)
        else:
            Exception.__init__(self, arg)


class UnificationException(Exception):
    def __init__(self, a, b):
        Exception.__init__(self, f"'{a}' cannot unify with '{b}'")


class DebugObject:
    def __init__(self, enabled=True, indent=0):
        self.enabled = enabled
        self.indent = indent

    def __add__(self, i):
        return DebugObject(self.enabled, self.indent + i)

    def line(self, line):
        if self.enabled:
            print("    " * self.indent + line)


def testResolutionProver():
    resolution_test(r"man(x)")
    resolution_test(r"(man(x) -> man(x))")
    resolution_test(r"(man(x) -> --man(x))")
    resolution_test(r"-(man(x) and -man(x))")
    resolution_test(r"(man(x) or -man(x))")
    resolution_test(r"(man(x) -> man(x))")
    resolution_test(r"-(man(x) and -man(x))")
    resolution_test(r"(man(x) or -man(x))")
    resolution_test(r"(man(x) -> man(x))")
    resolution_test(r"(man(x) iff man(x))")
    resolution_test(r"-(man(x) iff -man(x))")
    resolution_test("all x.man(x)")
    resolution_test("-all x.some y.F(x,y) & some x.all y.(-F(x,y))")
    resolution_test("some x.all y.sees(x,y)")

    p1 = Expression.fromstring(r"all x.(man(x) -> mortal(x))")
    p2 = Expression.fromstring(r"man(Socrates)")
    c = Expression.fromstring(r"mortal(Socrates)")
    print(f"{p1}, {p2} |- {c}: {ResolutionProver().prove(c, [p1, p2])}")

    p1 = Expression.fromstring(r"all x.(man(x) -> walks(x))")
    p2 = Expression.fromstring(r"man(John)")
    c = Expression.fromstring(r"some y.walks(y)")
    print(f"{p1}, {p2} |- {c}: {ResolutionProver().prove(c, [p1, p2])}")

    p = Expression.fromstring(r"some e1.some e2.(believe(e1,john,e2) & walk(e2,mary))")
    c = Expression.fromstring(r"some e0.walk(e0,mary)")
    print(f"{p} |- {c}: {ResolutionProver().prove(c, [p])}")


def resolution_test(e):
    f = Expression.fromstring(e)
    t = ResolutionProver().prove(f)
    print(f"|- {f}: {t}")


def test_clausify():
    lexpr = Expression.fromstring

    print(clausify(lexpr("P(x) | Q(x)")))
    print(clausify(lexpr("(P(x) & Q(x)) | R(x)")))
    print(clausify(lexpr("P(x) | (Q(x) & R(x))")))
    print(clausify(lexpr("(P(x) & Q(x)) | (R(x) & S(x))")))

    print(clausify(lexpr("P(x) | Q(x) | R(x)")))
    print(clausify(lexpr("P(x) | (Q(x) & R(x)) | S(x)")))

    print(clausify(lexpr("exists x.P(x) | Q(x)")))

    print(clausify(lexpr("-(-P(x) & Q(x))")))
    print(clausify(lexpr("P(x) <-> Q(x)")))
    print(clausify(lexpr("-(P(x) <-> Q(x))")))
    print(clausify(lexpr("-(all x.P(x))")))
    print(clausify(lexpr("-(some x.P(x))")))

    print(clausify(lexpr("some x.P(x)")))
    print(clausify(lexpr("some x.all y.P(x,y)")))
    print(clausify(lexpr("all y.some x.P(x,y)")))
    print(clausify(lexpr("all z.all y.some x.P(x,y,z)")))
    print(clausify(lexpr("all x.(all y.P(x,y) -> -all y.(Q(x,y) -> R(x,y)))")))


def demo():
    test_clausify()
    print()
    testResolutionProver()
    print()

    p = Expression.fromstring("man(x)")
    print(ResolutionProverCommand(p, [p]).prove())


if __name__ == "__main__":
    demo()
