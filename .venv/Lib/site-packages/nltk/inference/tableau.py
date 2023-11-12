# Natural Language Toolkit: First-Order Tableau Theorem Prover
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Dan Garrette <dhgarrette@gmail.com>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Module for a tableau-based First Order theorem prover.
"""

from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
    AbstractVariableExpression,
    AllExpression,
    AndExpression,
    ApplicationExpression,
    EqualityExpression,
    ExistsExpression,
    Expression,
    FunctionVariableExpression,
    IffExpression,
    ImpExpression,
    LambdaExpression,
    NegatedExpression,
    OrExpression,
    Variable,
    VariableExpression,
    unique_variable,
)

_counter = Counter()


class ProverParseError(Exception):
    pass


class TableauProver(Prover):
    _assume_false = False

    def _prove(self, goal=None, assumptions=None, verbose=False):
        if not assumptions:
            assumptions = []

        result = None
        try:
            agenda = Agenda()
            if goal:
                agenda.put(-goal)
            agenda.put_all(assumptions)
            debugger = Debug(verbose)
            result = self._attempt_proof(agenda, set(), set(), debugger)
        except RuntimeError as e:
            if self._assume_false and str(e).startswith(
                "maximum recursion depth exceeded"
            ):
                result = False
            else:
                if verbose:
                    print(e)
                else:
                    raise e
        return (result, "\n".join(debugger.lines))

    def _attempt_proof(self, agenda, accessible_vars, atoms, debug):
        (current, context), category = agenda.pop_first()

        # if there's nothing left in the agenda, and we haven't closed the path
        if not current:
            debug.line("AGENDA EMPTY")
            return False

        proof_method = {
            Categories.ATOM: self._attempt_proof_atom,
            Categories.PROP: self._attempt_proof_prop,
            Categories.N_ATOM: self._attempt_proof_n_atom,
            Categories.N_PROP: self._attempt_proof_n_prop,
            Categories.APP: self._attempt_proof_app,
            Categories.N_APP: self._attempt_proof_n_app,
            Categories.N_EQ: self._attempt_proof_n_eq,
            Categories.D_NEG: self._attempt_proof_d_neg,
            Categories.N_ALL: self._attempt_proof_n_all,
            Categories.N_EXISTS: self._attempt_proof_n_some,
            Categories.AND: self._attempt_proof_and,
            Categories.N_OR: self._attempt_proof_n_or,
            Categories.N_IMP: self._attempt_proof_n_imp,
            Categories.OR: self._attempt_proof_or,
            Categories.IMP: self._attempt_proof_imp,
            Categories.N_AND: self._attempt_proof_n_and,
            Categories.IFF: self._attempt_proof_iff,
            Categories.N_IFF: self._attempt_proof_n_iff,
            Categories.EQ: self._attempt_proof_eq,
            Categories.EXISTS: self._attempt_proof_some,
            Categories.ALL: self._attempt_proof_all,
        }[category]

        debug.line((current, context))
        return proof_method(current, context, agenda, accessible_vars, atoms, debug)

    def _attempt_proof_atom(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        # Check if the branch is closed.  Return 'True' if it is
        if (current, True) in atoms:
            debug.line("CLOSED", 1)
            return True

        if context:
            if isinstance(context.term, NegatedExpression):
                current = current.negate()
            agenda.put(context(current).simplify())
            return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        else:
            # mark all AllExpressions as 'not exhausted' into the agenda since we are (potentially) adding new accessible vars
            agenda.mark_alls_fresh()
            return self._attempt_proof(
                agenda,
                accessible_vars | set(current.args),
                atoms | {(current, False)},
                debug + 1,
            )

    def _attempt_proof_n_atom(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        # Check if the branch is closed.  Return 'True' if it is
        if (current.term, False) in atoms:
            debug.line("CLOSED", 1)
            return True

        if context:
            if isinstance(context.term, NegatedExpression):
                current = current.negate()
            agenda.put(context(current).simplify())
            return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        else:
            # mark all AllExpressions as 'not exhausted' into the agenda since we are (potentially) adding new accessible vars
            agenda.mark_alls_fresh()
            return self._attempt_proof(
                agenda,
                accessible_vars | set(current.term.args),
                atoms | {(current.term, True)},
                debug + 1,
            )

    def _attempt_proof_prop(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        # Check if the branch is closed.  Return 'True' if it is
        if (current, True) in atoms:
            debug.line("CLOSED", 1)
            return True

        # mark all AllExpressions as 'not exhausted' into the agenda since we are (potentially) adding new accessible vars
        agenda.mark_alls_fresh()
        return self._attempt_proof(
            agenda, accessible_vars, atoms | {(current, False)}, debug + 1
        )

    def _attempt_proof_n_prop(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        # Check if the branch is closed.  Return 'True' if it is
        if (current.term, False) in atoms:
            debug.line("CLOSED", 1)
            return True

        # mark all AllExpressions as 'not exhausted' into the agenda since we are (potentially) adding new accessible vars
        agenda.mark_alls_fresh()
        return self._attempt_proof(
            agenda, accessible_vars, atoms | {(current.term, True)}, debug + 1
        )

    def _attempt_proof_app(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        f, args = current.uncurry()
        for i, arg in enumerate(args):
            if not TableauProver.is_atom(arg):
                ctx = f
                nv = Variable("X%s" % _counter.get())
                for j, a in enumerate(args):
                    ctx = ctx(VariableExpression(nv)) if i == j else ctx(a)
                if context:
                    ctx = context(ctx).simplify()
                ctx = LambdaExpression(nv, ctx)
                agenda.put(arg, ctx)
                return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        raise Exception("If this method is called, there must be a non-atomic argument")

    def _attempt_proof_n_app(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        f, args = current.term.uncurry()
        for i, arg in enumerate(args):
            if not TableauProver.is_atom(arg):
                ctx = f
                nv = Variable("X%s" % _counter.get())
                for j, a in enumerate(args):
                    ctx = ctx(VariableExpression(nv)) if i == j else ctx(a)
                if context:
                    # combine new context with existing
                    ctx = context(ctx).simplify()
                ctx = LambdaExpression(nv, -ctx)
                agenda.put(-arg, ctx)
                return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        raise Exception("If this method is called, there must be a non-atomic argument")

    def _attempt_proof_n_eq(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        ###########################################################################
        # Since 'current' is of type '~(a=b)', the path is closed if 'a' == 'b'
        ###########################################################################
        if current.term.first == current.term.second:
            debug.line("CLOSED", 1)
            return True

        agenda[Categories.N_EQ].add((current, context))
        current._exhausted = True
        return self._attempt_proof(
            agenda,
            accessible_vars | {current.term.first, current.term.second},
            atoms,
            debug + 1,
        )

    def _attempt_proof_d_neg(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        agenda.put(current.term.term, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_all(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        agenda[Categories.EXISTS].add(
            (ExistsExpression(current.term.variable, -current.term.term), context)
        )
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_some(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        agenda[Categories.ALL].add(
            (AllExpression(current.term.variable, -current.term.term), context)
        )
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_and(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        agenda.put(current.first, context)
        agenda.put(current.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_or(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        agenda.put(-current.term.first, context)
        agenda.put(-current.term.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_imp(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        agenda.put(current.term.first, context)
        agenda.put(-current.term.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_or(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        new_agenda = agenda.clone()
        agenda.put(current.first, context)
        new_agenda.put(current.second, context)
        return self._attempt_proof(
            agenda, accessible_vars, atoms, debug + 1
        ) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_imp(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        new_agenda = agenda.clone()
        agenda.put(-current.first, context)
        new_agenda.put(current.second, context)
        return self._attempt_proof(
            agenda, accessible_vars, atoms, debug + 1
        ) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_and(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        new_agenda = agenda.clone()
        agenda.put(-current.term.first, context)
        new_agenda.put(-current.term.second, context)
        return self._attempt_proof(
            agenda, accessible_vars, atoms, debug + 1
        ) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_iff(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        new_agenda = agenda.clone()
        agenda.put(current.first, context)
        agenda.put(current.second, context)
        new_agenda.put(-current.first, context)
        new_agenda.put(-current.second, context)
        return self._attempt_proof(
            agenda, accessible_vars, atoms, debug + 1
        ) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_iff(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        new_agenda = agenda.clone()
        agenda.put(current.term.first, context)
        agenda.put(-current.term.second, context)
        new_agenda.put(-current.term.first, context)
        new_agenda.put(current.term.second, context)
        return self._attempt_proof(
            agenda, accessible_vars, atoms, debug + 1
        ) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_eq(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        #########################################################################
        # Since 'current' is of the form '(a = b)', replace ALL free instances
        # of 'a' with 'b'
        #########################################################################
        agenda.put_atoms(atoms)
        agenda.replace_all(current.first, current.second)
        accessible_vars.discard(current.first)
        agenda.mark_neqs_fresh()
        return self._attempt_proof(agenda, accessible_vars, set(), debug + 1)

    def _attempt_proof_some(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        new_unique_variable = VariableExpression(unique_variable())
        agenda.put(current.term.replace(current.variable, new_unique_variable), context)
        agenda.mark_alls_fresh()
        return self._attempt_proof(
            agenda, accessible_vars | {new_unique_variable}, atoms, debug + 1
        )

    def _attempt_proof_all(
        self, current, context, agenda, accessible_vars, atoms, debug
    ):
        try:
            current._used_vars
        except AttributeError:
            current._used_vars = set()

        # if there are accessible_vars on the path
        if accessible_vars:
            # get the set of bound variables that have not be used by this AllExpression
            bv_available = accessible_vars - current._used_vars

            if bv_available:
                variable_to_use = list(bv_available)[0]
                debug.line("--> Using '%s'" % variable_to_use, 2)
                current._used_vars |= {variable_to_use}
                agenda.put(
                    current.term.replace(current.variable, variable_to_use), context
                )
                agenda[Categories.ALL].add((current, context))
                return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

            else:
                # no more available variables to substitute
                debug.line("--> Variables Exhausted", 2)
                current._exhausted = True
                agenda[Categories.ALL].add((current, context))
                return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

        else:
            new_unique_variable = VariableExpression(unique_variable())
            debug.line("--> Using '%s'" % new_unique_variable, 2)
            current._used_vars |= {new_unique_variable}
            agenda.put(
                current.term.replace(current.variable, new_unique_variable), context
            )
            agenda[Categories.ALL].add((current, context))
            agenda.mark_alls_fresh()
            return self._attempt_proof(
                agenda, accessible_vars | {new_unique_variable}, atoms, debug + 1
            )

    @staticmethod
    def is_atom(e):
        if isinstance(e, NegatedExpression):
            e = e.term

        if isinstance(e, ApplicationExpression):
            for arg in e.args:
                if not TableauProver.is_atom(arg):
                    return False
            return True
        elif isinstance(e, AbstractVariableExpression) or isinstance(
            e, LambdaExpression
        ):
            return True
        else:
            return False


class TableauProverCommand(BaseProverCommand):
    def __init__(self, goal=None, assumptions=None, prover=None):
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        """
        if prover is not None:
            assert isinstance(prover, TableauProver)
        else:
            prover = TableauProver()

        BaseProverCommand.__init__(self, prover, goal, assumptions)


class Agenda:
    def __init__(self):
        self.sets = tuple(set() for i in range(21))

    def clone(self):
        new_agenda = Agenda()
        set_list = [s.copy() for s in self.sets]

        new_allExs = set()
        for allEx, _ in set_list[Categories.ALL]:
            new_allEx = AllExpression(allEx.variable, allEx.term)
            try:
                new_allEx._used_vars = {used for used in allEx._used_vars}
            except AttributeError:
                new_allEx._used_vars = set()
            new_allExs.add((new_allEx, None))
        set_list[Categories.ALL] = new_allExs

        set_list[Categories.N_EQ] = {
            (NegatedExpression(n_eq.term), ctx)
            for (n_eq, ctx) in set_list[Categories.N_EQ]
        }

        new_agenda.sets = tuple(set_list)
        return new_agenda

    def __getitem__(self, index):
        return self.sets[index]

    def put(self, expression, context=None):
        if isinstance(expression, AllExpression):
            ex_to_add = AllExpression(expression.variable, expression.term)
            try:
                ex_to_add._used_vars = {used for used in expression._used_vars}
            except AttributeError:
                ex_to_add._used_vars = set()
        else:
            ex_to_add = expression
        self.sets[self._categorize_expression(ex_to_add)].add((ex_to_add, context))

    def put_all(self, expressions):
        for expression in expressions:
            self.put(expression)

    def put_atoms(self, atoms):
        for atom, neg in atoms:
            if neg:
                self[Categories.N_ATOM].add((-atom, None))
            else:
                self[Categories.ATOM].add((atom, None))

    def pop_first(self):
        """Pop the first expression that appears in the agenda"""
        for i, s in enumerate(self.sets):
            if s:
                if i in [Categories.N_EQ, Categories.ALL]:
                    for ex in s:
                        try:
                            if not ex[0]._exhausted:
                                s.remove(ex)
                                return (ex, i)
                        except AttributeError:
                            s.remove(ex)
                            return (ex, i)
                else:
                    return (s.pop(), i)
        return ((None, None), None)

    def replace_all(self, old, new):
        for s in self.sets:
            for ex, ctx in s:
                ex.replace(old.variable, new)
                if ctx is not None:
                    ctx.replace(old.variable, new)

    def mark_alls_fresh(self):
        for u, _ in self.sets[Categories.ALL]:
            u._exhausted = False

    def mark_neqs_fresh(self):
        for neq, _ in self.sets[Categories.N_EQ]:
            neq._exhausted = False

    def _categorize_expression(self, current):
        if isinstance(current, NegatedExpression):
            return self._categorize_NegatedExpression(current)
        elif isinstance(current, FunctionVariableExpression):
            return Categories.PROP
        elif TableauProver.is_atom(current):
            return Categories.ATOM
        elif isinstance(current, AllExpression):
            return Categories.ALL
        elif isinstance(current, AndExpression):
            return Categories.AND
        elif isinstance(current, OrExpression):
            return Categories.OR
        elif isinstance(current, ImpExpression):
            return Categories.IMP
        elif isinstance(current, IffExpression):
            return Categories.IFF
        elif isinstance(current, EqualityExpression):
            return Categories.EQ
        elif isinstance(current, ExistsExpression):
            return Categories.EXISTS
        elif isinstance(current, ApplicationExpression):
            return Categories.APP
        else:
            raise ProverParseError("cannot categorize %s" % current.__class__.__name__)

    def _categorize_NegatedExpression(self, current):
        negated = current.term

        if isinstance(negated, NegatedExpression):
            return Categories.D_NEG
        elif isinstance(negated, FunctionVariableExpression):
            return Categories.N_PROP
        elif TableauProver.is_atom(negated):
            return Categories.N_ATOM
        elif isinstance(negated, AllExpression):
            return Categories.N_ALL
        elif isinstance(negated, AndExpression):
            return Categories.N_AND
        elif isinstance(negated, OrExpression):
            return Categories.N_OR
        elif isinstance(negated, ImpExpression):
            return Categories.N_IMP
        elif isinstance(negated, IffExpression):
            return Categories.N_IFF
        elif isinstance(negated, EqualityExpression):
            return Categories.N_EQ
        elif isinstance(negated, ExistsExpression):
            return Categories.N_EXISTS
        elif isinstance(negated, ApplicationExpression):
            return Categories.N_APP
        else:
            raise ProverParseError("cannot categorize %s" % negated.__class__.__name__)


class Debug:
    def __init__(self, verbose, indent=0, lines=None):
        self.verbose = verbose
        self.indent = indent

        if not lines:
            lines = []
        self.lines = lines

    def __add__(self, increment):
        return Debug(self.verbose, self.indent + 1, self.lines)

    def line(self, data, indent=0):
        if isinstance(data, tuple):
            ex, ctx = data
            if ctx:
                data = f"{ex}, {ctx}"
            else:
                data = "%s" % ex

            if isinstance(ex, AllExpression):
                try:
                    used_vars = "[%s]" % (
                        ",".join("%s" % ve.variable.name for ve in ex._used_vars)
                    )
                    data += ":   %s" % used_vars
                except AttributeError:
                    data += ":   []"

        newline = "{}{}".format("   " * (self.indent + indent), data)
        self.lines.append(newline)

        if self.verbose:
            print(newline)


class Categories:
    ATOM = 0
    PROP = 1
    N_ATOM = 2
    N_PROP = 3
    APP = 4
    N_APP = 5
    N_EQ = 6
    D_NEG = 7
    N_ALL = 8
    N_EXISTS = 9
    AND = 10
    N_OR = 11
    N_IMP = 12
    OR = 13
    IMP = 14
    N_AND = 15
    IFF = 16
    N_IFF = 17
    EQ = 18
    EXISTS = 19
    ALL = 20


def testTableauProver():
    tableau_test("P | -P")
    tableau_test("P & -P")
    tableau_test("Q", ["P", "(P -> Q)"])
    tableau_test("man(x)")
    tableau_test("(man(x) -> man(x))")
    tableau_test("(man(x) -> --man(x))")
    tableau_test("-(man(x) and -man(x))")
    tableau_test("(man(x) or -man(x))")
    tableau_test("(man(x) -> man(x))")
    tableau_test("-(man(x) and -man(x))")
    tableau_test("(man(x) or -man(x))")
    tableau_test("(man(x) -> man(x))")
    tableau_test("(man(x) iff man(x))")
    tableau_test("-(man(x) iff -man(x))")
    tableau_test("all x.man(x)")
    tableau_test("all x.all y.((x = y) -> (y = x))")
    tableau_test("all x.all y.all z.(((x = y) & (y = z)) -> (x = z))")
    #    tableau_test('-all x.some y.F(x,y) & some x.all y.(-F(x,y))')
    #    tableau_test('some x.all y.sees(x,y)')

    p1 = "all x.(man(x) -> mortal(x))"
    p2 = "man(Socrates)"
    c = "mortal(Socrates)"
    tableau_test(c, [p1, p2])

    p1 = "all x.(man(x) -> walks(x))"
    p2 = "man(John)"
    c = "some y.walks(y)"
    tableau_test(c, [p1, p2])

    p = "((x = y) & walks(y))"
    c = "walks(x)"
    tableau_test(c, [p])

    p = "((x = y) & ((y = z) & (z = w)))"
    c = "(x = w)"
    tableau_test(c, [p])

    p = "some e1.some e2.(believe(e1,john,e2) & walk(e2,mary))"
    c = "some e0.walk(e0,mary)"
    tableau_test(c, [p])

    c = "(exists x.exists z3.((x = Mary) & ((z3 = John) & sees(z3,x))) <-> exists x.exists z4.((x = John) & ((z4 = Mary) & sees(x,z4))))"
    tableau_test(c)


#    p = 'some e1.some e2.((believe e1 john e2) and (walk e2 mary))'
#    c = 'some x.some e3.some e4.((believe e3 x e4) and (walk e4 mary))'
#    tableau_test(c, [p])


def testHigherOrderTableauProver():
    tableau_test("believe(j, -lie(b))", ["believe(j, -lie(b) & -cheat(b))"])
    tableau_test("believe(j, lie(b) & cheat(b))", ["believe(j, lie(b))"])
    tableau_test(
        "believe(j, lie(b))", ["lie(b)"]
    )  # how do we capture that John believes all things that are true
    tableau_test(
        "believe(j, know(b, cheat(b)))",
        ["believe(j, know(b, lie(b)) & know(b, steals(b) & cheat(b)))"],
    )
    tableau_test("P(Q(y), R(y) & R(z))", ["P(Q(x) & Q(y), R(y) & R(z))"])

    tableau_test("believe(j, cheat(b) & lie(b))", ["believe(j, lie(b) & cheat(b))"])
    tableau_test("believe(j, -cheat(b) & -lie(b))", ["believe(j, -lie(b) & -cheat(b))"])


def tableau_test(c, ps=None, verbose=False):
    pc = Expression.fromstring(c)
    pps = [Expression.fromstring(p) for p in ps] if ps else []
    if not ps:
        ps = []
    print(
        "%s |- %s: %s"
        % (", ".join(ps), pc, TableauProver().prove(pc, pps, verbose=verbose))
    )


def demo():
    testTableauProver()
    testHigherOrderTableauProver()


if __name__ == "__main__":
    demo()
