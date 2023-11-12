# Natural Language Toolkit: Nonmonotonic Reasoning
#
# Author: Daniel H. Garrette <dhgarrette@gmail.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A module to perform nonmonotonic reasoning.  The ideas and demonstrations in
this module are based on "Logical Foundations of Artificial Intelligence" by
Michael R. Genesereth and Nils J. Nilsson.
"""

from collections import defaultdict
from functools import reduce

from nltk.inference.api import Prover, ProverCommandDecorator
from nltk.inference.prover9 import Prover9, Prover9Command
from nltk.sem.logic import (
    AbstractVariableExpression,
    AllExpression,
    AndExpression,
    ApplicationExpression,
    BooleanExpression,
    EqualityExpression,
    ExistsExpression,
    Expression,
    ImpExpression,
    NegatedExpression,
    Variable,
    VariableExpression,
    operator,
    unique_variable,
)


class ProverParseError(Exception):
    pass


def get_domain(goal, assumptions):
    if goal is None:
        all_expressions = assumptions
    else:
        all_expressions = assumptions + [-goal]
    return reduce(operator.or_, (a.constants() for a in all_expressions), set())


class ClosedDomainProver(ProverCommandDecorator):
    """
    This is a prover decorator that adds domain closure assumptions before
    proving.
    """

    def assumptions(self):
        assumptions = [a for a in self._command.assumptions()]
        goal = self._command.goal()
        domain = get_domain(goal, assumptions)
        return [self.replace_quants(ex, domain) for ex in assumptions]

    def goal(self):
        goal = self._command.goal()
        domain = get_domain(goal, self._command.assumptions())
        return self.replace_quants(goal, domain)

    def replace_quants(self, ex, domain):
        """
        Apply the closed domain assumption to the expression

        - Domain = union([e.free()|e.constants() for e in all_expressions])
        - translate "exists x.P" to "(z=d1 | z=d2 | ... ) & P.replace(x,z)" OR
                    "P.replace(x, d1) | P.replace(x, d2) | ..."
        - translate "all x.P" to "P.replace(x, d1) & P.replace(x, d2) & ..."

        :param ex: ``Expression``
        :param domain: set of {Variable}s
        :return: ``Expression``
        """
        if isinstance(ex, AllExpression):
            conjuncts = [
                ex.term.replace(ex.variable, VariableExpression(d)) for d in domain
            ]
            conjuncts = [self.replace_quants(c, domain) for c in conjuncts]
            return reduce(lambda x, y: x & y, conjuncts)
        elif isinstance(ex, BooleanExpression):
            return ex.__class__(
                self.replace_quants(ex.first, domain),
                self.replace_quants(ex.second, domain),
            )
        elif isinstance(ex, NegatedExpression):
            return -self.replace_quants(ex.term, domain)
        elif isinstance(ex, ExistsExpression):
            disjuncts = [
                ex.term.replace(ex.variable, VariableExpression(d)) for d in domain
            ]
            disjuncts = [self.replace_quants(d, domain) for d in disjuncts]
            return reduce(lambda x, y: x | y, disjuncts)
        else:
            return ex


class UniqueNamesProver(ProverCommandDecorator):
    """
    This is a prover decorator that adds unique names assumptions before
    proving.
    """

    def assumptions(self):
        """
        - Domain = union([e.free()|e.constants() for e in all_expressions])
        - if "d1 = d2" cannot be proven from the premises, then add "d1 != d2"
        """
        assumptions = self._command.assumptions()

        domain = list(get_domain(self._command.goal(), assumptions))

        # build a dictionary of obvious equalities
        eq_sets = SetHolder()
        for a in assumptions:
            if isinstance(a, EqualityExpression):
                av = a.first.variable
                bv = a.second.variable
                # put 'a' and 'b' in the same set
                eq_sets[av].add(bv)

        new_assumptions = []
        for i, a in enumerate(domain):
            for b in domain[i + 1 :]:
                # if a and b are not already in the same equality set
                if b not in eq_sets[a]:
                    newEqEx = EqualityExpression(
                        VariableExpression(a), VariableExpression(b)
                    )
                    if Prover9().prove(newEqEx, assumptions):
                        # we can prove that the names are the same entity.
                        # remember that they are equal so we don't re-check.
                        eq_sets[a].add(b)
                    else:
                        # we can't prove it, so assume unique names
                        new_assumptions.append(-newEqEx)

        return assumptions + new_assumptions


class SetHolder(list):
    """
    A list of sets of Variables.
    """

    def __getitem__(self, item):
        """
        :param item: ``Variable``
        :return: the set containing 'item'
        """
        assert isinstance(item, Variable)
        for s in self:
            if item in s:
                return s
        # item is not found in any existing set.  so create a new set
        new = {item}
        self.append(new)
        return new


class ClosedWorldProver(ProverCommandDecorator):
    """
    This is a prover decorator that completes predicates before proving.

    If the assumptions contain "P(A)", then "all x.(P(x) -> (x=A))" is the completion of "P".
    If the assumptions contain "all x.(ostrich(x) -> bird(x))", then "all x.(bird(x) -> ostrich(x))" is the completion of "bird".
    If the assumptions don't contain anything that are "P", then "all x.-P(x)" is the completion of "P".

    walk(Socrates)
    Socrates != Bill
    + all x.(walk(x) -> (x=Socrates))
    ----------------
    -walk(Bill)

    see(Socrates, John)
    see(John, Mary)
    Socrates != John
    John != Mary
    + all x.all y.(see(x,y) -> ((x=Socrates & y=John) | (x=John & y=Mary)))
    ----------------
    -see(Socrates, Mary)

    all x.(ostrich(x) -> bird(x))
    bird(Tweety)
    -ostrich(Sam)
    Sam != Tweety
    + all x.(bird(x) -> (ostrich(x) | x=Tweety))
    + all x.-ostrich(x)
    -------------------
    -bird(Sam)
    """

    def assumptions(self):
        assumptions = self._command.assumptions()

        predicates = self._make_predicate_dict(assumptions)

        new_assumptions = []
        for p in predicates:
            predHolder = predicates[p]
            new_sig = self._make_unique_signature(predHolder)
            new_sig_exs = [VariableExpression(v) for v in new_sig]

            disjuncts = []

            # Turn the signatures into disjuncts
            for sig in predHolder.signatures:
                equality_exs = []
                for v1, v2 in zip(new_sig_exs, sig):
                    equality_exs.append(EqualityExpression(v1, v2))
                disjuncts.append(reduce(lambda x, y: x & y, equality_exs))

            # Turn the properties into disjuncts
            for prop in predHolder.properties:
                # replace variables from the signature with new sig variables
                bindings = {}
                for v1, v2 in zip(new_sig_exs, prop[0]):
                    bindings[v2] = v1
                disjuncts.append(prop[1].substitute_bindings(bindings))

            # make the assumption
            if disjuncts:
                # disjuncts exist, so make an implication
                antecedent = self._make_antecedent(p, new_sig)
                consequent = reduce(lambda x, y: x | y, disjuncts)
                accum = ImpExpression(antecedent, consequent)
            else:
                # nothing has property 'p'
                accum = NegatedExpression(self._make_antecedent(p, new_sig))

            # quantify the implication
            for new_sig_var in new_sig[::-1]:
                accum = AllExpression(new_sig_var, accum)
            new_assumptions.append(accum)

        return assumptions + new_assumptions

    def _make_unique_signature(self, predHolder):
        """
        This method figures out how many arguments the predicate takes and
        returns a tuple containing that number of unique variables.
        """
        return tuple(unique_variable() for i in range(predHolder.signature_len))

    def _make_antecedent(self, predicate, signature):
        """
        Return an application expression with 'predicate' as the predicate
        and 'signature' as the list of arguments.
        """
        antecedent = predicate
        for v in signature:
            antecedent = antecedent(VariableExpression(v))
        return antecedent

    def _make_predicate_dict(self, assumptions):
        """
        Create a dictionary of predicates from the assumptions.

        :param assumptions: a list of ``Expression``s
        :return: dict mapping ``AbstractVariableExpression`` to ``PredHolder``
        """
        predicates = defaultdict(PredHolder)
        for a in assumptions:
            self._map_predicates(a, predicates)
        return predicates

    def _map_predicates(self, expression, predDict):
        if isinstance(expression, ApplicationExpression):
            func, args = expression.uncurry()
            if isinstance(func, AbstractVariableExpression):
                predDict[func].append_sig(tuple(args))
        elif isinstance(expression, AndExpression):
            self._map_predicates(expression.first, predDict)
            self._map_predicates(expression.second, predDict)
        elif isinstance(expression, AllExpression):
            # collect all the universally quantified variables
            sig = [expression.variable]
            term = expression.term
            while isinstance(term, AllExpression):
                sig.append(term.variable)
                term = term.term
            if isinstance(term, ImpExpression):
                if isinstance(term.first, ApplicationExpression) and isinstance(
                    term.second, ApplicationExpression
                ):
                    func1, args1 = term.first.uncurry()
                    func2, args2 = term.second.uncurry()
                    if (
                        isinstance(func1, AbstractVariableExpression)
                        and isinstance(func2, AbstractVariableExpression)
                        and sig == [v.variable for v in args1]
                        and sig == [v.variable for v in args2]
                    ):
                        predDict[func2].append_prop((tuple(sig), term.first))
                        predDict[func1].validate_sig_len(sig)


class PredHolder:
    """
    This class will be used by a dictionary that will store information
    about predicates to be used by the ``ClosedWorldProver``.

    The 'signatures' property is a list of tuples defining signatures for
    which the predicate is true.  For instance, 'see(john, mary)' would be
    result in the signature '(john,mary)' for 'see'.

    The second element of the pair is a list of pairs such that the first
    element of the pair is a tuple of variables and the second element is an
    expression of those variables that makes the predicate true.  For instance,
    'all x.all y.(see(x,y) -> know(x,y))' would result in "((x,y),('see(x,y)'))"
    for 'know'.
    """

    def __init__(self):
        self.signatures = []
        self.properties = []
        self.signature_len = None

    def append_sig(self, new_sig):
        self.validate_sig_len(new_sig)
        self.signatures.append(new_sig)

    def append_prop(self, new_prop):
        self.validate_sig_len(new_prop[0])
        self.properties.append(new_prop)

    def validate_sig_len(self, new_sig):
        if self.signature_len is None:
            self.signature_len = len(new_sig)
        elif self.signature_len != len(new_sig):
            raise Exception("Signature lengths do not match")

    def __str__(self):
        return f"({self.signatures},{self.properties},{self.signature_len})"

    def __repr__(self):
        return "%s" % self


def closed_domain_demo():
    lexpr = Expression.fromstring

    p1 = lexpr(r"exists x.walk(x)")
    p2 = lexpr(r"man(Socrates)")
    c = lexpr(r"walk(Socrates)")
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print("assumptions:")
    for a in cdp.assumptions():
        print("   ", a)
    print("goal:", cdp.goal())
    print(cdp.prove())

    p1 = lexpr(r"exists x.walk(x)")
    p2 = lexpr(r"man(Socrates)")
    p3 = lexpr(r"-walk(Bill)")
    c = lexpr(r"walk(Socrates)")
    prover = Prover9Command(c, [p1, p2, p3])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print("assumptions:")
    for a in cdp.assumptions():
        print("   ", a)
    print("goal:", cdp.goal())
    print(cdp.prove())

    p1 = lexpr(r"exists x.walk(x)")
    p2 = lexpr(r"man(Socrates)")
    p3 = lexpr(r"-walk(Bill)")
    c = lexpr(r"walk(Socrates)")
    prover = Prover9Command(c, [p1, p2, p3])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print("assumptions:")
    for a in cdp.assumptions():
        print("   ", a)
    print("goal:", cdp.goal())
    print(cdp.prove())

    p1 = lexpr(r"walk(Socrates)")
    p2 = lexpr(r"walk(Bill)")
    c = lexpr(r"all x.walk(x)")
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print("assumptions:")
    for a in cdp.assumptions():
        print("   ", a)
    print("goal:", cdp.goal())
    print(cdp.prove())

    p1 = lexpr(r"girl(mary)")
    p2 = lexpr(r"dog(rover)")
    p3 = lexpr(r"all x.(girl(x) -> -dog(x))")
    p4 = lexpr(r"all x.(dog(x) -> -girl(x))")
    p5 = lexpr(r"chase(mary, rover)")
    c = lexpr(r"exists y.(dog(y) & all x.(girl(x) -> chase(x,y)))")
    prover = Prover9Command(c, [p1, p2, p3, p4, p5])
    print(prover.prove())
    cdp = ClosedDomainProver(prover)
    print("assumptions:")
    for a in cdp.assumptions():
        print("   ", a)
    print("goal:", cdp.goal())
    print(cdp.prove())


def unique_names_demo():
    lexpr = Expression.fromstring

    p1 = lexpr(r"man(Socrates)")
    p2 = lexpr(r"man(Bill)")
    c = lexpr(r"exists x.exists y.(x != y)")
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    unp = UniqueNamesProver(prover)
    print("assumptions:")
    for a in unp.assumptions():
        print("   ", a)
    print("goal:", unp.goal())
    print(unp.prove())

    p1 = lexpr(r"all x.(walk(x) -> (x = Socrates))")
    p2 = lexpr(r"Bill = William")
    p3 = lexpr(r"Bill = Billy")
    c = lexpr(r"-walk(William)")
    prover = Prover9Command(c, [p1, p2, p3])
    print(prover.prove())
    unp = UniqueNamesProver(prover)
    print("assumptions:")
    for a in unp.assumptions():
        print("   ", a)
    print("goal:", unp.goal())
    print(unp.prove())


def closed_world_demo():
    lexpr = Expression.fromstring

    p1 = lexpr(r"walk(Socrates)")
    p2 = lexpr(r"(Socrates != Bill)")
    c = lexpr(r"-walk(Bill)")
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    cwp = ClosedWorldProver(prover)
    print("assumptions:")
    for a in cwp.assumptions():
        print("   ", a)
    print("goal:", cwp.goal())
    print(cwp.prove())

    p1 = lexpr(r"see(Socrates, John)")
    p2 = lexpr(r"see(John, Mary)")
    p3 = lexpr(r"(Socrates != John)")
    p4 = lexpr(r"(John != Mary)")
    c = lexpr(r"-see(Socrates, Mary)")
    prover = Prover9Command(c, [p1, p2, p3, p4])
    print(prover.prove())
    cwp = ClosedWorldProver(prover)
    print("assumptions:")
    for a in cwp.assumptions():
        print("   ", a)
    print("goal:", cwp.goal())
    print(cwp.prove())

    p1 = lexpr(r"all x.(ostrich(x) -> bird(x))")
    p2 = lexpr(r"bird(Tweety)")
    p3 = lexpr(r"-ostrich(Sam)")
    p4 = lexpr(r"Sam != Tweety")
    c = lexpr(r"-bird(Sam)")
    prover = Prover9Command(c, [p1, p2, p3, p4])
    print(prover.prove())
    cwp = ClosedWorldProver(prover)
    print("assumptions:")
    for a in cwp.assumptions():
        print("   ", a)
    print("goal:", cwp.goal())
    print(cwp.prove())


def combination_prover_demo():
    lexpr = Expression.fromstring

    p1 = lexpr(r"see(Socrates, John)")
    p2 = lexpr(r"see(John, Mary)")
    c = lexpr(r"-see(Socrates, Mary)")
    prover = Prover9Command(c, [p1, p2])
    print(prover.prove())
    command = ClosedDomainProver(UniqueNamesProver(ClosedWorldProver(prover)))
    for a in command.assumptions():
        print(a)
    print(command.prove())


def default_reasoning_demo():
    lexpr = Expression.fromstring

    premises = []

    # define taxonomy
    premises.append(lexpr(r"all x.(elephant(x)        -> animal(x))"))
    premises.append(lexpr(r"all x.(bird(x)            -> animal(x))"))
    premises.append(lexpr(r"all x.(dove(x)            -> bird(x))"))
    premises.append(lexpr(r"all x.(ostrich(x)         -> bird(x))"))
    premises.append(lexpr(r"all x.(flying_ostrich(x)  -> ostrich(x))"))

    # default properties
    premises.append(
        lexpr(r"all x.((animal(x)  & -Ab1(x)) -> -fly(x))")
    )  # normal animals don't fly
    premises.append(
        lexpr(r"all x.((bird(x)    & -Ab2(x)) -> fly(x))")
    )  # normal birds fly
    premises.append(
        lexpr(r"all x.((ostrich(x) & -Ab3(x)) -> -fly(x))")
    )  # normal ostriches don't fly

    # specify abnormal entities
    premises.append(lexpr(r"all x.(bird(x)           -> Ab1(x))"))  # flight
    premises.append(lexpr(r"all x.(ostrich(x)        -> Ab2(x))"))  # non-flying bird
    premises.append(lexpr(r"all x.(flying_ostrich(x) -> Ab3(x))"))  # flying ostrich

    # define entities
    premises.append(lexpr(r"elephant(E)"))
    premises.append(lexpr(r"dove(D)"))
    premises.append(lexpr(r"ostrich(O)"))

    # print the assumptions
    prover = Prover9Command(None, premises)
    command = UniqueNamesProver(ClosedWorldProver(prover))
    for a in command.assumptions():
        print(a)

    print_proof("-fly(E)", premises)
    print_proof("fly(D)", premises)
    print_proof("-fly(O)", premises)


def print_proof(goal, premises):
    lexpr = Expression.fromstring
    prover = Prover9Command(lexpr(goal), premises)
    command = UniqueNamesProver(ClosedWorldProver(prover))
    print(goal, prover.prove(), command.prove())


def demo():
    closed_domain_demo()
    unique_names_demo()
    closed_world_demo()
    combination_prover_demo()
    default_reasoning_demo()


if __name__ == "__main__":
    demo()
