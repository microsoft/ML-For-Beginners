# This file is part of Patsy
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This file has the code that figures out how each factor in some given Term
# should be coded. This is complicated by dealing with models with categorical
# factors like:
#   1 + a + a:b
# then technically 'a' (which represents the space of vectors that can be
# produced as linear combinations of the dummy coding of the levels of the
# factor a) is collinear with the intercept, and 'a:b' (which represents the
# space of vectors that can be produced as linear combinations of the dummy
# coding *of a new factor whose levels are the cartesian product of a and b)
# is collinear with both 'a' and the intercept.
#
# In such a case, the rule is that we find some way to code each term so that
# the full space of vectors that it represents *is present in the model* BUT
# there is no collinearity between the different terms. In effect, we have to
# choose a set of vectors that spans everything that that term wants to span,
# *except* that part of the vector space which was already spanned by earlier
# terms.

# How? We replace each term with the set of "subterms" that it covers, like
# so:
#   1 -> ()
#   a -> (), a-
#   a:b -> (), a-, b-, a-:b-
# where "-" means "coded so as not to span the intercept". So that example
# above expands to
#   [()] + [() + a-] + [() + a- + b- + a-:b-]
# so we go through from left to right, and for each term we:
#   1) toss out all the subterms that have already been used (this is a simple
#      equality test, no magic)
#   2) simplify the terms that are left, according to rules like
#        () + a- = a+
#      (here + means, "coded to span the intercept")
#   3) use the resulting subterm list as our coding for this term!
# So in the above, we go:
#   (): stays the same, coded as intercept
#   () + a-: reduced to just a-, which is what we code
#   () + a- + b- + a-:b-: reduced to b- + a-:b-, which is simplified to a+:b-.

from __future__ import print_function

from patsy.util import no_pickling

# This should really be a named tuple, but those don't exist until Python
# 2.6...
class _ExpandedFactor(object):
    """A factor, with an additional annotation for whether it is coded
    full-rank (includes_intercept=True) or not.

    These objects are treated as immutable."""
    def __init__(self, includes_intercept, factor):
        self.includes_intercept = includes_intercept
        self.factor = factor

    def __hash__(self):
        return hash((_ExpandedFactor, self.includes_intercept, self.factor))

    def __eq__(self, other):
        return (isinstance(other, _ExpandedFactor)
                and other.includes_intercept == self.includes_intercept
                and other.factor == self.factor)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if self.includes_intercept:
            suffix = "+"
        else:
            suffix = "-"
        return "%r%s" % (self.factor, suffix)

    __getstate__ = no_pickling

class _Subterm(object):
    "Also immutable."
    def __init__(self, efactors):
        self.efactors = frozenset(efactors)

    def can_absorb(self, other):
        # returns True if 'self' is like a-:b-, and 'other' is like a-
        return (len(self.efactors) - len(other.efactors) == 1
                and self.efactors.issuperset(other.efactors))

    def absorb(self, other):
        diff = self.efactors.difference(other.efactors)
        assert len(diff) == 1
        efactor = list(diff)[0]
        assert not efactor.includes_intercept
        new_factors = set(other.efactors)
        new_factors.add(_ExpandedFactor(True, efactor.factor))
        return _Subterm(new_factors)

    def __hash__(self):
        return hash((_Subterm, self.efactors))

    def __eq__(self, other):
        return (isinstance(other, _Subterm)
                and self.efactors == self.efactors)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, list(self.efactors))

    __getstate__ = no_pickling

# For testing: takes a shorthand description of a list of subterms like
#   [(), ("a-",), ("a-", "b+")]
# and expands it into a list of _Subterm and _ExpandedFactor objects.
def _expand_test_abbrevs(short_subterms):
    subterms = []
    for subterm in short_subterms:
        factors = []
        for factor_name in subterm:
            assert factor_name[-1] in ("+", "-")
            factors.append(_ExpandedFactor(factor_name[-1] == "+",
                                           factor_name[:-1]))
        subterms.append(_Subterm(factors))
    return subterms

def test__Subterm():
    s_ab = _expand_test_abbrevs([["a-", "b-"]])[0]
    s_abc = _expand_test_abbrevs([["a-", "b-", "c-"]])[0]
    s_null = _expand_test_abbrevs([[]])[0]
    s_cd = _expand_test_abbrevs([["c-", "d-"]])[0]
    s_a = _expand_test_abbrevs([["a-"]])[0]
    s_ap = _expand_test_abbrevs([["a+"]])[0]
    s_abp = _expand_test_abbrevs([["a-", "b+"]])[0]
    for bad in s_abc, s_null, s_cd, s_ap, s_abp:
        assert not s_ab.can_absorb(bad)
    assert s_ab.can_absorb(s_a)
    assert s_ab.absorb(s_a) == s_abp

# Importantly, this preserves the order of the input. Both the items inside
# each subset are in the order they were in the original tuple, and the tuples
# are emitted so that they're sorted with respect to their elements position
# in the original tuple.
def _subsets_sorted(tupl):
    def helper(seq):
        if not seq:
            yield ()
        else:
            obj = seq[0]
            for subset in _subsets_sorted(seq[1:]):
                yield subset
                yield (obj,) + subset
    # Transform each obj -> (idx, obj) tuple, so that we can later sort them
    # by their position in the original list.
    expanded = list(enumerate(tupl))
    expanded_subsets = list(helper(expanded))
    # This exploits Python's stable sort: we want short before long, and ties
    # broken by natural ordering on the (idx, obj) entries in each subset. So
    # we sort by the latter first, then by the former.
    expanded_subsets.sort()
    expanded_subsets.sort(key=len)
    # And finally, we strip off the idx's:
    for subset in expanded_subsets:
        yield tuple([obj for (idx, obj) in subset])
    
def test__subsets_sorted():
    assert list(_subsets_sorted((1, 2))) == [(), (1,), (2,), (1, 2)]
    assert (list(_subsets_sorted((1, 2, 3)))
            == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)])
    assert len(list(_subsets_sorted(range(5)))) == 2 ** 5

def _simplify_one_subterm(subterms):
    # We simplify greedily from left to right.
    # Returns True if succeeded, False otherwise
    for short_i, short_subterm in enumerate(subterms):
        for long_i, long_subterm in enumerate(subterms[short_i + 1:]):
            if long_subterm.can_absorb(short_subterm):
                new_subterm = long_subterm.absorb(short_subterm)
                subterms[short_i + 1 + long_i] = new_subterm
                subterms.pop(short_i)
                return True
    return False
            
def _simplify_subterms(subterms):
    while _simplify_one_subterm(subterms):
        pass

def test__simplify_subterms():
    def t(given, expected):
        given = _expand_test_abbrevs(given)
        expected = _expand_test_abbrevs(expected)
        print("testing if:", given, "->", expected)
        _simplify_subterms(given)
        assert given == expected
    t([("a-",)], [("a-",)])
    t([(), ("a-",)], [("a+",)])
    t([(), ("a-",), ("b-",), ("a-", "b-")], [("a+", "b+")])
    t([(), ("a-",), ("a-", "b-")], [("a+",), ("a-", "b-")])
    t([("a-",), ("b-",), ("a-", "b-")], [("b-",), ("a-", "b+")])

# 'term' is a Term
# 'numeric_factors' is any set-like object which lists the
#   numeric/non-categorical factors in this term. Such factors are just
#   ignored by this routine.
# 'used_subterms' is a set which records which subterms have previously been
#   used. E.g., a:b has subterms (), a, b, a:b, and if we're processing
#    y ~ a + a:b
#   then by the time we reach a:b, the () and a subterms will have already
#   been used. This is an in/out argument, and should be treated as opaque by
#   callers -- really it is a way for multiple invocations of this routine to
#   talk to each other. Each time it is called, this routine adds the subterms
#   of each factor to this set in place. So the first time this routine is
#   called, pass in an empty set, and then just keep passing the same set to
#   any future calls.
# Returns: a list of dicts. Each dict maps from factors to booleans. The
# coding for the given term should use a full-rank contrast for those factors
# which map to True, a (n-1)-rank contrast for those factors which map to
# False, and any factors which are not mentioned are numeric and should be
# added back in. These dicts should add columns to the design matrix from left
# to right.
def pick_contrasts_for_term(term, numeric_factors, used_subterms):
    categorical_factors = [f for f in term.factors if f not in numeric_factors]
    # Converts a term into an expanded list of subterms like:
    #   a:b  ->  1 + a- + b- + a-:b-
    # and discards the ones that have already been used.
    subterms = []
    for subset in _subsets_sorted(categorical_factors):
        subterm = _Subterm([_ExpandedFactor(False, f) for f in subset])
        if subterm not in used_subterms:
            subterms.append(subterm)
    used_subterms.update(subterms)
    _simplify_subterms(subterms)
    factor_codings = []
    for subterm in subterms:
        factor_coding = {}
        for expanded in subterm.efactors:
            factor_coding[expanded.factor] = expanded.includes_intercept
        factor_codings.append(factor_coding)
    return factor_codings

def test_pick_contrasts_for_term():
    from patsy.desc import Term
    used = set()
    codings = pick_contrasts_for_term(Term([]), set(), used)
    assert codings == [{}]
    codings = pick_contrasts_for_term(Term(["a", "x"]), set(["x"]), used)
    assert codings == [{"a": False}]
    codings = pick_contrasts_for_term(Term(["a", "b"]), set(), used)
    assert codings == [{"a": True, "b": False}]
    used_snapshot = set(used)
    codings = pick_contrasts_for_term(Term(["c", "d"]), set(), used)
    assert codings == [{"d": False}, {"c": False, "d": True}]
    # Do it again backwards, to make sure we're deterministic with respect to
    # order:
    codings = pick_contrasts_for_term(Term(["d", "c"]), set(), used_snapshot)
    assert codings == [{"c": False}, {"c": True, "d": False}]
