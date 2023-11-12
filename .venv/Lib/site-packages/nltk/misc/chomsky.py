# Chomsky random text generator, version 1.1, Raymond Hettinger, 2005/09/13
# https://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/440546

"""
CHOMSKY is an aid to writing linguistic papers in the style
of the great master.  It is based on selected phrases taken
from actual books and articles written by Noam Chomsky.
Upon request, it assembles the phrases in the elegant
stylistic patterns that Chomsky is noted for.
To generate n sentences of linguistic wisdom, type

    (CHOMSKY n)  -- for example
    (CHOMSKY 5) generates half a screen of linguistic truth.
"""

leadins = """To characterize a linguistic level L,
    On the other hand,
    This suggests that
    It appears that
    Furthermore,
    We will bring evidence in favor of the following thesis:
    To provide a constituent structure for T(Z,K),
    From C1, it follows that
    For any transformation which is sufficiently diversified in \
application to be of any interest,
    Analogously,
    Clearly,
    Note that
    Of course,
    Suppose, for instance, that
    Thus
    With this clarification,
    Conversely,
    We have already seen that
    By combining adjunctions and certain deformations,
    I suggested that these results would follow from the assumption that
    If the position of the trace in (99c) were only relatively \
inaccessible to movement,
    However, this assumption is not correct, since
    Comparing these examples with their parasitic gap counterparts in \
(96) and (97), we see that
    In the discussion of resumptive pronouns following (81),
    So far,
    Nevertheless,
    For one thing,
    Summarizing, then, we assume that
    A consequence of the approach just outlined is that
    Presumably,
    On our assumptions,
    It may be, then, that
    It must be emphasized, once again, that
    Let us continue to suppose that
    Notice, incidentally, that """
# List of LEADINs to buy time.

subjects = """ the notion of level of grammaticalness
    a case of semigrammaticalness of a different sort
    most of the methodological work in modern linguistics
    a subset of English sentences interesting on quite independent grounds
    the natural general principle that will subsume this case
    an important property of these three types of EC
    any associated supporting element
    the appearance of parasitic gaps in domains relatively inaccessible \
to ordinary extraction
    the speaker-hearer's linguistic intuition
    the descriptive power of the base component
    the earlier discussion of deviance
    this analysis of a formative as a pair of sets of features
    this selectionally introduced contextual feature
    a descriptively adequate grammar
    the fundamental error of regarding functional notions as categorial
    relational information
    the systematic use of complex symbols
    the theory of syntactic features developed earlier"""
# List of SUBJECTs chosen for maximum professorial macho.

verbs = """can be defined in such a way as to impose
    delimits
    suffices to account for
    cannot be arbitrary in
    is not subject to
    does not readily tolerate
    raises serious doubts about
    is not quite equivalent to
    does not affect the structure of
    may remedy and, at the same time, eliminate
    is not to be considered in determining
    is to be regarded as
    is unspecified with respect to
    is, apparently, determined by
    is necessary to impose an interpretation on
    appears to correlate rather closely with
    is rather different from"""
# List of VERBs chosen for autorecursive obfuscation.

objects = """ problems of phonemic and morphological analysis.
    a corpus of utterance tokens upon which conformity has been defined \
by the paired utterance test.
    the traditional practice of grammarians.
    the levels of acceptability from fairly high (e.g. (99a)) to virtual \
gibberish (e.g. (98d)).
    a stipulation to place the constructions into these various categories.
    a descriptive fact.
    a parasitic gap construction.
    the extended c-command discussed in connection with (34).
    the ultimate standard that determines the accuracy of any proposed grammar.
    the system of base rules exclusive of the lexicon.
    irrelevant intervening contexts in selectional rules.
    nondistinctness in the sense of distinctive feature theory.
    a general convention regarding the forms of the grammar.
    an abstract underlying order.
    an important distinction in language use.
    the requirement that branching is not tolerated within the dominance \
scope of a complex symbol.
    the strong generative capacity of the theory."""
# List of OBJECTs selected for profound sententiousness.

import random
import textwrap
from itertools import chain, islice


def generate_chomsky(times=5, line_length=72):
    parts = []
    for part in (leadins, subjects, verbs, objects):
        phraselist = list(map(str.strip, part.splitlines()))
        random.shuffle(phraselist)
        parts.append(phraselist)
    output = chain.from_iterable(islice(zip(*parts), 0, times))
    print(textwrap.fill(" ".join(output), line_length))


if __name__ == "__main__":
    generate_chomsky()
