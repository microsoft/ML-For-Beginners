# Natural Language Toolkit: WordNet
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bethard <Steven.Bethard@colorado.edu>
#         Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
#         Nitin Madnani <nmadnani@ets.org>
#         Nasruddin A’aidil Shari
#         Sim Wei Ying Geraldine
#         Soe Lynn
#         Francis Bond <bond@ieee.org>
#         Eric Kafe <kafe.eric@gmail.com>

# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
An NLTK interface for WordNet

WordNet is a lexical database of English.
Using synsets, helps find conceptual relationships between words
such as hypernyms, hyponyms, synonyms, antonyms etc.

For details about WordNet see:
https://wordnet.princeton.edu/

This module also allows you to find lemmas in languages
other than English from the Open Multilingual Wordnet
https://omwn.org/

"""

import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter

from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file

######################################################################
# Table of Contents
######################################################################
# - Constants
# - Data Classes
#   - WordNetError
#   - Lemma
#   - Synset
# - WordNet Corpus Reader
# - WordNet Information Content Corpus Reader
# - Similarity Metrics
# - Demo

######################################################################
# Constants
######################################################################

#: Positive infinity (for similarity functions)
_INF = 1e300

# { Part-of-speech constants
ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
# }

POS_LIST = [NOUN, VERB, ADJ, ADV]

# A table of strings that are used to express verb frames.
VERB_FRAME_STRINGS = (
    None,
    "Something %s",
    "Somebody %s",
    "It is %sing",
    "Something is %sing PP",
    "Something %s something Adjective/Noun",
    "Something %s Adjective/Noun",
    "Somebody %s Adjective",
    "Somebody %s something",
    "Somebody %s somebody",
    "Something %s somebody",
    "Something %s something",
    "Something %s to somebody",
    "Somebody %s on something",
    "Somebody %s somebody something",
    "Somebody %s something to somebody",
    "Somebody %s something from somebody",
    "Somebody %s somebody with something",
    "Somebody %s somebody of something",
    "Somebody %s something on somebody",
    "Somebody %s somebody PP",
    "Somebody %s something PP",
    "Somebody %s PP",
    "Somebody's (body part) %s",
    "Somebody %s somebody to INFINITIVE",
    "Somebody %s somebody INFINITIVE",
    "Somebody %s that CLAUSE",
    "Somebody %s to somebody",
    "Somebody %s to INFINITIVE",
    "Somebody %s whether INFINITIVE",
    "Somebody %s somebody into V-ing something",
    "Somebody %s something with something",
    "Somebody %s INFINITIVE",
    "Somebody %s VERB-ing",
    "It %s that CLAUSE",
    "Something %s INFINITIVE",
    # OEWN additions:
    "Somebody %s at something",
    "Somebody %s for something",
    "Somebody %s on somebody",
    "Somebody %s out of somebody",
)

SENSENUM_RE = re.compile(r"\.[\d]+\.")


######################################################################
# Data Classes
######################################################################


class WordNetError(Exception):
    """An exception class for wordnet-related errors."""


@total_ordering
class _WordNetObject:
    """A common base class for lemmas and synsets."""

    def hypernyms(self):
        return self._related("@")

    def _hypernyms(self):
        return self._related("@")

    def instance_hypernyms(self):
        return self._related("@i")

    def _instance_hypernyms(self):
        return self._related("@i")

    def hyponyms(self):
        return self._related("~")

    def instance_hyponyms(self):
        return self._related("~i")

    def member_holonyms(self):
        return self._related("#m")

    def substance_holonyms(self):
        return self._related("#s")

    def part_holonyms(self):
        return self._related("#p")

    def member_meronyms(self):
        return self._related("%m")

    def substance_meronyms(self):
        return self._related("%s")

    def part_meronyms(self):
        return self._related("%p")

    def topic_domains(self):
        return self._related(";c")

    def in_topic_domains(self):
        return self._related("-c")

    def region_domains(self):
        return self._related(";r")

    def in_region_domains(self):
        return self._related("-r")

    def usage_domains(self):
        return self._related(";u")

    def in_usage_domains(self):
        return self._related("-u")

    def attributes(self):
        return self._related("=")

    def entailments(self):
        return self._related("*")

    def causes(self):
        return self._related(">")

    def also_sees(self):
        return self._related("^")

    def verb_groups(self):
        return self._related("$")

    def similar_tos(self):
        return self._related("&")

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        return self._name == other._name

    def __ne__(self, other):
        return self._name != other._name

    def __lt__(self, other):
        return self._name < other._name


class Lemma(_WordNetObject):
    """
    The lexical entry for a single morphological form of a
    sense-disambiguated word.

    Create a Lemma from a "<word>.<pos>.<number>.<lemma>" string where:
    <word> is the morphological stem identifying the synset
    <pos> is one of the module attributes ADJ, ADJ_SAT, ADV, NOUN or VERB
    <number> is the sense number, counting from 0.
    <lemma> is the morphological form of interest

    Note that <word> and <lemma> can be different, e.g. the Synset
    'salt.n.03' has the Lemmas 'salt.n.03.salt', 'salt.n.03.saltiness' and
    'salt.n.03.salinity'.

    Lemma attributes, accessible via methods with the same name:

    - name: The canonical name of this lemma.
    - synset: The synset that this lemma belongs to.
    - syntactic_marker: For adjectives, the WordNet string identifying the
      syntactic position relative modified noun. See:
      https://wordnet.princeton.edu/documentation/wninput5wn
      For all other parts of speech, this attribute is None.
    - count: The frequency of this lemma in wordnet.

    Lemma methods:

    Lemmas have the following methods for retrieving related Lemmas. They
    correspond to the names for the pointer symbols defined here:
    https://wordnet.princeton.edu/documentation/wninput5wn
    These methods all return lists of Lemmas:

    - antonyms
    - hypernyms, instance_hypernyms
    - hyponyms, instance_hyponyms
    - member_holonyms, substance_holonyms, part_holonyms
    - member_meronyms, substance_meronyms, part_meronyms
    - topic_domains, region_domains, usage_domains
    - attributes
    - derivationally_related_forms
    - entailments
    - causes
    - also_sees
    - verb_groups
    - similar_tos
    - pertainyms
    """

    __slots__ = [
        "_wordnet_corpus_reader",
        "_name",
        "_syntactic_marker",
        "_synset",
        "_frame_strings",
        "_frame_ids",
        "_lexname_index",
        "_lex_id",
        "_lang",
        "_key",
    ]

    def __init__(
        self,
        wordnet_corpus_reader,
        synset,
        name,
        lexname_index,
        lex_id,
        syntactic_marker,
    ):
        self._wordnet_corpus_reader = wordnet_corpus_reader
        self._name = name
        self._syntactic_marker = syntactic_marker
        self._synset = synset
        self._frame_strings = []
        self._frame_ids = []
        self._lexname_index = lexname_index
        self._lex_id = lex_id
        self._lang = "eng"

        self._key = None  # gets set later.

    def name(self):
        return self._name

    def syntactic_marker(self):
        return self._syntactic_marker

    def synset(self):
        return self._synset

    def frame_strings(self):
        return self._frame_strings

    def frame_ids(self):
        return self._frame_ids

    def lang(self):
        return self._lang

    def key(self):
        return self._key

    def __repr__(self):
        tup = type(self).__name__, self._synset._name, self._name
        return "%s('%s.%s')" % tup

    def _related(self, relation_symbol):
        get_synset = self._wordnet_corpus_reader.synset_from_pos_and_offset
        if (self._name, relation_symbol) not in self._synset._lemma_pointers:
            return []
        return [
            get_synset(pos, offset)._lemmas[lemma_index]
            for pos, offset, lemma_index in self._synset._lemma_pointers[
                self._name, relation_symbol
            ]
        ]

    def count(self):
        """Return the frequency count for this Lemma"""
        return self._wordnet_corpus_reader.lemma_count(self)

    def antonyms(self):
        return self._related("!")

    def derivationally_related_forms(self):
        return self._related("+")

    def pertainyms(self):
        return self._related("\\")


class Synset(_WordNetObject):
    """Create a Synset from a "<lemma>.<pos>.<number>" string where:
    <lemma> is the word's morphological stem
    <pos> is one of the module attributes ADJ, ADJ_SAT, ADV, NOUN or VERB
    <number> is the sense number, counting from 0.

    Synset attributes, accessible via methods with the same name:

    - name: The canonical name of this synset, formed using the first lemma
      of this synset. Note that this may be different from the name
      passed to the constructor if that string used a different lemma to
      identify the synset.
    - pos: The synset's part of speech, matching one of the module level
      attributes ADJ, ADJ_SAT, ADV, NOUN or VERB.
    - lemmas: A list of the Lemma objects for this synset.
    - definition: The definition for this synset.
    - examples: A list of example strings for this synset.
    - offset: The offset in the WordNet dict file of this synset.
    - lexname: The name of the lexicographer file containing this synset.

    Synset methods:

    Synsets have the following methods for retrieving related Synsets.
    They correspond to the names for the pointer symbols defined here:
    https://wordnet.princeton.edu/documentation/wninput5wn
    These methods all return lists of Synsets.

    - hypernyms, instance_hypernyms
    - hyponyms, instance_hyponyms
    - member_holonyms, substance_holonyms, part_holonyms
    - member_meronyms, substance_meronyms, part_meronyms
    - attributes
    - entailments
    - causes
    - also_sees
    - verb_groups
    - similar_tos

    Additionally, Synsets support the following methods specific to the
    hypernym relation:

    - root_hypernyms
    - common_hypernyms
    - lowest_common_hypernyms

    Note that Synsets do not support the following relations because
    these are defined by WordNet as lexical relations:

    - antonyms
    - derivationally_related_forms
    - pertainyms
    """

    __slots__ = [
        "_pos",
        "_offset",
        "_name",
        "_frame_ids",
        "_lemmas",
        "_lemma_names",
        "_definition",
        "_examples",
        "_lexname",
        "_pointers",
        "_lemma_pointers",
        "_max_depth",
        "_min_depth",
    ]

    def __init__(self, wordnet_corpus_reader):
        self._wordnet_corpus_reader = wordnet_corpus_reader
        # All of these attributes get initialized by
        # WordNetCorpusReader._synset_from_pos_and_line()

        self._pos = None
        self._offset = None
        self._name = None
        self._frame_ids = []
        self._lemmas = []
        self._lemma_names = []
        self._definition = None
        self._examples = []
        self._lexname = None  # lexicographer name
        self._all_hypernyms = None

        self._pointers = defaultdict(set)
        self._lemma_pointers = defaultdict(list)

    def pos(self):
        return self._pos

    def offset(self):
        return self._offset

    def name(self):
        return self._name

    def frame_ids(self):
        return self._frame_ids

    def _doc(self, doc_type, default, lang="eng"):
        """Helper method for Synset.definition and Synset.examples"""
        corpus = self._wordnet_corpus_reader
        if lang not in corpus.langs():
            return None
        elif lang == "eng":
            return default
        else:
            corpus._load_lang_data(lang)
            of = corpus.ss2of(self)
            i = corpus.lg_attrs.index(doc_type)
            if of in corpus._lang_data[lang][i]:
                return corpus._lang_data[lang][i][of]
            else:
                return None

    def definition(self, lang="eng"):
        """Return definition in specified language"""
        return self._doc("def", self._definition, lang=lang)

    def examples(self, lang="eng"):
        """Return examples in specified language"""
        return self._doc("exe", self._examples, lang=lang)

    def lexname(self):
        return self._lexname

    def _needs_root(self):
        if self._pos == NOUN and self._wordnet_corpus_reader.get_version() != "1.6":
            return False
        else:
            return True

    def lemma_names(self, lang="eng"):
        """Return all the lemma_names associated with the synset"""
        if lang == "eng":
            return self._lemma_names
        else:
            reader = self._wordnet_corpus_reader
            reader._load_lang_data(lang)
            i = reader.ss2of(self)
            if i in reader._lang_data[lang][0]:
                return reader._lang_data[lang][0][i]
            else:
                return []

    def lemmas(self, lang="eng"):
        """Return all the lemma objects associated with the synset"""
        if lang == "eng":
            return self._lemmas
        elif self._name:
            self._wordnet_corpus_reader._load_lang_data(lang)
            lemmark = []
            lemmy = self.lemma_names(lang)
            for lem in lemmy:
                temp = Lemma(
                    self._wordnet_corpus_reader,
                    self,
                    lem,
                    self._wordnet_corpus_reader._lexnames.index(self.lexname()),
                    0,
                    None,
                )
                temp._lang = lang
                lemmark.append(temp)
            return lemmark

    def root_hypernyms(self):
        """Get the topmost hypernyms of this synset in WordNet."""

        result = []
        seen = set()
        todo = [self]
        while todo:
            next_synset = todo.pop()
            if next_synset not in seen:
                seen.add(next_synset)
                next_hypernyms = (
                    next_synset.hypernyms() + next_synset.instance_hypernyms()
                )
                if not next_hypernyms:
                    result.append(next_synset)
                else:
                    todo.extend(next_hypernyms)
        return result

    # Simpler implementation which makes incorrect assumption that
    # hypernym hierarchy is acyclic:
    #
    #        if not self.hypernyms():
    #            return [self]
    #        else:
    #            return list(set(root for h in self.hypernyms()
    #                            for root in h.root_hypernyms()))
    def max_depth(self):
        """
        :return: The length of the longest hypernym path from this
            synset to the root.
        """

        if "_max_depth" not in self.__dict__:
            hypernyms = self.hypernyms() + self.instance_hypernyms()
            if not hypernyms:
                self._max_depth = 0
            else:
                self._max_depth = 1 + max(h.max_depth() for h in hypernyms)
        return self._max_depth

    def min_depth(self):
        """
        :return: The length of the shortest hypernym path from this
            synset to the root.
        """

        if "_min_depth" not in self.__dict__:
            hypernyms = self.hypernyms() + self.instance_hypernyms()
            if not hypernyms:
                self._min_depth = 0
            else:
                self._min_depth = 1 + min(h.min_depth() for h in hypernyms)
        return self._min_depth

    def closure(self, rel, depth=-1):
        """
        Return the transitive closure of source under the rel
        relationship, breadth-first, discarding cycles:

        >>> from nltk.corpus import wordnet as wn
        >>> computer = wn.synset('computer.n.01')
        >>> topic = lambda s:s.topic_domains()
        >>> print(list(computer.closure(topic)))
        [Synset('computer_science.n.01')]

        UserWarning: Discarded redundant search for Synset('computer.n.01') at depth 2


        Include redundant paths (but only once), avoiding duplicate searches
        (from 'animal.n.01' to 'entity.n.01'):

        >>> dog = wn.synset('dog.n.01')
        >>> hyp = lambda s:s.hypernyms()
        >>> print(list(dog.closure(hyp)))
        [Synset('canine.n.02'), Synset('domestic_animal.n.01'), Synset('carnivore.n.01'),\
 Synset('animal.n.01'), Synset('placental.n.01'), Synset('organism.n.01'),\
 Synset('mammal.n.01'), Synset('living_thing.n.01'), Synset('vertebrate.n.01'),\
 Synset('whole.n.02'), Synset('chordate.n.01'), Synset('object.n.01'),\
 Synset('physical_entity.n.01'), Synset('entity.n.01')]

        UserWarning: Discarded redundant search for Synset('animal.n.01') at depth 7
        """

        from nltk.util import acyclic_breadth_first

        for synset in acyclic_breadth_first(self, rel, depth):
            if synset != self:
                yield synset

    from nltk.util import acyclic_depth_first as acyclic_tree
    from nltk.util import unweighted_minimum_spanning_tree as mst

    # Also add this shortcut?
    #    from nltk.util import unweighted_minimum_spanning_digraph as umsd

    def tree(self, rel, depth=-1, cut_mark=None):
        """
        Return the full relation tree, including self,
        discarding cycles:

        >>> from nltk.corpus import wordnet as wn
        >>> from pprint import pprint
        >>> computer = wn.synset('computer.n.01')
        >>> topic = lambda s:s.topic_domains()
        >>> pprint(computer.tree(topic))
        [Synset('computer.n.01'), [Synset('computer_science.n.01')]]

        UserWarning: Discarded redundant search for Synset('computer.n.01') at depth -3


        But keep duplicate branches (from 'animal.n.01' to 'entity.n.01'):

        >>> dog = wn.synset('dog.n.01')
        >>> hyp = lambda s:s.hypernyms()
        >>> pprint(dog.tree(hyp))
        [Synset('dog.n.01'),
         [Synset('canine.n.02'),
          [Synset('carnivore.n.01'),
           [Synset('placental.n.01'),
            [Synset('mammal.n.01'),
             [Synset('vertebrate.n.01'),
              [Synset('chordate.n.01'),
               [Synset('animal.n.01'),
                [Synset('organism.n.01'),
                 [Synset('living_thing.n.01'),
                  [Synset('whole.n.02'),
                   [Synset('object.n.01'),
                    [Synset('physical_entity.n.01'),
                     [Synset('entity.n.01')]]]]]]]]]]]]],
         [Synset('domestic_animal.n.01'),
          [Synset('animal.n.01'),
           [Synset('organism.n.01'),
            [Synset('living_thing.n.01'),
             [Synset('whole.n.02'),
              [Synset('object.n.01'),
               [Synset('physical_entity.n.01'), [Synset('entity.n.01')]]]]]]]]]
        """

        from nltk.util import acyclic_branches_depth_first

        return acyclic_branches_depth_first(self, rel, depth, cut_mark)

    def hypernym_paths(self):
        """
        Get the path(s) from this synset to the root, where each path is a
        list of the synset nodes traversed on the way to the root.

        :return: A list of lists, where each list gives the node sequence
           connecting the initial ``Synset`` node and a root node.
        """
        paths = []

        hypernyms = self.hypernyms() + self.instance_hypernyms()
        if len(hypernyms) == 0:
            paths = [[self]]

        for hypernym in hypernyms:
            for ancestor_list in hypernym.hypernym_paths():
                ancestor_list.append(self)
                paths.append(ancestor_list)
        return paths

    def common_hypernyms(self, other):
        """
        Find all synsets that are hypernyms of this synset and the
        other synset.

        :type other: Synset
        :param other: other input synset.
        :return: The synsets that are hypernyms of both synsets.
        """
        if not self._all_hypernyms:
            self._all_hypernyms = {
                self_synset
                for self_synsets in self._iter_hypernym_lists()
                for self_synset in self_synsets
            }
        if not other._all_hypernyms:
            other._all_hypernyms = {
                other_synset
                for other_synsets in other._iter_hypernym_lists()
                for other_synset in other_synsets
            }
        return list(self._all_hypernyms.intersection(other._all_hypernyms))

    def lowest_common_hypernyms(self, other, simulate_root=False, use_min_depth=False):
        """
        Get a list of lowest synset(s) that both synsets have as a hypernym.
        When `use_min_depth == False` this means that the synset which appears
        as a hypernym of both `self` and `other` with the lowest maximum depth
        is returned or if there are multiple such synsets at the same depth
        they are all returned

        However, if `use_min_depth == True` then the synset(s) which has/have
        the lowest minimum depth and appear(s) in both paths is/are returned.

        By setting the use_min_depth flag to True, the behavior of NLTK2 can be
        preserved. This was changed in NLTK3 to give more accurate results in a
        small set of cases, generally with synsets concerning people. (eg:
        'chef.n.01', 'fireman.n.01', etc.)

        This method is an implementation of Ted Pedersen's "Lowest Common
        Subsumer" method from the Perl Wordnet module. It can return either
        "self" or "other" if they are a hypernym of the other.

        :type other: Synset
        :param other: other input synset
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (False by default)
            creates a fake root that connects all the taxonomies. Set it
            to True to enable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will need to be added
            for nouns as well.
        :type use_min_depth: bool
        :param use_min_depth: This setting mimics older (v2) behavior of NLTK
            wordnet If True, will use the min_depth function to calculate the
            lowest common hypernyms. This is known to give strange results for
            some synset pairs (eg: 'chef.n.01', 'fireman.n.01') but is retained
            for backwards compatibility
        :return: The synsets that are the lowest common hypernyms of both
            synsets
        """
        synsets = self.common_hypernyms(other)
        if simulate_root:
            fake_synset = Synset(None)
            fake_synset._name = "*ROOT*"
            fake_synset.hypernyms = lambda: []
            fake_synset.instance_hypernyms = lambda: []
            synsets.append(fake_synset)

        try:
            if use_min_depth:
                max_depth = max(s.min_depth() for s in synsets)
                unsorted_lch = [s for s in synsets if s.min_depth() == max_depth]
            else:
                max_depth = max(s.max_depth() for s in synsets)
                unsorted_lch = [s for s in synsets if s.max_depth() == max_depth]
            return sorted(unsorted_lch)
        except ValueError:
            return []

    def hypernym_distances(self, distance=0, simulate_root=False):
        """
        Get the path(s) from this synset to the root, counting the distance
        of each node from the initial node on the way. A set of
        (synset, distance) tuples is returned.

        :type distance: int
        :param distance: the distance (number of edges) from this hypernym to
            the original hypernym ``Synset`` on which this method was called.
        :return: A set of ``(Synset, int)`` tuples where each ``Synset`` is
           a hypernym of the first ``Synset``.
        """
        distances = {(self, distance)}
        for hypernym in self._hypernyms() + self._instance_hypernyms():
            distances |= hypernym.hypernym_distances(distance + 1, simulate_root=False)
        if simulate_root:
            fake_synset = Synset(None)
            fake_synset._name = "*ROOT*"
            fake_synset_distance = max(distances, key=itemgetter(1))[1]
            distances.add((fake_synset, fake_synset_distance + 1))
        return distances

    def _shortest_hypernym_paths(self, simulate_root):
        if self._name == "*ROOT*":
            return {self: 0}

        queue = deque([(self, 0)])
        path = {}

        while queue:
            s, depth = queue.popleft()
            if s in path:
                continue
            path[s] = depth

            depth += 1
            queue.extend((hyp, depth) for hyp in s._hypernyms())
            queue.extend((hyp, depth) for hyp in s._instance_hypernyms())

        if simulate_root:
            fake_synset = Synset(None)
            fake_synset._name = "*ROOT*"
            path[fake_synset] = max(path.values()) + 1

        return path

    def shortest_path_distance(self, other, simulate_root=False):
        """
        Returns the distance of the shortest path linking the two synsets (if
        one exists). For each synset, all the ancestor nodes and their
        distances are recorded and compared. The ancestor node common to both
        synsets that can be reached with the minimum number of traversals is
        used. If no ancestor nodes are common, None is returned. If a node is
        compared with itself 0 is returned.

        :type other: Synset
        :param other: The Synset to which the shortest path will be found.
        :return: The number of edges in the shortest path connecting the two
            nodes, or None if no path exists.
        """

        if self == other:
            return 0

        dist_dict1 = self._shortest_hypernym_paths(simulate_root)
        dist_dict2 = other._shortest_hypernym_paths(simulate_root)

        # For each ancestor synset common to both subject synsets, find the
        # connecting path length. Return the shortest of these.

        inf = float("inf")
        path_distance = inf
        for synset, d1 in dist_dict1.items():
            d2 = dist_dict2.get(synset, inf)
            path_distance = min(path_distance, d1 + d2)

        return None if math.isinf(path_distance) else path_distance

    # interface to similarity methods
    def path_similarity(self, other, verbose=False, simulate_root=True):
        """
        Path Distance Similarity:
        Return a score denoting how similar two word senses are, based on the
        shortest path that connects the senses in the is-a (hypernym/hypnoym)
        taxonomy. The score is in the range 0 to 1, except in those cases where
        a path cannot be found (will only be true for verbs as there are many
        distinct verb taxonomies), in which case None is returned. A score of
        1 represents identity i.e. comparing a sense with itself will return 1.

        :type other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (True by default)
            creates a fake root that connects all the taxonomies. Set it
            to false to disable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will be added for nouns
            as well.
        :return: A score denoting the similarity of the two ``Synset`` objects,
            normally between 0 and 1. None is returned if no connecting path
            could be found. 1 is returned if a ``Synset`` is compared with
            itself.
        """

        distance = self.shortest_path_distance(
            other,
            simulate_root=simulate_root and (self._needs_root() or other._needs_root()),
        )
        if distance is None or distance < 0:
            return None
        return 1.0 / (distance + 1)

    def lch_similarity(self, other, verbose=False, simulate_root=True):
        """
        Leacock Chodorow Similarity:
        Return a score denoting how similar two word senses are, based on the
        shortest path that connects the senses (as above) and the maximum depth
        of the taxonomy in which the senses occur. The relationship is given as
        -log(p/2d) where p is the shortest path length and d is the taxonomy
        depth.

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (True by default)
            creates a fake root that connects all the taxonomies. Set it
            to false to disable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will be added for nouns
            as well.
        :return: A score denoting the similarity of the two ``Synset`` objects,
            normally greater than 0. None is returned if no connecting path
            could be found. If a ``Synset`` is compared with itself, the
            maximum score is returned, which varies depending on the taxonomy
            depth.
        """

        if self._pos != other._pos:
            raise WordNetError(
                "Computing the lch similarity requires "
                "%s and %s to have the same part of speech." % (self, other)
            )

        need_root = self._needs_root()

        if self._pos not in self._wordnet_corpus_reader._max_depth:
            self._wordnet_corpus_reader._compute_max_depth(self._pos, need_root)

        depth = self._wordnet_corpus_reader._max_depth[self._pos]

        distance = self.shortest_path_distance(
            other, simulate_root=simulate_root and need_root
        )

        if distance is None or distance < 0 or depth == 0:
            return None
        return -math.log((distance + 1) / (2.0 * depth))

    def wup_similarity(self, other, verbose=False, simulate_root=True):
        """
        Wu-Palmer Similarity:
        Return a score denoting how similar two word senses are, based on the
        depth of the two senses in the taxonomy and that of their Least Common
        Subsumer (most specific ancestor node). Previously, the scores computed
        by this implementation did _not_ always agree with those given by
        Pedersen's Perl implementation of WordNet Similarity. However, with
        the addition of the simulate_root flag (see below), the score for
        verbs now almost always agree but not always for nouns.

        The LCS does not necessarily feature in the shortest path connecting
        the two senses, as it is by definition the common ancestor deepest in
        the taxonomy, not closest to the two senses. Typically, however, it
        will so feature. Where multiple candidates for the LCS exist, that
        whose shortest path to the root node is the longest will be selected.
        Where the LCS has multiple paths to the root, the longer path is used
        for the purposes of the calculation.

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (True by default)
            creates a fake root that connects all the taxonomies. Set it
            to false to disable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will be added for nouns
            as well.
        :return: A float score denoting the similarity of the two ``Synset``
            objects, normally greater than zero. If no connecting path between
            the two senses can be found, None is returned.

        """
        need_root = self._needs_root() or other._needs_root()

        # Note that to preserve behavior from NLTK2 we set use_min_depth=True
        # It is possible that more accurate results could be obtained by
        # removing this setting and it should be tested later on
        subsumers = self.lowest_common_hypernyms(
            other, simulate_root=simulate_root and need_root, use_min_depth=True
        )

        # If no LCS was found return None
        if len(subsumers) == 0:
            return None

        subsumer = self if self in subsumers else subsumers[0]

        # Get the longest path from the LCS to the root,
        # including a correction:
        # - add one because the calculations include both the start and end
        #   nodes
        depth = subsumer.max_depth() + 1

        # Note: No need for an additional add-one correction for non-nouns
        # to account for an imaginary root node because that is now
        # automatically handled by simulate_root
        # if subsumer._pos != NOUN:
        #     depth += 1

        # Get the shortest path from the LCS to each of the synsets it is
        # subsuming.  Add this to the LCS path length to get the path
        # length from each synset to the root.
        len1 = self.shortest_path_distance(
            subsumer, simulate_root=simulate_root and need_root
        )
        len2 = other.shortest_path_distance(
            subsumer, simulate_root=simulate_root and need_root
        )
        if len1 is None or len2 is None:
            return None
        len1 += depth
        len2 += depth
        return (2.0 * depth) / (len1 + len2)

    def res_similarity(self, other, ic, verbose=False):
        """
        Resnik Similarity:
        Return a score denoting how similar two word senses are, based on the
        Information Content (IC) of the Least Common Subsumer (most specific
        ancestor node).

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type ic: dict
        :param ic: an information content object (as returned by
            ``nltk.corpus.wordnet_ic.ic()``).
        :return: A float score denoting the similarity of the two ``Synset``
            objects. Synsets whose LCS is the root node of the taxonomy will
            have a score of 0 (e.g. N['dog'][0] and N['table'][0]).
        """

        ic1, ic2, lcs_ic = _lcs_ic(self, other, ic)
        return lcs_ic

    def jcn_similarity(self, other, ic, verbose=False):
        """
        Jiang-Conrath Similarity:
        Return a score denoting how similar two word senses are, based on the
        Information Content (IC) of the Least Common Subsumer (most specific
        ancestor node) and that of the two input Synsets. The relationship is
        given by the equation 1 / (IC(s1) + IC(s2) - 2 * IC(lcs)).

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type  ic: dict
        :param ic: an information content object (as returned by
            ``nltk.corpus.wordnet_ic.ic()``).
        :return: A float score denoting the similarity of the two ``Synset``
            objects.
        """

        if self == other:
            return _INF

        ic1, ic2, lcs_ic = _lcs_ic(self, other, ic)

        # If either of the input synsets are the root synset, or have a
        # frequency of 0 (sparse data problem), return 0.
        if ic1 == 0 or ic2 == 0:
            return 0

        ic_difference = ic1 + ic2 - 2 * lcs_ic

        if ic_difference == 0:
            return _INF

        return 1 / ic_difference

    def lin_similarity(self, other, ic, verbose=False):
        """
        Lin Similarity:
        Return a score denoting how similar two word senses are, based on the
        Information Content (IC) of the Least Common Subsumer (most specific
        ancestor node) and that of the two input Synsets. The relationship is
        given by the equation 2 * IC(lcs) / (IC(s1) + IC(s2)).

        :type other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type ic: dict
        :param ic: an information content object (as returned by
            ``nltk.corpus.wordnet_ic.ic()``).
        :return: A float score denoting the similarity of the two ``Synset``
            objects, in the range 0 to 1.
        """

        ic1, ic2, lcs_ic = _lcs_ic(self, other, ic)
        return (2.0 * lcs_ic) / (ic1 + ic2)

    def _iter_hypernym_lists(self):
        """
        :return: An iterator over ``Synset`` objects that are either proper
        hypernyms or instance of hypernyms of the synset.
        """
        todo = [self]
        seen = set()
        while todo:
            for synset in todo:
                seen.add(synset)
            yield todo
            todo = [
                hypernym
                for synset in todo
                for hypernym in (synset.hypernyms() + synset.instance_hypernyms())
                if hypernym not in seen
            ]

    def __repr__(self):
        return f"{type(self).__name__}('{self._name}')"

    def _related(self, relation_symbol, sort=True):
        get_synset = self._wordnet_corpus_reader.synset_from_pos_and_offset
        if relation_symbol not in self._pointers:
            return []
        pointer_tuples = self._pointers[relation_symbol]
        r = [get_synset(pos, offset) for pos, offset in pointer_tuples]
        if sort:
            r.sort()
        return r


######################################################################
# WordNet Corpus Reader
######################################################################


class WordNetCorpusReader(CorpusReader):
    """
    A corpus reader used to access wordnet or its variants.
    """

    _ENCODING = "utf8"

    # { Part-of-speech constants
    ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
    # }

    # { Filename constants
    _FILEMAP = {ADJ: "adj", ADV: "adv", NOUN: "noun", VERB: "verb"}
    # }

    # { Part of speech constants
    _pos_numbers = {NOUN: 1, VERB: 2, ADJ: 3, ADV: 4, ADJ_SAT: 5}
    _pos_names = dict(tup[::-1] for tup in _pos_numbers.items())
    # }

    #: A list of file identifiers for all the fileids used by this
    #: corpus reader.
    _FILES = (
        "cntlist.rev",
        "lexnames",
        "index.sense",
        "index.adj",
        "index.adv",
        "index.noun",
        "index.verb",
        "data.adj",
        "data.adv",
        "data.noun",
        "data.verb",
        "adj.exc",
        "adv.exc",
        "noun.exc",
        "verb.exc",
    )

    def __init__(self, root, omw_reader):
        """
        Construct a new wordnet corpus reader, with the given root
        directory.
        """

        super().__init__(root, self._FILES, encoding=self._ENCODING)

        # A index that provides the file offset
        # Map from lemma -> pos -> synset_index -> offset
        self._lemma_pos_offset_map = defaultdict(dict)

        # A cache so we don't have to reconstruct synsets
        # Map from pos -> offset -> synset
        self._synset_offset_cache = defaultdict(dict)

        # A lookup for the maximum depth of each part of speech.  Useful for
        # the lch similarity metric.
        self._max_depth = defaultdict(dict)

        # Corpus reader containing omw data.
        self._omw_reader = omw_reader

        # Corpus reader containing extended_omw data.
        self._exomw_reader = None

        self.provenances = defaultdict(str)
        self.provenances["eng"] = ""

        if self._omw_reader is None:
            warnings.warn(
                "The multilingual functions are not available with this Wordnet version"
            )

        self.omw_langs = set()

        # A cache to store the wordnet data of multiple languages
        self._lang_data = defaultdict(list)

        self._data_file_map = {}
        self._exception_map = {}
        self._lexnames = []
        self._key_count_file = None
        self._key_synset_file = None

        # Load the lexnames
        with self.open("lexnames") as fp:
            for i, line in enumerate(fp):
                index, lexname, _ = line.split()
                assert int(index) == i
                self._lexnames.append(lexname)

        # Load the indices for lemmas and synset offsets
        self._load_lemma_pos_offset_map()

        # load the exception file data into memory
        self._load_exception_map()

        self.nomap = []
        self.splits = {}

        # map from WordNet 3.0 for OMW data
        self.map30 = self.map_wn30()

        # Language data attributes
        self.lg_attrs = ["lemma", "none", "def", "exe"]

    def index_sense(self, version=None):
        """Read sense key to synset id mapping from index.sense file in corpus directory"""
        fn = "index.sense"
        if version:
            from nltk.corpus import CorpusReader, LazyCorpusLoader

            ixreader = LazyCorpusLoader(version, CorpusReader, r".*/" + fn)
        else:
            ixreader = self
        with ixreader.open(fn) as fp:
            sensekey_map = {}
            for line in fp:
                fields = line.strip().split()
                sensekey = fields[0]
                pos = self._pos_names[int(sensekey.split("%")[1].split(":")[0])]
                sensekey_map[sensekey] = f"{fields[1]}-{pos}"
        return sensekey_map

    def map_to_many(self):
        sensekey_map1 = self.index_sense("wordnet")
        sensekey_map2 = self.index_sense()
        synset_to_many = {}
        for synsetid in set(sensekey_map1.values()):
            synset_to_many[synsetid] = []
        for sensekey in set(sensekey_map1.keys()).intersection(
            set(sensekey_map2.keys())
        ):
            source = sensekey_map1[sensekey]
            target = sensekey_map2[sensekey]
            synset_to_many[source].append(target)
        return synset_to_many

    def map_to_one(self):
        synset_to_many = self.map_to_many()
        synset_to_one = {}
        for source in synset_to_many:
            candidates_bag = synset_to_many[source]
            if candidates_bag:
                candidates_set = set(candidates_bag)
                if len(candidates_set) == 1:
                    target = candidates_bag[0]
                else:
                    counts = []
                    for candidate in candidates_set:
                        counts.append((candidates_bag.count(candidate), candidate))
                    self.splits[source] = counts
                    target = max(counts)[1]
                synset_to_one[source] = target
                if source[-1] == "s":
                    # Add a mapping from "a" to target for applications like omw,
                    # where only Lithuanian and Slovak use the "s" ss_type.
                    synset_to_one[f"{source[:-1]}a"] = target
            else:
                self.nomap.append(source)
        return synset_to_one

    def map_wn30(self):
        """Mapping from Wordnet 3.0 to currently loaded Wordnet version"""
        if self.get_version() == "3.0":
            return None
        else:
            return self.map_to_one()

    # Open Multilingual WordNet functions, contributed by
    # Nasruddin A’aidil Shari, Sim Wei Ying Geraldine, and Soe Lynn

    def of2ss(self, of):
        """take an id and return the synsets"""
        return self.synset_from_pos_and_offset(of[-1], int(of[:8]))

    def ss2of(self, ss):
        """return the ID of the synset"""
        if ss:
            return f"{ss.offset():08d}-{ss.pos()}"

    def _load_lang_data(self, lang):
        """load the wordnet data of the requested language from the file to
        the cache, _lang_data"""

        if lang in self._lang_data:
            return

        if self._omw_reader and not self.omw_langs:
            self.add_omw()

        if lang not in self.langs():
            raise WordNetError("Language is not supported.")

        if self._exomw_reader and lang not in self.omw_langs:
            reader = self._exomw_reader
        else:
            reader = self._omw_reader

        prov = self.provenances[lang]
        if prov in ["cldr", "wikt"]:
            prov2 = prov
        else:
            prov2 = "data"

        with reader.open(f"{prov}/wn-{prov2}-{lang.split('_')[0]}.tab") as fp:
            self.custom_lemmas(fp, lang)
        self.disable_custom_lemmas(lang)

    def add_provs(self, reader):
        """Add languages from Multilingual Wordnet to the provenance dictionary"""
        fileids = reader.fileids()
        for fileid in fileids:
            prov, langfile = os.path.split(fileid)
            file_name, file_extension = os.path.splitext(langfile)
            if file_extension == ".tab":
                lang = file_name.split("-")[-1]
                if lang in self.provenances or prov in ["cldr", "wikt"]:
                    # We already have another resource for this lang,
                    # so we need to further specify the lang id:
                    lang = f"{lang}_{prov}"
                self.provenances[lang] = prov

    def add_omw(self):
        self.add_provs(self._omw_reader)
        self.omw_langs = set(self.provenances.keys())

    def add_exomw(self):
        """
        Add languages from Extended OMW

        >>> import nltk
        >>> from nltk.corpus import wordnet as wn
        >>> wn.add_exomw()
        >>> print(wn.synset('intrinsically.r.01').lemmas(lang="eng_wikt"))
        [Lemma('intrinsically.r.01.per_se'), Lemma('intrinsically.r.01.as_such')]
        """
        from nltk.corpus import extended_omw

        self.add_omw()
        self._exomw_reader = extended_omw
        self.add_provs(self._exomw_reader)

    def langs(self):
        """return a list of languages supported by Multilingual Wordnet"""
        return list(self.provenances.keys())

    def _load_lemma_pos_offset_map(self):
        for suffix in self._FILEMAP.values():

            # parse each line of the file (ignoring comment lines)
            with self.open("index.%s" % suffix) as fp:
                for i, line in enumerate(fp):
                    if line.startswith(" "):
                        continue

                    _iter = iter(line.split())

                    def _next_token():
                        return next(_iter)

                    try:

                        # get the lemma and part-of-speech
                        lemma = _next_token()
                        pos = _next_token()

                        # get the number of synsets for this lemma
                        n_synsets = int(_next_token())
                        assert n_synsets > 0

                        # get and ignore the pointer symbols for all synsets of
                        # this lemma
                        n_pointers = int(_next_token())
                        [_next_token() for _ in range(n_pointers)]

                        # same as number of synsets
                        n_senses = int(_next_token())
                        assert n_synsets == n_senses

                        # get and ignore number of senses ranked according to
                        # frequency
                        _next_token()

                        # get synset offsets
                        synset_offsets = [int(_next_token()) for _ in range(n_synsets)]

                    # raise more informative error with file name and line number
                    except (AssertionError, ValueError) as e:
                        tup = ("index.%s" % suffix), (i + 1), e
                        raise WordNetError("file %s, line %i: %s" % tup) from e

                    # map lemmas and parts of speech to synsets
                    self._lemma_pos_offset_map[lemma][pos] = synset_offsets
                    if pos == ADJ:
                        self._lemma_pos_offset_map[lemma][ADJ_SAT] = synset_offsets

    def _load_exception_map(self):
        # load the exception file data into memory
        for pos, suffix in self._FILEMAP.items():
            self._exception_map[pos] = {}
            with self.open("%s.exc" % suffix) as fp:
                for line in fp:
                    terms = line.split()
                    self._exception_map[pos][terms[0]] = terms[1:]
        self._exception_map[ADJ_SAT] = self._exception_map[ADJ]

    def _compute_max_depth(self, pos, simulate_root):
        """
        Compute the max depth for the given part of speech.  This is
        used by the lch similarity metric.
        """
        depth = 0
        for ii in self.all_synsets(pos):
            try:
                depth = max(depth, ii.max_depth())
            except RuntimeError:
                print(ii)
        if simulate_root:
            depth += 1
        self._max_depth[pos] = depth

    def get_version(self):
        fh = self._data_file(ADJ)
        fh.seek(0)
        for line in fh:
            match = re.search(r"Word[nN]et (\d+|\d+\.\d+) Copyright", line)
            if match is not None:
                version = match.group(1)
                fh.seek(0)
                return version

    #############################################################
    # Loading Lemmas
    #############################################################

    def lemma(self, name, lang="eng"):
        """Return lemma object that matches the name"""
        # cannot simply split on first '.',
        # e.g.: '.45_caliber.a.01..45_caliber'
        separator = SENSENUM_RE.search(name).end()

        synset_name, lemma_name = name[: separator - 1], name[separator:]

        synset = self.synset(synset_name)
        for lemma in synset.lemmas(lang):
            if lemma._name == lemma_name:
                return lemma
        raise WordNetError(f"No lemma {lemma_name!r} in {synset_name!r}")

    def lemma_from_key(self, key):
        # Keys are case sensitive and always lower-case
        key = key.lower()

        lemma_name, lex_sense = key.split("%")
        pos_number, lexname_index, lex_id, _, _ = lex_sense.split(":")
        pos = self._pos_names[int(pos_number)]

        # open the key -> synset file if necessary
        if self._key_synset_file is None:
            self._key_synset_file = self.open("index.sense")

        # Find the synset for the lemma.
        synset_line = _binary_search_file(self._key_synset_file, key)
        if not synset_line:
            raise WordNetError("No synset found for key %r" % key)
        offset = int(synset_line.split()[1])
        synset = self.synset_from_pos_and_offset(pos, offset)
        # return the corresponding lemma
        for lemma in synset._lemmas:
            if lemma._key == key:
                return lemma
        raise WordNetError("No lemma found for for key %r" % key)

    #############################################################
    # Loading Synsets
    #############################################################
    def synset(self, name):
        # split name into lemma, part of speech and synset number
        lemma, pos, synset_index_str = name.lower().rsplit(".", 2)
        synset_index = int(synset_index_str) - 1

        # get the offset for this synset
        try:
            offset = self._lemma_pos_offset_map[lemma][pos][synset_index]
        except KeyError as e:
            raise WordNetError(f"No lemma {lemma!r} with part of speech {pos!r}") from e
        except IndexError as e:
            n_senses = len(self._lemma_pos_offset_map[lemma][pos])
            raise WordNetError(
                f"Lemma {lemma!r} with part of speech {pos!r} only "
                f"has {n_senses} {'sense' if n_senses == 1 else 'senses'}"
            ) from e

        # load synset information from the appropriate file
        synset = self.synset_from_pos_and_offset(pos, offset)

        # some basic sanity checks on loaded attributes
        if pos == "s" and synset._pos == "a":
            message = (
                "Adjective satellite requested but only plain "
                "adjective found for lemma %r"
            )
            raise WordNetError(message % lemma)
        assert synset._pos == pos or (pos == "a" and synset._pos == "s")

        # Return the synset object.
        return synset

    def _data_file(self, pos):
        """
        Return an open file pointer for the data file for the given
        part of speech.
        """
        if pos == ADJ_SAT:
            pos = ADJ
        if self._data_file_map.get(pos) is None:
            fileid = "data.%s" % self._FILEMAP[pos]
            self._data_file_map[pos] = self.open(fileid)
        return self._data_file_map[pos]

    def synset_from_pos_and_offset(self, pos, offset):
        """
        - pos: The synset's part of speech, matching one of the module level
          attributes ADJ, ADJ_SAT, ADV, NOUN or VERB ('a', 's', 'r', 'n', or 'v').
        - offset: The byte offset of this synset in the WordNet dict file
          for this pos.

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.synset_from_pos_and_offset('n', 1740))
        Synset('entity.n.01')
        """
        # Check to see if the synset is in the cache
        if offset in self._synset_offset_cache[pos]:
            return self._synset_offset_cache[pos][offset]

        data_file = self._data_file(pos)
        data_file.seek(offset)
        data_file_line = data_file.readline()
        # If valid, the offset equals the 8-digit 0-padded integer found at the start of the line:
        line_offset = data_file_line[:8]
        if (
            line_offset.isalnum()
            and line_offset == f"{'0'*(8-len(str(offset)))}{str(offset)}"
        ):
            synset = self._synset_from_pos_and_line(pos, data_file_line)
            assert synset._offset == offset
            self._synset_offset_cache[pos][offset] = synset
        else:
            synset = None
            warnings.warn(f"No WordNet synset found for pos={pos} at offset={offset}.")
        data_file.seek(0)
        return synset

    @deprecated("Use public method synset_from_pos_and_offset() instead")
    def _synset_from_pos_and_offset(self, *args, **kwargs):
        """
        Hack to help people like the readers of
        https://stackoverflow.com/a/27145655/1709587
        who were using this function before it was officially a public method
        """
        return self.synset_from_pos_and_offset(*args, **kwargs)

    def _synset_from_pos_and_line(self, pos, data_file_line):
        # Construct a new (empty) synset.
        synset = Synset(self)

        # parse the entry for this synset
        try:

            # parse out the definitions and examples from the gloss
            columns_str, gloss = data_file_line.strip().split("|")
            definition = re.sub(r"[\"].*?[\"]", "", gloss).strip()
            examples = re.findall(r'"([^"]*)"', gloss)
            for example in examples:
                synset._examples.append(example)

            synset._definition = definition.strip("; ")

            # split the other info into fields
            _iter = iter(columns_str.split())

            def _next_token():
                return next(_iter)

            # get the offset
            synset._offset = int(_next_token())

            # determine the lexicographer file name
            lexname_index = int(_next_token())
            synset._lexname = self._lexnames[lexname_index]

            # get the part of speech
            synset._pos = _next_token()

            # create Lemma objects for each lemma
            n_lemmas = int(_next_token(), 16)
            for _ in range(n_lemmas):
                # get the lemma name
                lemma_name = _next_token()
                # get the lex_id (used for sense_keys)
                lex_id = int(_next_token(), 16)
                # If the lemma has a syntactic marker, extract it.
                m = re.match(r"(.*?)(\(.*\))?$", lemma_name)
                lemma_name, syn_mark = m.groups()
                # create the lemma object
                lemma = Lemma(self, synset, lemma_name, lexname_index, lex_id, syn_mark)
                synset._lemmas.append(lemma)
                synset._lemma_names.append(lemma._name)

            # collect the pointer tuples
            n_pointers = int(_next_token())
            for _ in range(n_pointers):
                symbol = _next_token()
                offset = int(_next_token())
                pos = _next_token()
                lemma_ids_str = _next_token()
                if lemma_ids_str == "0000":
                    synset._pointers[symbol].add((pos, offset))
                else:
                    source_index = int(lemma_ids_str[:2], 16) - 1
                    target_index = int(lemma_ids_str[2:], 16) - 1
                    source_lemma_name = synset._lemmas[source_index]._name
                    lemma_pointers = synset._lemma_pointers
                    tups = lemma_pointers[source_lemma_name, symbol]
                    tups.append((pos, offset, target_index))

            # read the verb frames
            try:
                frame_count = int(_next_token())
            except StopIteration:
                pass
            else:
                for _ in range(frame_count):
                    # read the plus sign
                    plus = _next_token()
                    assert plus == "+"
                    # read the frame and lemma number
                    frame_number = int(_next_token())
                    frame_string_fmt = VERB_FRAME_STRINGS[frame_number]
                    lemma_number = int(_next_token(), 16)
                    # lemma number of 00 means all words in the synset
                    if lemma_number == 0:
                        synset._frame_ids.append(frame_number)
                        for lemma in synset._lemmas:
                            lemma._frame_ids.append(frame_number)
                            lemma._frame_strings.append(frame_string_fmt % lemma._name)
                    # only a specific word in the synset
                    else:
                        lemma = synset._lemmas[lemma_number - 1]
                        lemma._frame_ids.append(frame_number)
                        lemma._frame_strings.append(frame_string_fmt % lemma._name)

        # raise a more informative error with line text
        except ValueError as e:
            raise WordNetError(f"line {data_file_line!r}: {e}") from e

        # set sense keys for Lemma objects - note that this has to be
        # done afterwards so that the relations are available
        for lemma in synset._lemmas:
            if synset._pos == ADJ_SAT:
                head_lemma = synset.similar_tos()[0]._lemmas[0]
                head_name = head_lemma._name
                head_id = "%02d" % head_lemma._lex_id
            else:
                head_name = head_id = ""
            tup = (
                lemma._name,
                WordNetCorpusReader._pos_numbers[synset._pos],
                lemma._lexname_index,
                lemma._lex_id,
                head_name,
                head_id,
            )
            lemma._key = ("%s%%%d:%02d:%02d:%s:%s" % tup).lower()

        # the canonical name is based on the first lemma
        lemma_name = synset._lemmas[0]._name.lower()
        offsets = self._lemma_pos_offset_map[lemma_name][synset._pos]
        sense_index = offsets.index(synset._offset)
        tup = lemma_name, synset._pos, sense_index + 1
        synset._name = "%s.%s.%02i" % tup

        return synset

    def synset_from_sense_key(self, sense_key):
        """
        Retrieves synset based on a given sense_key. Sense keys can be
        obtained from lemma.key()

        From https://wordnet.princeton.edu/documentation/senseidx5wn:
        A sense_key is represented as::

            lemma % lex_sense (e.g. 'dog%1:18:01::')

        where lex_sense is encoded as::

            ss_type:lex_filenum:lex_id:head_word:head_id

        :lemma:       ASCII text of word/collocation, in lower case
        :ss_type:     synset type for the sense (1 digit int)
                      The synset type is encoded as follows::

                          1    NOUN
                          2    VERB
                          3    ADJECTIVE
                          4    ADVERB
                          5    ADJECTIVE SATELLITE
        :lex_filenum: name of lexicographer file containing the synset for the sense (2 digit int)
        :lex_id:      when paired with lemma, uniquely identifies a sense in the lexicographer file (2 digit int)
        :head_word:   lemma of the first word in satellite's head synset
                      Only used if sense is in an adjective satellite synset
        :head_id:     uniquely identifies sense in a lexicographer file when paired with head_word
                      Only used if head_word is present (2 digit int)

        >>> import nltk
        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.synset_from_sense_key("drive%1:04:03::"))
        Synset('drive.n.06')

        >>> print(wn.synset_from_sense_key("driving%1:04:03::"))
        Synset('drive.n.06')
        """
        return self.lemma_from_key(sense_key).synset()

    #############################################################
    # Retrieve synsets and lemmas.
    #############################################################

    def synsets(self, lemma, pos=None, lang="eng", check_exceptions=True):
        """Load all synsets with a given lemma and part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        If lang is specified, all the synsets associated with the lemma name
        of that language will be returned.
        """
        lemma = lemma.lower()

        if lang == "eng":
            get_synset = self.synset_from_pos_and_offset
            index = self._lemma_pos_offset_map
            if pos is None:
                pos = POS_LIST
            return [
                get_synset(p, offset)
                for p in pos
                for form in self._morphy(lemma, p, check_exceptions)
                for offset in index[form].get(p, [])
            ]

        else:
            self._load_lang_data(lang)
            synset_list = []
            if lemma in self._lang_data[lang][1]:
                for l in self._lang_data[lang][1][lemma]:
                    if pos is not None and l[-1] != pos:
                        continue
                    synset_list.append(self.of2ss(l))
            return synset_list

    def lemmas(self, lemma, pos=None, lang="eng"):
        """Return all Lemma objects with a name matching the specified lemma
        name and part of speech tag. Matches any part of speech tag if none is
        specified."""

        lemma = lemma.lower()
        if lang == "eng":
            return [
                lemma_obj
                for synset in self.synsets(lemma, pos)
                for lemma_obj in synset.lemmas()
                if lemma_obj.name().lower() == lemma
            ]

        else:
            self._load_lang_data(lang)
            lemmas = []
            syn = self.synsets(lemma, lang=lang)
            for s in syn:
                if pos is not None and s.pos() != pos:
                    continue
                for lemma_obj in s.lemmas(lang=lang):
                    if lemma_obj.name().lower() == lemma:
                        lemmas.append(lemma_obj)
            return lemmas

    def all_lemma_names(self, pos=None, lang="eng"):
        """Return all lemma names for all synsets for the given
        part of speech tag and language or languages. If pos is
        not specified, all synsets for all parts of speech will
        be used."""

        if lang == "eng":
            if pos is None:
                return iter(self._lemma_pos_offset_map)
            else:
                return (
                    lemma
                    for lemma in self._lemma_pos_offset_map
                    if pos in self._lemma_pos_offset_map[lemma]
                )
        else:
            self._load_lang_data(lang)
            lemma = []
            for i in self._lang_data[lang][0]:
                if pos is not None and i[-1] != pos:
                    continue
                lemma.extend(self._lang_data[lang][0][i])

            lemma = iter(set(lemma))
            return lemma

    def all_omw_synsets(self, pos=None, lang=None):
        if lang not in self.langs():
            return None
        self._load_lang_data(lang)
        for of in self._lang_data[lang][0]:
            if not pos or of[-1] == pos:
                ss = self.of2ss(of)
                if ss:
                    yield ss

    #            else:
    # A few OMW offsets don't exist in Wordnet 3.0.
    #                warnings.warn(f"Language {lang}: no synset found for {of}")

    def all_synsets(self, pos=None, lang="eng"):
        """Iterate over all synsets with a given part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        """
        if lang == "eng":
            return self.all_eng_synsets(pos=pos)
        else:
            return self.all_omw_synsets(pos=pos, lang=lang)

    def all_eng_synsets(self, pos=None):
        if pos is None:
            pos_tags = self._FILEMAP.keys()
        else:
            pos_tags = [pos]

        cache = self._synset_offset_cache
        from_pos_and_line = self._synset_from_pos_and_line

        # generate all synsets for each part of speech
        for pos_tag in pos_tags:
            # Open the file for reading.  Note that we can not re-use
            # the file pointers from self._data_file_map here, because
            # we're defining an iterator, and those file pointers might
            # be moved while we're not looking.
            if pos_tag == ADJ_SAT:
                pos_file = ADJ
            else:
                pos_file = pos_tag
            fileid = "data.%s" % self._FILEMAP[pos_file]
            data_file = self.open(fileid)

            try:
                # generate synsets for each line in the POS file
                offset = data_file.tell()
                line = data_file.readline()
                while line:
                    if not line[0].isspace():
                        if offset in cache[pos_tag]:
                            # See if the synset is cached
                            synset = cache[pos_tag][offset]
                        else:
                            # Otherwise, parse the line
                            synset = from_pos_and_line(pos_tag, line)
                            cache[pos_tag][offset] = synset

                        # adjective satellites are in the same file as
                        # adjectives so only yield the synset if it's actually
                        # a satellite
                        if pos_tag == ADJ_SAT and synset._pos == ADJ_SAT:
                            yield synset
                        # for all other POS tags, yield all synsets (this means
                        # that adjectives also include adjective satellites)
                        elif pos_tag != ADJ_SAT:
                            yield synset
                    offset = data_file.tell()
                    line = data_file.readline()

            # close the extra file handle we opened
            except:
                data_file.close()
                raise
            else:
                data_file.close()

    def words(self, lang="eng"):
        """return lemmas of the given language as list of words"""
        return self.all_lemma_names(lang=lang)

    def synonyms(self, word, lang="eng"):
        """return nested list with the synonyms of the different senses of word in the given language"""
        return [
            sorted(list(set(ss.lemma_names(lang=lang)) - {word}))
            for ss in self.synsets(word, lang=lang)
        ]

    def doc(self, file="README", lang="eng"):
        """Return the contents of readme, license or citation file
        use lang=lang to get the file for an individual language"""
        if lang == "eng":
            reader = self
        else:
            reader = self._omw_reader
            if lang in self.langs():
                file = f"{os.path.join(self.provenances[lang],file)}"
        try:
            with reader.open(file) as fp:
                return fp.read()
        except:
            if lang in self._lang_data:
                return f"Cannot determine {file} for {lang}"
            else:
                return f"Language {lang} is not supported."

    def license(self, lang="eng"):
        """Return the contents of LICENSE (for omw)
        use lang=lang to get the license for an individual language"""
        return self.doc(file="LICENSE", lang=lang)

    def readme(self, lang="eng"):
        """Return the contents of README (for omw)
        use lang=lang to get the readme for an individual language"""
        return self.doc(file="README", lang=lang)

    def citation(self, lang="eng"):
        """Return the contents of citation.bib file (for omw)
        use lang=lang to get the citation for an individual language"""
        return self.doc(file="citation.bib", lang=lang)

    #############################################################
    # Misc
    #############################################################
    def lemma_count(self, lemma):
        """Return the frequency count for this Lemma"""
        # Currently, count is only work for English
        if lemma._lang != "eng":
            return 0
        # open the count file if we haven't already
        if self._key_count_file is None:
            self._key_count_file = self.open("cntlist.rev")
        # find the key in the counts file and return the count
        line = _binary_search_file(self._key_count_file, lemma._key)
        if line:
            return int(line.rsplit(" ", 1)[-1])
        else:
            return 0

    def path_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
        return synset1.path_similarity(synset2, verbose, simulate_root)

    path_similarity.__doc__ = Synset.path_similarity.__doc__

    def lch_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
        return synset1.lch_similarity(synset2, verbose, simulate_root)

    lch_similarity.__doc__ = Synset.lch_similarity.__doc__

    def wup_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
        return synset1.wup_similarity(synset2, verbose, simulate_root)

    wup_similarity.__doc__ = Synset.wup_similarity.__doc__

    def res_similarity(self, synset1, synset2, ic, verbose=False):
        return synset1.res_similarity(synset2, ic, verbose)

    res_similarity.__doc__ = Synset.res_similarity.__doc__

    def jcn_similarity(self, synset1, synset2, ic, verbose=False):
        return synset1.jcn_similarity(synset2, ic, verbose)

    jcn_similarity.__doc__ = Synset.jcn_similarity.__doc__

    def lin_similarity(self, synset1, synset2, ic, verbose=False):
        return synset1.lin_similarity(synset2, ic, verbose)

    lin_similarity.__doc__ = Synset.lin_similarity.__doc__

    #############################################################
    # Morphy
    #############################################################
    # Morphy, adapted from Oliver Steele's pywordnet
    def morphy(self, form, pos=None, check_exceptions=True):
        """
        Find a possible base form for the given form, with the given
        part of speech, by checking WordNet's list of exceptional
        forms, and by recursively stripping affixes for this part of
        speech until a form in WordNet is found.

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.morphy('dogs'))
        dog
        >>> print(wn.morphy('churches'))
        church
        >>> print(wn.morphy('aardwolves'))
        aardwolf
        >>> print(wn.morphy('abaci'))
        abacus
        >>> wn.morphy('hardrock', wn.ADV)
        >>> print(wn.morphy('book', wn.NOUN))
        book
        >>> wn.morphy('book', wn.ADJ)
        """

        if pos is None:
            morphy = self._morphy
            analyses = chain(a for p in POS_LIST for a in morphy(form, p))
        else:
            analyses = self._morphy(form, pos, check_exceptions)

        # get the first one we find
        first = list(islice(analyses, 1))
        if len(first) == 1:
            return first[0]
        else:
            return None

    MORPHOLOGICAL_SUBSTITUTIONS = {
        NOUN: [
            ("s", ""),
            ("ses", "s"),
            ("ves", "f"),
            ("xes", "x"),
            ("zes", "z"),
            ("ches", "ch"),
            ("shes", "sh"),
            ("men", "man"),
            ("ies", "y"),
        ],
        VERB: [
            ("s", ""),
            ("ies", "y"),
            ("es", "e"),
            ("es", ""),
            ("ed", "e"),
            ("ed", ""),
            ("ing", "e"),
            ("ing", ""),
        ],
        ADJ: [("er", ""), ("est", ""), ("er", "e"), ("est", "e")],
        ADV: [],
    }

    MORPHOLOGICAL_SUBSTITUTIONS[ADJ_SAT] = MORPHOLOGICAL_SUBSTITUTIONS[ADJ]

    def _morphy(self, form, pos, check_exceptions=True):
        # from jordanbg:
        # Given an original string x
        # 1. Apply rules once to the input to get y1, y2, y3, etc.
        # 2. Return all that are in the database
        # 3. If there are no matches, keep applying rules until you either
        #    find a match or you can't go any further

        exceptions = self._exception_map[pos]
        substitutions = self.MORPHOLOGICAL_SUBSTITUTIONS[pos]

        def apply_rules(forms):
            return [
                form[: -len(old)] + new
                for form in forms
                for old, new in substitutions
                if form.endswith(old)
            ]

        def filter_forms(forms):
            result = []
            seen = set()
            for form in forms:
                if form in self._lemma_pos_offset_map:
                    if pos in self._lemma_pos_offset_map[form]:
                        if form not in seen:
                            result.append(form)
                            seen.add(form)
            return result

        # 0. Check the exception lists
        if check_exceptions:
            if form in exceptions:
                return filter_forms([form] + exceptions[form])

        # 1. Apply rules once to the input to get y1, y2, y3, etc.
        forms = apply_rules([form])

        # 2. Return all that are in the database (and check the original too)
        results = filter_forms([form] + forms)
        if results:
            return results

        # 3. If there are no matches, keep applying rules until we find a match
        while forms:
            forms = apply_rules(forms)
            results = filter_forms(forms)
            if results:
                return results

        # Return an empty list if we can't find anything
        return []

    #############################################################
    # Create information content from corpus
    #############################################################
    def ic(self, corpus, weight_senses_equally=False, smoothing=1.0):
        """
        Creates an information content lookup dictionary from a corpus.

        :type corpus: CorpusReader
        :param corpus: The corpus from which we create an information
            content dictionary.
        :type weight_senses_equally: bool
        :param weight_senses_equally: If this is True, gives all
            possible senses equal weight rather than dividing by the
            number of possible senses.  (If a word has 3 synses, each
            sense gets 0.3333 per appearance when this is False, 1.0 when
            it is true.)
        :param smoothing: How much do we smooth synset counts (default is 1.0)
        :type smoothing: float
        :return: An information content dictionary
        """
        counts = FreqDist()
        for ww in corpus.words():
            counts[ww] += 1

        ic = {}
        for pp in POS_LIST:
            ic[pp] = defaultdict(float)

        # Initialize the counts with the smoothing value
        if smoothing > 0.0:
            for pp in POS_LIST:
                ic[pp][0] = smoothing
            for ss in self.all_synsets():
                pos = ss._pos
                if pos == ADJ_SAT:
                    pos = ADJ
                ic[pos][ss._offset] = smoothing

        for ww in counts:
            possible_synsets = self.synsets(ww)
            if len(possible_synsets) == 0:
                continue

            # Distribute weight among possible synsets
            weight = float(counts[ww])
            if not weight_senses_equally:
                weight /= float(len(possible_synsets))

            for ss in possible_synsets:
                pos = ss._pos
                if pos == ADJ_SAT:
                    pos = ADJ
                for level in ss._iter_hypernym_lists():
                    for hh in level:
                        ic[pos][hh._offset] += weight
                # Add the weight to the root
                ic[pos][0] += weight
        return ic

    def custom_lemmas(self, tab_file, lang):
        """
        Reads a custom tab file containing mappings of lemmas in the given
        language to Princeton WordNet 3.0 synset offsets, allowing NLTK's
        WordNet functions to then be used with that language.

        See the "Tab files" section at https://omwn.org/omw1.html for
        documentation on the Multilingual WordNet tab file format.

        :param tab_file: Tab file as a file or file-like object
        :type: lang str
        :param: lang ISO 639-3 code of the language of the tab file
        """
        lg = lang.split("_")[0]
        if len(lg) != 3:
            raise ValueError("lang should be a (3 character) ISO 639-3 code")
        self._lang_data[lang] = [
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        ]
        for line in tab_file.readlines():
            if isinstance(line, bytes):
                # Support byte-stream files (e.g. as returned by Python 2's
                # open() function) as well as text-stream ones
                line = line.decode("utf-8")
            if not line.startswith("#"):
                triple = line.strip().split("\t")
                if len(triple) < 3:
                    continue
                offset_pos, label = triple[:2]
                val = triple[-1]
                if self.map30:
                    if offset_pos in self.map30:
                        # Map offset_pos to current Wordnet version:
                        offset_pos = self.map30[offset_pos]
                    else:
                        # Some OMW offsets were never in Wordnet:
                        if (
                            offset_pos not in self.nomap
                            and offset_pos.replace("a", "s") not in self.nomap
                        ):
                            warnings.warn(
                                f"{lang}: invalid offset {offset_pos} in '{line}'"
                            )
                        continue
                elif offset_pos[-1] == "a":
                    wnss = self.of2ss(offset_pos)
                    if wnss and wnss.pos() == "s":  # Wordnet pos is "s"
                        # Label OMW adjective satellites back to their Wordnet pos ("s")
                        offset_pos = self.ss2of(wnss)
                pair = label.split(":")
                attr = pair[-1]
                if len(pair) == 1 or pair[0] == lg:
                    if attr == "lemma":
                        val = val.strip().replace(" ", "_")
                        self._lang_data[lang][1][val.lower()].append(offset_pos)
                    if attr in self.lg_attrs:
                        self._lang_data[lang][self.lg_attrs.index(attr)][
                            offset_pos
                        ].append(val)

    def disable_custom_lemmas(self, lang):
        """prevent synsets from being mistakenly added"""
        for n in range(len(self.lg_attrs)):
            self._lang_data[lang][n].default_factory = None

    ######################################################################
    # Visualize WordNet relation graphs using Graphviz
    ######################################################################

    def digraph(
        self,
        inputs,
        rel=lambda s: s.hypernyms(),
        pos=None,
        maxdepth=-1,
        shapes=None,
        attr=None,
        verbose=False,
    ):
        """
        Produce a graphical representation from 'inputs' (a list of
        start nodes, which can be a mix of Synsets, Lemmas and/or words),
        and a synset relation, for drawing with the 'dot' graph visualisation
        program from the Graphviz package.

        Return a string in the DOT graph file language, which can then be
        converted to an image by nltk.parse.dependencygraph.dot2img(dot_string).

        Optional Parameters:
        :rel: Wordnet synset relation
        :pos: for words, restricts Part of Speech to 'n', 'v', 'a' or 'r'
        :maxdepth: limit the longest path
        :shapes: dictionary of strings that trigger a specified shape
        :attr: dictionary with global graph attributes
        :verbose: warn about cycles

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.digraph([wn.synset('dog.n.01')]))
        digraph G {
        "Synset('animal.n.01')" -> "Synset('organism.n.01')";
        "Synset('canine.n.02')" -> "Synset('carnivore.n.01')";
        "Synset('carnivore.n.01')" -> "Synset('placental.n.01')";
        "Synset('chordate.n.01')" -> "Synset('animal.n.01')";
        "Synset('dog.n.01')" -> "Synset('canine.n.02')";
        "Synset('dog.n.01')" -> "Synset('domestic_animal.n.01')";
        "Synset('domestic_animal.n.01')" -> "Synset('animal.n.01')";
        "Synset('living_thing.n.01')" -> "Synset('whole.n.02')";
        "Synset('mammal.n.01')" -> "Synset('vertebrate.n.01')";
        "Synset('object.n.01')" -> "Synset('physical_entity.n.01')";
        "Synset('organism.n.01')" -> "Synset('living_thing.n.01')";
        "Synset('physical_entity.n.01')" -> "Synset('entity.n.01')";
        "Synset('placental.n.01')" -> "Synset('mammal.n.01')";
        "Synset('vertebrate.n.01')" -> "Synset('chordate.n.01')";
        "Synset('whole.n.02')" -> "Synset('object.n.01')";
        }
        <BLANKLINE>
        """
        from nltk.util import edge_closure, edges2dot

        synsets = set()
        edges = set()
        if not shapes:
            shapes = dict()
        if not attr:
            attr = dict()

        def add_lemma(lem):
            ss = lem.synset()
            synsets.add(ss)
            edges.add((lem, ss))

        for node in inputs:
            typ = type(node)
            if typ == Synset:
                synsets.add(node)
            elif typ == Lemma:
                add_lemma(node)
            elif typ == str:
                for lemma in self.lemmas(node, pos):
                    add_lemma(lemma)

        for ss in synsets:
            edges = edges.union(edge_closure(ss, rel, maxdepth, verbose))
        dot_string = edges2dot(sorted(list(edges)), shapes=shapes, attr=attr)
        return dot_string


######################################################################
# WordNet Information Content Corpus Reader
######################################################################


class WordNetICCorpusReader(CorpusReader):
    """
    A corpus reader for the WordNet information content corpus.
    """

    def __init__(self, root, fileids):
        CorpusReader.__init__(self, root, fileids, encoding="utf8")

    # this load function would be more efficient if the data was pickled
    # Note that we can't use NLTK's frequency distributions because
    # synsets are overlapping (each instance of a synset also counts
    # as an instance of its hypernyms)
    def ic(self, icfile):
        """
        Load an information content file from the wordnet_ic corpus
        and return a dictionary.  This dictionary has just two keys,
        NOUN and VERB, whose values are dictionaries that map from
        synsets to information content values.

        :type icfile: str
        :param icfile: The name of the wordnet_ic file (e.g. "ic-brown.dat")
        :return: An information content dictionary
        """
        ic = {}
        ic[NOUN] = defaultdict(float)
        ic[VERB] = defaultdict(float)
        with self.open(icfile) as fp:
            for num, line in enumerate(fp):
                if num == 0:  # skip the header
                    continue
                fields = line.split()
                offset = int(fields[0][:-1])
                value = float(fields[1])
                pos = _get_pos(fields[0])
                if len(fields) == 3 and fields[2] == "ROOT":
                    # Store root count.
                    ic[pos][0] += value
                if value != 0:
                    ic[pos][offset] = value
        return ic


######################################################################
# Similarity metrics
######################################################################

# TODO: Add in the option to manually add a new root node; this will be
# useful for verb similarity as there exist multiple verb taxonomies.

# More information about the metrics is available at
# http://marimba.d.umn.edu/similarity/measures.html


def path_similarity(synset1, synset2, verbose=False, simulate_root=True):
    return synset1.path_similarity(
        synset2, verbose=verbose, simulate_root=simulate_root
    )


def lch_similarity(synset1, synset2, verbose=False, simulate_root=True):
    return synset1.lch_similarity(synset2, verbose=verbose, simulate_root=simulate_root)


def wup_similarity(synset1, synset2, verbose=False, simulate_root=True):
    return synset1.wup_similarity(synset2, verbose=verbose, simulate_root=simulate_root)


def res_similarity(synset1, synset2, ic, verbose=False):
    return synset1.res_similarity(synset2, ic, verbose=verbose)


def jcn_similarity(synset1, synset2, ic, verbose=False):
    return synset1.jcn_similarity(synset2, ic, verbose=verbose)


def lin_similarity(synset1, synset2, ic, verbose=False):
    return synset1.lin_similarity(synset2, ic, verbose=verbose)


path_similarity.__doc__ = Synset.path_similarity.__doc__
lch_similarity.__doc__ = Synset.lch_similarity.__doc__
wup_similarity.__doc__ = Synset.wup_similarity.__doc__
res_similarity.__doc__ = Synset.res_similarity.__doc__
jcn_similarity.__doc__ = Synset.jcn_similarity.__doc__
lin_similarity.__doc__ = Synset.lin_similarity.__doc__


def _lcs_ic(synset1, synset2, ic, verbose=False):
    """
    Get the information content of the least common subsumer that has
    the highest information content value.  If two nodes have no
    explicit common subsumer, assume that they share an artificial
    root node that is the hypernym of all explicit roots.

    :type synset1: Synset
    :param synset1: First input synset.
    :type synset2: Synset
    :param synset2: Second input synset.  Must be the same part of
    speech as the first synset.
    :type  ic: dict
    :param ic: an information content object (as returned by ``load_ic()``).
    :return: The information content of the two synsets and their most
    informative subsumer
    """
    if synset1._pos != synset2._pos:
        raise WordNetError(
            "Computing the least common subsumer requires "
            "%s and %s to have the same part of speech." % (synset1, synset2)
        )

    ic1 = information_content(synset1, ic)
    ic2 = information_content(synset2, ic)
    subsumers = synset1.common_hypernyms(synset2)
    if len(subsumers) == 0:
        subsumer_ic = 0
    else:
        subsumer_ic = max(information_content(s, ic) for s in subsumers)

    if verbose:
        print("> LCS Subsumer by content:", subsumer_ic)

    return ic1, ic2, subsumer_ic


# Utility functions


def information_content(synset, ic):
    pos = synset._pos
    if pos == ADJ_SAT:
        pos = ADJ
    try:
        icpos = ic[pos]
    except KeyError as e:
        msg = "Information content file has no entries for part-of-speech: %s"
        raise WordNetError(msg % pos) from e

    counts = icpos[synset._offset]
    if counts == 0:
        return _INF
    else:
        return -math.log(counts / icpos[0])


# get the part of speech (NOUN or VERB) from the information content record
# (each identifier has a 'n' or 'v' suffix)


def _get_pos(field):
    if field[-1] == "n":
        return NOUN
    elif field[-1] == "v":
        return VERB
    else:
        msg = (
            "Unidentified part of speech in WordNet Information Content file "
            "for field %s" % field
        )
        raise ValueError(msg)
