# Natural Language Toolkit: Sequential Backoff Taggers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
#         Tiago Tresoldi <tresoldi@users.sf.net> (original affix tagger)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Classes for tagging sentences sequentially, left to right.  The
abstract base class SequentialBackoffTagger serves as the base
class for all the taggers in this module.  Tagging of individual words
is performed by the method ``choose_tag()``, which is defined by
subclasses of SequentialBackoffTagger.  If a tagger is unable to
determine a tag for the specified token, then its backoff tagger is
consulted instead.  Any SequentialBackoffTagger may serve as a
backoff tagger for any other SequentialBackoffTagger.
"""
import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple

from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI


######################################################################
# Abstract Base Classes
######################################################################
class SequentialBackoffTagger(TaggerI):
    """
    An abstract base class for taggers that tags words sequentially,
    left to right.  Tagging of individual words is performed by the
    ``choose_tag()`` method, which should be defined by subclasses.  If
    a tagger is unable to determine a tag for the specified token,
    then its backoff tagger is consulted.

    :ivar _taggers: A list of all the taggers that should be tried to
        tag a token (i.e., self and its backoff taggers).
    """

    def __init__(self, backoff=None):
        if backoff is None:
            self._taggers = [self]
        else:
            self._taggers = [self] + backoff._taggers

    @property
    def backoff(self):
        """The backoff tagger for this tagger."""
        return self._taggers[1] if len(self._taggers) > 1 else None

    def tag(self, tokens):
        # docs inherited from TaggerI
        tags = []
        for i in range(len(tokens)):
            tags.append(self.tag_one(tokens, i, tags))
        return list(zip(tokens, tags))

    def tag_one(self, tokens, index, history):
        """
        Determine an appropriate tag for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, then its backoff tagger is consulted.

        :rtype: str
        :type tokens: list
        :param tokens: The list of words that are being tagged.
        :type index: int
        :param index: The index of the word whose tag should be
            returned.
        :type history: list(str)
        :param history: A list of the tags for all words before *index*.
        """
        tag = None
        for tagger in self._taggers:
            tag = tagger.choose_tag(tokens, index, history)
            if tag is not None:
                break
        return tag

    @abstractmethod
    def choose_tag(self, tokens, index, history):
        """
        Decide which tag should be used for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, return None -- do not consult
        the backoff tagger.  This method should be overridden by
        subclasses of SequentialBackoffTagger.

        :rtype: str
        :type tokens: list
        :param tokens: The list of words that are being tagged.
        :type index: int
        :param index: The index of the word whose tag should be
            returned.
        :type history: list(str)
        :param history: A list of the tags for all words before *index*.
        """


class ContextTagger(SequentialBackoffTagger):
    """
    An abstract base class for sequential backoff taggers that choose
    a tag for a token based on the value of its "context".  Different
    subclasses are used to define different contexts.

    A ContextTagger chooses the tag for a token by calculating the
    token's context, and looking up the corresponding tag in a table.
    This table can be constructed manually; or it can be automatically
    constructed based on a training corpus, using the ``_train()``
    factory method.

    :ivar _context_to_tag: Dictionary mapping contexts to tags.
    """

    def __init__(self, context_to_tag, backoff=None):
        """
        :param context_to_tag: A dictionary mapping contexts to tags.
        :param backoff: The backoff tagger that should be used for this tagger.
        """
        super().__init__(backoff)
        self._context_to_tag = context_to_tag if context_to_tag else {}

    @abstractmethod
    def context(self, tokens, index, history):
        """
        :return: the context that should be used to look up the tag
            for the specified token; or None if the specified token
            should not be handled by this tagger.
        :rtype: (hashable)
        """

    def choose_tag(self, tokens, index, history):
        context = self.context(tokens, index, history)
        return self._context_to_tag.get(context)

    def size(self):
        """
        :return: The number of entries in the table used by this
            tagger to map from contexts to tags.
        """
        return len(self._context_to_tag)

    def __repr__(self):
        return f"<{self.__class__.__name__}: size={self.size()}>"

    def _train(self, tagged_corpus, cutoff=0, verbose=False):
        """
        Initialize this ContextTagger's ``_context_to_tag`` table
        based on the given training data.  In particular, for each
        context ``c`` in the training data, set
        ``_context_to_tag[c]`` to the most frequent tag for that
        context.  However, exclude any contexts that are already
        tagged perfectly by the backoff tagger(s).

        The old value of ``self._context_to_tag`` (if any) is discarded.

        :param tagged_corpus: A tagged corpus.  Each item should be
            a list of (word, tag tuples.
        :param cutoff: If the most likely tag for a context occurs
            fewer than cutoff times, then exclude it from the
            context-to-tag table for the new tagger.
        """

        token_count = hit_count = 0

        # A context is considered 'useful' if it's not already tagged
        # perfectly by the backoff tagger.
        useful_contexts = set()

        # Count how many times each tag occurs in each context.
        fd = ConditionalFreqDist()
        for sentence in tagged_corpus:
            tokens, tags = zip(*sentence)
            for index, (token, tag) in enumerate(sentence):
                # Record the event.
                token_count += 1
                context = self.context(tokens, index, tags[:index])
                if context is None:
                    continue
                fd[context][tag] += 1
                # If the backoff got it wrong, this context is useful:
                if self.backoff is None or tag != self.backoff.tag_one(
                    tokens, index, tags[:index]
                ):
                    useful_contexts.add(context)

        # Build the context_to_tag table -- for each context, figure
        # out what the most likely tag is.  Only include contexts that
        # we've seen at least `cutoff` times.
        for context in useful_contexts:
            best_tag = fd[context].max()
            hits = fd[context][best_tag]
            if hits > cutoff:
                self._context_to_tag[context] = best_tag
                hit_count += hits

        # Display some stats, if requested.
        if verbose:
            size = len(self._context_to_tag)
            backoff = 100 - (hit_count * 100.0) / token_count
            pruning = 100 - (size * 100.0) / len(fd.conditions())
            print("[Trained Unigram tagger:", end=" ")
            print(
                "size={}, backoff={:.2f}%, pruning={:.2f}%]".format(
                    size, backoff, pruning
                )
            )


######################################################################
# Tagger Classes
######################################################################


@jsontags.register_tag
class DefaultTagger(SequentialBackoffTagger):
    """
    A tagger that assigns the same tag to every token.

        >>> from nltk.tag import DefaultTagger
        >>> default_tagger = DefaultTagger('NN')
        >>> list(default_tagger.tag('This is a test'.split()))
        [('This', 'NN'), ('is', 'NN'), ('a', 'NN'), ('test', 'NN')]

    This tagger is recommended as a backoff tagger, in cases where
    a more powerful tagger is unable to assign a tag to the word
    (e.g. because the word was not seen during training).

    :param tag: The tag to assign to each token
    :type tag: str
    """

    json_tag = "nltk.tag.sequential.DefaultTagger"

    def __init__(self, tag):
        self._tag = tag
        super().__init__(None)

    def encode_json_obj(self):
        return self._tag

    @classmethod
    def decode_json_obj(cls, obj):
        tag = obj
        return cls(tag)

    def choose_tag(self, tokens, index, history):
        return self._tag  # ignore token and history

    def __repr__(self):
        return f"<DefaultTagger: tag={self._tag}>"


@jsontags.register_tag
class NgramTagger(ContextTagger):
    """
    A tagger that chooses a token's tag based on its word string and
    on the preceding n word's tags.  In particular, a tuple
    (tags[i-n:i-1], words[i]) is looked up in a table, and the
    corresponding tag is returned.  N-gram taggers are typically
    trained on a tagged corpus.

    Train a new NgramTagger using the given training data or
    the supplied model.  In particular, construct a new tagger
    whose table maps from each context (tag[i-n:i-1], word[i])
    to the most frequent tag for that context.  But exclude any
    contexts that are already tagged perfectly by the backoff
    tagger.

    :param train: A tagged corpus consisting of a list of tagged
        sentences, where each sentence is a list of (word, tag) tuples.
    :param backoff: A backoff tagger, to be used by the new
        tagger if it encounters an unknown context.
    :param cutoff: If the most likely tag for a context occurs
        fewer than *cutoff* times, then exclude it from the
        context-to-tag table for the new tagger.
    """

    json_tag = "nltk.tag.sequential.NgramTagger"

    def __init__(
        self, n, train=None, model=None, backoff=None, cutoff=0, verbose=False
    ):
        self._n = n
        self._check_params(train, model)

        super().__init__(model, backoff)

        if train:
            self._train(train, cutoff, verbose)

    def encode_json_obj(self):
        _context_to_tag = {repr(k): v for k, v in self._context_to_tag.items()}
        if "NgramTagger" in self.__class__.__name__:
            return self._n, _context_to_tag, self.backoff
        else:
            return _context_to_tag, self.backoff

    @classmethod
    def decode_json_obj(cls, obj):
        try:
            _n, _context_to_tag, backoff = obj
        except ValueError:
            _context_to_tag, backoff = obj

        if not _context_to_tag:
            return backoff

        _context_to_tag = {ast.literal_eval(k): v for k, v in _context_to_tag.items()}

        if "NgramTagger" in cls.__name__:
            return cls(_n, model=_context_to_tag, backoff=backoff)
        else:
            return cls(model=_context_to_tag, backoff=backoff)

    def context(self, tokens, index, history):
        tag_context = tuple(history[max(0, index - self._n + 1) : index])
        return tag_context, tokens[index]


@jsontags.register_tag
class UnigramTagger(NgramTagger):
    """
    Unigram Tagger

    The UnigramTagger finds the most likely tag for each word in a training
    corpus, and then uses that information to assign tags to new tokens.

        >>> from nltk.corpus import brown
        >>> from nltk.tag import UnigramTagger
        >>> test_sent = brown.sents(categories='news')[0]
        >>> unigram_tagger = UnigramTagger(brown.tagged_sents(categories='news')[:500])
        >>> for tok, tag in unigram_tagger.tag(test_sent):
        ...     print("({}, {}), ".format(tok, tag)) # doctest: +NORMALIZE_WHITESPACE
        (The, AT), (Fulton, NP-TL), (County, NN-TL), (Grand, JJ-TL),
        (Jury, NN-TL), (said, VBD), (Friday, NR), (an, AT),
        (investigation, NN), (of, IN), (Atlanta's, NP$), (recent, JJ),
        (primary, NN), (election, NN), (produced, VBD), (``, ``),
        (no, AT), (evidence, NN), ('', ''), (that, CS), (any, DTI),
        (irregularities, NNS), (took, VBD), (place, NN), (., .),

    :param train: The corpus of training data, a list of tagged sentences
    :type train: list(list(tuple(str, str)))
    :param model: The tagger model
    :type model: dict
    :param backoff: Another tagger which this tagger will consult when it is
        unable to tag a word
    :type backoff: TaggerI
    :param cutoff: The number of instances of training data the tagger must see
        in order not to use the backoff tagger
    :type cutoff: int
    """

    json_tag = "nltk.tag.sequential.UnigramTagger"

    def __init__(self, train=None, model=None, backoff=None, cutoff=0, verbose=False):
        super().__init__(1, train, model, backoff, cutoff, verbose)

    def context(self, tokens, index, history):
        return tokens[index]


@jsontags.register_tag
class BigramTagger(NgramTagger):
    """
    A tagger that chooses a token's tag based its word string and on
    the preceding words' tag.  In particular, a tuple consisting
    of the previous tag and the word is looked up in a table, and
    the corresponding tag is returned.

    :param train: The corpus of training data, a list of tagged sentences
    :type train: list(list(tuple(str, str)))
    :param model: The tagger model
    :type model: dict
    :param backoff: Another tagger which this tagger will consult when it is
        unable to tag a word
    :type backoff: TaggerI
    :param cutoff: The number of instances of training data the tagger must see
        in order not to use the backoff tagger
    :type cutoff: int
    """

    json_tag = "nltk.tag.sequential.BigramTagger"

    def __init__(self, train=None, model=None, backoff=None, cutoff=0, verbose=False):
        super().__init__(2, train, model, backoff, cutoff, verbose)


@jsontags.register_tag
class TrigramTagger(NgramTagger):
    """
    A tagger that chooses a token's tag based its word string and on
    the preceding two words' tags.  In particular, a tuple consisting
    of the previous two tags and the word is looked up in a table, and
    the corresponding tag is returned.

    :param train: The corpus of training data, a list of tagged sentences
    :type train: list(list(tuple(str, str)))
    :param model: The tagger model
    :type model: dict
    :param backoff: Another tagger which this tagger will consult when it is
        unable to tag a word
    :type backoff: TaggerI
    :param cutoff: The number of instances of training data the tagger must see
        in order not to use the backoff tagger
    :type cutoff: int
    """

    json_tag = "nltk.tag.sequential.TrigramTagger"

    def __init__(self, train=None, model=None, backoff=None, cutoff=0, verbose=False):
        super().__init__(3, train, model, backoff, cutoff, verbose)


@jsontags.register_tag
class AffixTagger(ContextTagger):
    """
    A tagger that chooses a token's tag based on a leading or trailing
    substring of its word string.  (It is important to note that these
    substrings are not necessarily "true" morphological affixes).  In
    particular, a fixed-length substring of the word is looked up in a
    table, and the corresponding tag is returned.  Affix taggers are
    typically constructed by training them on a tagged corpus.

    Construct a new affix tagger.

    :param affix_length: The length of the affixes that should be
        considered during training and tagging.  Use negative
        numbers for suffixes.
    :param min_stem_length: Any words whose length is less than
        min_stem_length+abs(affix_length) will be assigned a
        tag of None by this tagger.
    """

    json_tag = "nltk.tag.sequential.AffixTagger"

    def __init__(
        self,
        train=None,
        model=None,
        affix_length=-3,
        min_stem_length=2,
        backoff=None,
        cutoff=0,
        verbose=False,
    ):

        self._check_params(train, model)

        super().__init__(model, backoff)

        self._affix_length = affix_length
        self._min_word_length = min_stem_length + abs(affix_length)

        if train:
            self._train(train, cutoff, verbose)

    def encode_json_obj(self):
        return (
            self._affix_length,
            self._min_word_length,
            self._context_to_tag,
            self.backoff,
        )

    @classmethod
    def decode_json_obj(cls, obj):
        _affix_length, _min_word_length, _context_to_tag, backoff = obj
        return cls(
            affix_length=_affix_length,
            min_stem_length=_min_word_length - abs(_affix_length),
            model=_context_to_tag,
            backoff=backoff,
        )

    def context(self, tokens, index, history):
        token = tokens[index]
        if len(token) < self._min_word_length:
            return None
        elif self._affix_length > 0:
            return token[: self._affix_length]
        else:
            return token[self._affix_length :]


@jsontags.register_tag
class RegexpTagger(SequentialBackoffTagger):
    r"""
    Regular Expression Tagger

    The RegexpTagger assigns tags to tokens by comparing their
    word strings to a series of regular expressions.  The following tagger
    uses word suffixes to make guesses about the correct Brown Corpus part
    of speech tag:

        >>> from nltk.corpus import brown
        >>> from nltk.tag import RegexpTagger
        >>> test_sent = brown.sents(categories='news')[0]
        >>> regexp_tagger = RegexpTagger(
        ...     [(r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
        ...      (r'(The|the|A|a|An|an)$', 'AT'),   # articles
        ...      (r'.*able$', 'JJ'),                # adjectives
        ...      (r'.*ness$', 'NN'),                # nouns formed from adjectives
        ...      (r'.*ly$', 'RB'),                  # adverbs
        ...      (r'.*s$', 'NNS'),                  # plural nouns
        ...      (r'.*ing$', 'VBG'),                # gerunds
        ...      (r'.*ed$', 'VBD'),                 # past tense verbs
        ...      (r'.*', 'NN')                      # nouns (default)
        ... ])
        >>> regexp_tagger
        <Regexp Tagger: size=9>
        >>> regexp_tagger.tag(test_sent) # doctest: +NORMALIZE_WHITESPACE
        [('The', 'AT'), ('Fulton', 'NN'), ('County', 'NN'), ('Grand', 'NN'), ('Jury', 'NN'),
        ('said', 'NN'), ('Friday', 'NN'), ('an', 'AT'), ('investigation', 'NN'), ('of', 'NN'),
        ("Atlanta's", 'NNS'), ('recent', 'NN'), ('primary', 'NN'), ('election', 'NN'),
        ('produced', 'VBD'), ('``', 'NN'), ('no', 'NN'), ('evidence', 'NN'), ("''", 'NN'),
        ('that', 'NN'), ('any', 'NN'), ('irregularities', 'NNS'), ('took', 'NN'),
        ('place', 'NN'), ('.', 'NN')]

    :type regexps: list(tuple(str, str))
    :param regexps: A list of ``(regexp, tag)`` pairs, each of
        which indicates that a word matching ``regexp`` should
        be tagged with ``tag``.  The pairs will be evaluated in
        order.  If none of the regexps match a word, then the
        optional backoff tagger is invoked, else it is
        assigned the tag None.
    """

    json_tag = "nltk.tag.sequential.RegexpTagger"

    def __init__(
        self, regexps: List[Tuple[str, str]], backoff: Optional[TaggerI] = None
    ):
        super().__init__(backoff)
        self._regexps = []
        for regexp, tag in regexps:
            try:
                self._regexps.append((re.compile(regexp), tag))
            except Exception as e:
                raise Exception(
                    f"Invalid RegexpTagger regexp: {e}\n- regexp: {regexp!r}\n- tag: {tag!r}"
                ) from e

    def encode_json_obj(self):
        return [(regexp.pattern, tag) for regexp, tag in self._regexps], self.backoff

    @classmethod
    def decode_json_obj(cls, obj):
        regexps, backoff = obj
        return cls(regexps, backoff)

    def choose_tag(self, tokens, index, history):
        for regexp, tag in self._regexps:
            if re.match(regexp, tokens[index]):
                return tag
        return None

    def __repr__(self):
        return f"<Regexp Tagger: size={len(self._regexps)}>"


class ClassifierBasedTagger(SequentialBackoffTagger, FeaturesetTaggerI):
    """
    A sequential tagger that uses a classifier to choose the tag for
    each token in a sentence.  The featureset input for the classifier
    is generated by a feature detector function::

        feature_detector(tokens, index, history) -> featureset

    Where tokens is the list of unlabeled tokens in the sentence;
    index is the index of the token for which feature detection
    should be performed; and history is list of the tags for all
    tokens before index.

    Construct a new classifier-based sequential tagger.

    :param feature_detector: A function used to generate the
        featureset input for the classifier::
        feature_detector(tokens, index, history) -> featureset

    :param train: A tagged corpus consisting of a list of tagged
        sentences, where each sentence is a list of (word, tag) tuples.

    :param backoff: A backoff tagger, to be used by the new tagger
        if it encounters an unknown context.

    :param classifier_builder: A function used to train a new
        classifier based on the data in *train*.  It should take
        one argument, a list of labeled featuresets (i.e.,
        (featureset, label) tuples).

    :param classifier: The classifier that should be used by the
        tagger.  This is only useful if you want to manually
        construct the classifier; normally, you would use *train*
        instead.

    :param backoff: A backoff tagger, used if this tagger is
        unable to determine a tag for a given token.

    :param cutoff_prob: If specified, then this tagger will fall
        back on its backoff tagger if the probability of the most
        likely tag is less than *cutoff_prob*.
    """

    def __init__(
        self,
        feature_detector=None,
        train=None,
        classifier_builder=NaiveBayesClassifier.train,
        classifier=None,
        backoff=None,
        cutoff_prob=None,
        verbose=False,
    ):
        self._check_params(train, classifier)

        super().__init__(backoff)

        if (train and classifier) or (not train and not classifier):
            raise ValueError(
                "Must specify either training data or " "trained classifier."
            )

        if feature_detector is not None:
            self._feature_detector = feature_detector
            # The feature detector function, used to generate a featureset
            # or each token: feature_detector(tokens, index, history) -> featureset

        self._cutoff_prob = cutoff_prob
        """Cutoff probability for tagging -- if the probability of the
           most likely tag is less than this, then use backoff."""

        self._classifier = classifier
        """The classifier used to choose a tag for each token."""

        if train:
            self._train(train, classifier_builder, verbose)

    def choose_tag(self, tokens, index, history):
        # Use our feature detector to get the featureset.
        featureset = self.feature_detector(tokens, index, history)

        # Use the classifier to pick a tag.  If a cutoff probability
        # was specified, then check that the tag's probability is
        # higher than that cutoff first; otherwise, return None.
        if self._cutoff_prob is None:
            return self._classifier.classify(featureset)

        pdist = self._classifier.prob_classify(featureset)
        tag = pdist.max()
        return tag if pdist.prob(tag) >= self._cutoff_prob else None

    def _train(self, tagged_corpus, classifier_builder, verbose):
        """
        Build a new classifier, based on the given training data
        *tagged_corpus*.
        """

        classifier_corpus = []
        if verbose:
            print("Constructing training corpus for classifier.")

        for sentence in tagged_corpus:
            history = []
            untagged_sentence, tags = zip(*sentence)
            for index in range(len(sentence)):
                featureset = self.feature_detector(untagged_sentence, index, history)
                classifier_corpus.append((featureset, tags[index]))
                history.append(tags[index])

        if verbose:
            print(f"Training classifier ({len(classifier_corpus)} instances)")
        self._classifier = classifier_builder(classifier_corpus)

    def __repr__(self):
        return f"<ClassifierBasedTagger: {self._classifier}>"

    def feature_detector(self, tokens, index, history):
        """
        Return the feature detector that this tagger uses to generate
        featuresets for its classifier.  The feature detector is a
        function with the signature::

          feature_detector(tokens, index, history) -> featureset

        See ``classifier()``
        """
        return self._feature_detector(tokens, index, history)

    def classifier(self):
        """
        Return the classifier that this tagger uses to choose a tag
        for each word in a sentence.  The input for this classifier is
        generated using this tagger's feature detector.
        See ``feature_detector()``
        """
        return self._classifier


class ClassifierBasedPOSTagger(ClassifierBasedTagger):
    """
    A classifier based part of speech tagger.
    """

    def feature_detector(self, tokens, index, history):
        word = tokens[index]
        if index == 0:
            prevword = prevprevword = None
            prevtag = prevprevtag = None
        elif index == 1:
            prevword = tokens[index - 1].lower()
            prevprevword = None
            prevtag = history[index - 1]
            prevprevtag = None
        else:
            prevword = tokens[index - 1].lower()
            prevprevword = tokens[index - 2].lower()
            prevtag = history[index - 1]
            prevprevtag = history[index - 2]

        if re.match(r"[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$", word):
            shape = "number"
        elif re.match(r"\W+$", word):
            shape = "punct"
        elif re.match("[A-Z][a-z]+$", word):
            shape = "upcase"
        elif re.match("[a-z]+$", word):
            shape = "downcase"
        elif re.match(r"\w+$", word):
            shape = "mixedcase"
        else:
            shape = "other"

        features = {
            "prevtag": prevtag,
            "prevprevtag": prevprevtag,
            "word": word,
            "word.lower": word.lower(),
            "suffix3": word.lower()[-3:],
            "suffix2": word.lower()[-2:],
            "suffix1": word.lower()[-1:],
            "prevprevword": prevprevword,
            "prevword": prevword,
            "prevtag+word": f"{prevtag}+{word.lower()}",
            "prevprevtag+word": f"{prevprevtag}+{word.lower()}",
            "prevword+word": f"{prevword}+{word.lower()}",
            "shape": shape,
        }
        return features
