# Natural Language Toolkit: Stack decoder
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Tah Wei Hoon <hoon.tw@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A decoder that uses stacks to implement phrase-based translation.

In phrase-based translation, the source sentence is segmented into
phrases of one or more words, and translations for those phrases are
used to build the target sentence.

Hypothesis data structures are used to keep track of the source words
translated so far and the partial output. A hypothesis can be expanded
by selecting an untranslated phrase, looking up its translation in a
phrase table, and appending that translation to the partial output.
Translation is complete when a hypothesis covers all source words.

The search space is huge because the source sentence can be segmented
in different ways, the source phrases can be selected in any order,
and there could be multiple translations for the same source phrase in
the phrase table. To make decoding tractable, stacks are used to limit
the number of candidate hypotheses by doing histogram and/or threshold
pruning.

Hypotheses with the same number of words translated are placed in the
same stack. In histogram pruning, each stack has a size limit, and
the hypothesis with the lowest score is removed when the stack is full.
In threshold pruning, hypotheses that score below a certain threshold
of the best hypothesis in that stack are removed.

Hypothesis scoring can include various factors such as phrase
translation probability, language model probability, length of
translation, cost of remaining words to be translated, and so on.


References:
Philipp Koehn. 2010. Statistical Machine Translation.
Cambridge University Press, New York.
"""

import warnings
from collections import defaultdict
from math import log


class StackDecoder:
    """
    Phrase-based stack decoder for machine translation

    >>> from nltk.translate import PhraseTable
    >>> phrase_table = PhraseTable()
    >>> phrase_table.add(('niemand',), ('nobody',), log(0.8))
    >>> phrase_table.add(('niemand',), ('no', 'one'), log(0.2))
    >>> phrase_table.add(('erwartet',), ('expects',), log(0.8))
    >>> phrase_table.add(('erwartet',), ('expecting',), log(0.2))
    >>> phrase_table.add(('niemand', 'erwartet'), ('one', 'does', 'not', 'expect'), log(0.1))
    >>> phrase_table.add(('die', 'spanische', 'inquisition'), ('the', 'spanish', 'inquisition'), log(0.8))
    >>> phrase_table.add(('!',), ('!',), log(0.8))

    >>> #  nltk.model should be used here once it is implemented
    >>> from collections import defaultdict
    >>> language_prob = defaultdict(lambda: -999.0)
    >>> language_prob[('nobody',)] = log(0.5)
    >>> language_prob[('expects',)] = log(0.4)
    >>> language_prob[('the', 'spanish', 'inquisition')] = log(0.2)
    >>> language_prob[('!',)] = log(0.1)
    >>> language_model = type('',(object,),{'probability_change': lambda self, context, phrase: language_prob[phrase], 'probability': lambda self, phrase: language_prob[phrase]})()

    >>> stack_decoder = StackDecoder(phrase_table, language_model)

    >>> stack_decoder.translate(['niemand', 'erwartet', 'die', 'spanische', 'inquisition', '!'])
    ['nobody', 'expects', 'the', 'spanish', 'inquisition', '!']

    """

    def __init__(self, phrase_table, language_model):
        """
        :param phrase_table: Table of translations for source language
            phrases and the log probabilities for those translations.
        :type phrase_table: PhraseTable

        :param language_model: Target language model. Must define a
            ``probability_change`` method that calculates the change in
            log probability of a sentence, if a given string is appended
            to it.
            This interface is experimental and will likely be replaced
            with nltk.model once it is implemented.
        :type language_model: object
        """
        self.phrase_table = phrase_table
        self.language_model = language_model

        self.word_penalty = 0.0
        """
        float: Influences the translation length exponentially.
            If positive, shorter translations are preferred.
            If negative, longer translations are preferred.
            If zero, no penalty is applied.
        """

        self.beam_threshold = 0.0
        """
        float: Hypotheses that score below this factor of the best
            hypothesis in a stack are dropped from consideration.
            Value between 0.0 and 1.0.
        """

        self.stack_size = 100
        """
        int: Maximum number of hypotheses to consider in a stack.
            Higher values increase the likelihood of a good translation,
            but increases processing time.
        """

        self.__distortion_factor = 0.5
        self.__compute_log_distortion()

    @property
    def distortion_factor(self):
        """
        float: Amount of reordering of source phrases.
            Lower values favour monotone translation, suitable when
            word order is similar for both source and target languages.
            Value between 0.0 and 1.0. Default 0.5.
        """
        return self.__distortion_factor

    @distortion_factor.setter
    def distortion_factor(self, d):
        self.__distortion_factor = d
        self.__compute_log_distortion()

    def __compute_log_distortion(self):
        # cache log(distortion_factor) so we don't have to recompute it
        # when scoring hypotheses
        if self.__distortion_factor == 0.0:
            self.__log_distortion_factor = log(1e-9)  # 1e-9 is almost zero
        else:
            self.__log_distortion_factor = log(self.__distortion_factor)

    def translate(self, src_sentence):
        """
        :param src_sentence: Sentence to be translated
        :type src_sentence: list(str)

        :return: Translated sentence
        :rtype: list(str)
        """
        sentence = tuple(src_sentence)  # prevent accidental modification
        sentence_length = len(sentence)
        stacks = [
            _Stack(self.stack_size, self.beam_threshold)
            for _ in range(0, sentence_length + 1)
        ]
        empty_hypothesis = _Hypothesis()
        stacks[0].push(empty_hypothesis)

        all_phrases = self.find_all_src_phrases(sentence)
        future_score_table = self.compute_future_scores(sentence)
        for stack in stacks:
            for hypothesis in stack:
                possible_expansions = StackDecoder.valid_phrases(
                    all_phrases, hypothesis
                )
                for src_phrase_span in possible_expansions:
                    src_phrase = sentence[src_phrase_span[0] : src_phrase_span[1]]
                    for translation_option in self.phrase_table.translations_for(
                        src_phrase
                    ):
                        raw_score = self.expansion_score(
                            hypothesis, translation_option, src_phrase_span
                        )
                        new_hypothesis = _Hypothesis(
                            raw_score=raw_score,
                            src_phrase_span=src_phrase_span,
                            trg_phrase=translation_option.trg_phrase,
                            previous=hypothesis,
                        )
                        new_hypothesis.future_score = self.future_score(
                            new_hypothesis, future_score_table, sentence_length
                        )
                        total_words = new_hypothesis.total_translated_words()
                        stacks[total_words].push(new_hypothesis)

        if not stacks[sentence_length]:
            warnings.warn(
                "Unable to translate all words. "
                "The source sentence contains words not in "
                "the phrase table"
            )
            # Instead of returning empty output, perhaps a partial
            # translation could be returned
            return []

        best_hypothesis = stacks[sentence_length].best()
        return best_hypothesis.translation_so_far()

    def find_all_src_phrases(self, src_sentence):
        """
        Finds all subsequences in src_sentence that have a phrase
        translation in the translation table

        :type src_sentence: tuple(str)

        :return: Subsequences that have a phrase translation,
            represented as a table of lists of end positions.
            For example, if result[2] is [5, 6, 9], then there are
            three phrases starting from position 2 in ``src_sentence``,
            ending at positions 5, 6, and 9 exclusive. The list of
            ending positions are in ascending order.
        :rtype: list(list(int))
        """
        sentence_length = len(src_sentence)
        phrase_indices = [[] for _ in src_sentence]
        for start in range(0, sentence_length):
            for end in range(start + 1, sentence_length + 1):
                potential_phrase = src_sentence[start:end]
                if potential_phrase in self.phrase_table:
                    phrase_indices[start].append(end)
        return phrase_indices

    def compute_future_scores(self, src_sentence):
        """
        Determines the approximate scores for translating every
        subsequence in ``src_sentence``

        Future scores can be used a look-ahead to determine the
        difficulty of translating the remaining parts of a src_sentence.

        :type src_sentence: tuple(str)

        :return: Scores of subsequences referenced by their start and
            end positions. For example, result[2][5] is the score of the
            subsequence covering positions 2, 3, and 4.
        :rtype: dict(int: (dict(int): float))
        """
        scores = defaultdict(lambda: defaultdict(lambda: float("-inf")))
        for seq_length in range(1, len(src_sentence) + 1):
            for start in range(0, len(src_sentence) - seq_length + 1):
                end = start + seq_length
                phrase = src_sentence[start:end]
                if phrase in self.phrase_table:
                    score = self.phrase_table.translations_for(phrase)[
                        0
                    ].log_prob  # pick best (first) translation
                    # Warning: API of language_model is subject to change
                    score += self.language_model.probability(phrase)
                    scores[start][end] = score

                # check if a better score can be obtained by combining
                # two child subsequences
                for mid in range(start + 1, end):
                    combined_score = scores[start][mid] + scores[mid][end]
                    if combined_score > scores[start][end]:
                        scores[start][end] = combined_score
        return scores

    def future_score(self, hypothesis, future_score_table, sentence_length):
        """
        Determines the approximate score for translating the
        untranslated words in ``hypothesis``
        """
        score = 0.0
        for span in hypothesis.untranslated_spans(sentence_length):
            score += future_score_table[span[0]][span[1]]
        return score

    def expansion_score(self, hypothesis, translation_option, src_phrase_span):
        """
        Calculate the score of expanding ``hypothesis`` with
        ``translation_option``

        :param hypothesis: Hypothesis being expanded
        :type hypothesis: _Hypothesis

        :param translation_option: Information about the proposed expansion
        :type translation_option: PhraseTableEntry

        :param src_phrase_span: Word position span of the source phrase
        :type src_phrase_span: tuple(int, int)
        """
        score = hypothesis.raw_score
        score += translation_option.log_prob
        # The API of language_model is subject to change; it could accept
        # a string, a list of words, and/or some other type
        score += self.language_model.probability_change(
            hypothesis, translation_option.trg_phrase
        )
        score += self.distortion_score(hypothesis, src_phrase_span)
        score -= self.word_penalty * len(translation_option.trg_phrase)
        return score

    def distortion_score(self, hypothesis, next_src_phrase_span):
        if not hypothesis.src_phrase_span:
            return 0.0
        next_src_phrase_start = next_src_phrase_span[0]
        prev_src_phrase_end = hypothesis.src_phrase_span[1]
        distortion_distance = next_src_phrase_start - prev_src_phrase_end
        return abs(distortion_distance) * self.__log_distortion_factor

    @staticmethod
    def valid_phrases(all_phrases_from, hypothesis):
        """
        Extract phrases from ``all_phrases_from`` that contains words
        that have not been translated by ``hypothesis``

        :param all_phrases_from: Phrases represented by their spans, in
            the same format as the return value of
            ``find_all_src_phrases``
        :type all_phrases_from: list(list(int))

        :type hypothesis: _Hypothesis

        :return: A list of phrases, represented by their spans, that
            cover untranslated positions.
        :rtype: list(tuple(int, int))
        """
        untranslated_spans = hypothesis.untranslated_spans(len(all_phrases_from))
        valid_phrases = []
        for available_span in untranslated_spans:
            start = available_span[0]
            available_end = available_span[1]
            while start < available_end:
                for phrase_end in all_phrases_from[start]:
                    if phrase_end > available_end:
                        # Subsequent elements in all_phrases_from[start]
                        # will also be > available_end, since the
                        # elements are in ascending order
                        break
                    valid_phrases.append((start, phrase_end))
                start += 1
        return valid_phrases


class _Hypothesis:
    """
    Partial solution to a translation.

    Records the word positions of the phrase being translated, its
    translation, raw score, and the cost of the untranslated parts of
    the sentence. When the next phrase is selected to build upon the
    partial solution, a new _Hypothesis object is created, with a back
    pointer to the previous hypothesis.

    To find out which words have been translated so far, look at the
    ``src_phrase_span`` in the hypothesis chain. Similarly, the
    translation output can be found by traversing up the chain.
    """

    def __init__(
        self,
        raw_score=0.0,
        src_phrase_span=(),
        trg_phrase=(),
        previous=None,
        future_score=0.0,
    ):
        """
        :param raw_score: Likelihood of hypothesis so far.
            Higher is better. Does not account for untranslated words.
        :type raw_score: float

        :param src_phrase_span: Span of word positions covered by the
            source phrase in this hypothesis expansion. For example,
            (2, 5) means that the phrase is from the second word up to,
            but not including the fifth word in the source sentence.
        :type src_phrase_span: tuple(int)

        :param trg_phrase: Translation of the source phrase in this
            hypothesis expansion
        :type trg_phrase: tuple(str)

        :param previous: Previous hypothesis before expansion to this one
        :type previous: _Hypothesis

        :param future_score: Approximate score for translating the
            remaining words not covered by this hypothesis. Higher means
            that the remaining words are easier to translate.
        :type future_score: float
        """
        self.raw_score = raw_score
        self.src_phrase_span = src_phrase_span
        self.trg_phrase = trg_phrase
        self.previous = previous
        self.future_score = future_score

    def score(self):
        """
        Overall score of hypothesis after accounting for local and
        global features
        """
        return self.raw_score + self.future_score

    def untranslated_spans(self, sentence_length):
        """
        Starting from each untranslated word, find the longest
        continuous span of untranslated positions

        :param sentence_length: Length of source sentence being
            translated by the hypothesis
        :type sentence_length: int

        :rtype: list(tuple(int, int))
        """
        translated_positions = self.translated_positions()
        translated_positions.sort()
        translated_positions.append(sentence_length)  # add sentinel position

        untranslated_spans = []
        start = 0
        # each untranslated span must end in one of the translated_positions
        for end in translated_positions:
            if start < end:
                untranslated_spans.append((start, end))
            start = end + 1

        return untranslated_spans

    def translated_positions(self):
        """
        List of positions in the source sentence of words already
        translated. The list is not sorted.

        :rtype: list(int)
        """
        translated_positions = []
        current_hypothesis = self
        while current_hypothesis.previous is not None:
            translated_span = current_hypothesis.src_phrase_span
            translated_positions.extend(range(translated_span[0], translated_span[1]))
            current_hypothesis = current_hypothesis.previous
        return translated_positions

    def total_translated_words(self):
        return len(self.translated_positions())

    def translation_so_far(self):
        translation = []
        self.__build_translation(self, translation)
        return translation

    def __build_translation(self, hypothesis, output):
        if hypothesis.previous is None:
            return
        self.__build_translation(hypothesis.previous, output)
        output.extend(hypothesis.trg_phrase)


class _Stack:
    """
    Collection of _Hypothesis objects
    """

    def __init__(self, max_size=100, beam_threshold=0.0):
        """
        :param beam_threshold: Hypotheses that score less than this
            factor of the best hypothesis are discarded from the stack.
            Value must be between 0.0 and 1.0.
        :type beam_threshold: float
        """
        self.max_size = max_size
        self.items = []

        if beam_threshold == 0.0:
            self.__log_beam_threshold = float("-inf")
        else:
            self.__log_beam_threshold = log(beam_threshold)

    def push(self, hypothesis):
        """
        Add ``hypothesis`` to the stack.
        Removes lowest scoring hypothesis if the stack is full.
        After insertion, hypotheses that score less than
        ``beam_threshold`` times the score of the best hypothesis
        are removed.
        """
        self.items.append(hypothesis)
        self.items.sort(key=lambda h: h.score(), reverse=True)
        while len(self.items) > self.max_size:
            self.items.pop()
        self.threshold_prune()

    def threshold_prune(self):
        if not self.items:
            return
        #  log(score * beam_threshold) = log(score) + log(beam_threshold)
        threshold = self.items[0].score() + self.__log_beam_threshold
        for hypothesis in reversed(self.items):
            if hypothesis.score() < threshold:
                self.items.pop()
            else:
                break

    def best(self):
        """
        :return: Hypothesis with the highest score in the stack
        :rtype: _Hypothesis
        """
        if self.items:
            return self.items[0]
        return None

    def __iter__(self):
        return iter(self.items)

    def __contains__(self, hypothesis):
        return hypothesis in self.items

    def __bool__(self):
        return len(self.items) != 0

    __nonzero__ = __bool__
