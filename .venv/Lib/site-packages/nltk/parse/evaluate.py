# Natural Language Toolkit: evaluation of dependency parser
#
# Author: Long Duong <longdt219@gmail.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import unicodedata


class DependencyEvaluator:
    """
    Class for measuring labelled and unlabelled attachment score for
    dependency parsing. Note that the evaluation ignores punctuation.

    >>> from nltk.parse import DependencyGraph, DependencyEvaluator

    >>> gold_sent = DependencyGraph(\"""
    ... Pierre  NNP     2       NMOD
    ... Vinken  NNP     8       SUB
    ... ,       ,       2       P
    ... 61      CD      5       NMOD
    ... years   NNS     6       AMOD
    ... old     JJ      2       NMOD
    ... ,       ,       2       P
    ... will    MD      0       ROOT
    ... join    VB      8       VC
    ... the     DT      11      NMOD
    ... board   NN      9       OBJ
    ... as      IN      9       VMOD
    ... a       DT      15      NMOD
    ... nonexecutive    JJ      15      NMOD
    ... director        NN      12      PMOD
    ... Nov.    NNP     9       VMOD
    ... 29      CD      16      NMOD
    ... .       .       9       VMOD
    ... \""")

    >>> parsed_sent = DependencyGraph(\"""
    ... Pierre  NNP     8       NMOD
    ... Vinken  NNP     1       SUB
    ... ,       ,       3       P
    ... 61      CD      6       NMOD
    ... years   NNS     6       AMOD
    ... old     JJ      2       NMOD
    ... ,       ,       3       AMOD
    ... will    MD      0       ROOT
    ... join    VB      8       VC
    ... the     DT      11      AMOD
    ... board   NN      9       OBJECT
    ... as      IN      9       NMOD
    ... a       DT      15      NMOD
    ... nonexecutive    JJ      15      NMOD
    ... director        NN      12      PMOD
    ... Nov.    NNP     9       VMOD
    ... 29      CD      16      NMOD
    ... .       .       9       VMOD
    ... \""")

    >>> de = DependencyEvaluator([parsed_sent],[gold_sent])
    >>> las, uas = de.eval()
    >>> las
    0.6
    >>> uas
    0.8
    >>> abs(uas - 0.8) < 0.00001
    True
    """

    def __init__(self, parsed_sents, gold_sents):
        """
        :param parsed_sents: the list of parsed_sents as the output of parser
        :type parsed_sents: list(DependencyGraph)
        """
        self._parsed_sents = parsed_sents
        self._gold_sents = gold_sents

    def _remove_punct(self, inStr):
        """
        Function to remove punctuation from Unicode string.
        :param input: the input string
        :return: Unicode string after remove all punctuation
        """
        punc_cat = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
        return "".join(x for x in inStr if unicodedata.category(x) not in punc_cat)

    def eval(self):
        """
        Return the Labeled Attachment Score (LAS) and Unlabeled Attachment Score (UAS)

        :return : tuple(float,float)
        """
        if len(self._parsed_sents) != len(self._gold_sents):
            raise ValueError(
                " Number of parsed sentence is different with number of gold sentence."
            )

        corr = 0
        corrL = 0
        total = 0

        for i in range(len(self._parsed_sents)):
            parsed_sent_nodes = self._parsed_sents[i].nodes
            gold_sent_nodes = self._gold_sents[i].nodes

            if len(parsed_sent_nodes) != len(gold_sent_nodes):
                raise ValueError("Sentences must have equal length.")

            for parsed_node_address, parsed_node in parsed_sent_nodes.items():
                gold_node = gold_sent_nodes[parsed_node_address]

                if parsed_node["word"] is None:
                    continue
                if parsed_node["word"] != gold_node["word"]:
                    raise ValueError("Sentence sequence is not matched.")

                # Ignore if word is punctuation by default
                # if (parsed_sent[j]["word"] in string.punctuation):
                if self._remove_punct(parsed_node["word"]) == "":
                    continue

                total += 1
                if parsed_node["head"] == gold_node["head"]:
                    corr += 1
                    if parsed_node["rel"] == gold_node["rel"]:
                        corrL += 1

        return corrL / total, corr / total
