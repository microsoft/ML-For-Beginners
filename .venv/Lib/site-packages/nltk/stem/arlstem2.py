#
# Natural Language Toolkit: ARLSTem Stemmer v2
#
# Copyright (C) 2001-2023 NLTK Project
#
# Author: Kheireddine Abainia (x-programer) <k.abainia@gmail.com>
# Algorithms: Kheireddine Abainia <k.abainia@gmail.com>
#                         Hamza Rebbani <hamrebbani@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


"""
ARLSTem2 Arabic Light Stemmer
The details about the implementation of this algorithm are described in:
K. Abainia and H. Rebbani, Comparing the Effectiveness of the Improved ARLSTem
Algorithm with Existing Arabic Light Stemmers, International Conference on
Theoretical and Applicative Aspects of Computer Science (ICTAACS'19), Skikda,
Algeria, December 15-16, 2019.
ARLSTem2 is an Arabic light stemmer based on removing the affixes from
the words (i.e. prefixes, suffixes and infixes). It is an improvement
of the previous Arabic light stemmer (ARLSTem). The new version was compared to
the original algorithm and several existing Arabic light stemmers, where the
results showed that the new version considerably improves the under-stemming
errors that are common to light stemmers. Both ARLSTem and ARLSTem2 can be run
online and do not use any dictionary.
"""
import re

from nltk.stem.api import StemmerI


class ARLSTem2(StemmerI):
    """
    Return a stemmed Arabic word after removing affixes. This an improved
    version of the previous algorithm, which reduces under-stemming errors.
    Typically used in Arabic search engine, information retrieval and NLP.

        >>> from nltk.stem import arlstem2
        >>> stemmer = ARLSTem2()
        >>> word = stemmer.stem('يعمل')
        >>> print(word)
        عمل

    :param token: The input Arabic word (unicode) to be stemmed
    :type token: unicode
    :return: A unicode Arabic word
    """

    def __init__(self):
        # different Alif with hamza
        self.re_hamzated_alif = re.compile(r"[\u0622\u0623\u0625]")
        self.re_alifMaqsura = re.compile(r"[\u0649]")
        self.re_diacritics = re.compile(r"[\u064B-\u065F]")

        # Alif Laam, Laam Laam, Fa Laam, Fa Ba
        self.pr2 = ["\u0627\u0644", "\u0644\u0644", "\u0641\u0644", "\u0641\u0628"]
        # Ba Alif Laam, Kaaf Alif Laam, Waaw Alif Laam
        self.pr3 = ["\u0628\u0627\u0644", "\u0643\u0627\u0644", "\u0648\u0627\u0644"]
        # Fa Laam Laam, Waaw Laam Laam
        self.pr32 = ["\u0641\u0644\u0644", "\u0648\u0644\u0644"]
        # Fa Ba Alif Laam, Waaw Ba Alif Laam, Fa Kaaf Alif Laam
        self.pr4 = [
            "\u0641\u0628\u0627\u0644",
            "\u0648\u0628\u0627\u0644",
            "\u0641\u0643\u0627\u0644",
        ]

        # Kaf Yaa, Kaf Miim
        self.su2 = ["\u0643\u064A", "\u0643\u0645"]
        # Ha Alif, Ha Miim
        self.su22 = ["\u0647\u0627", "\u0647\u0645"]
        # Kaf Miim Alif, Kaf Noon Shadda
        self.su3 = ["\u0643\u0645\u0627", "\u0643\u0646\u0651"]
        # Ha Miim Alif, Ha Noon Shadda
        self.su32 = ["\u0647\u0645\u0627", "\u0647\u0646\u0651"]

        # Alif Noon, Ya Noon, Waaw Noon
        self.pl_si2 = ["\u0627\u0646", "\u064A\u0646", "\u0648\u0646"]
        # Taa Alif Noon, Taa Ya Noon
        self.pl_si3 = ["\u062A\u0627\u0646", "\u062A\u064A\u0646"]

        # Alif Noon, Waaw Noon
        self.verb_su2 = ["\u0627\u0646", "\u0648\u0646"]
        # Siin Taa, Siin Yaa
        self.verb_pr2 = ["\u0633\u062A", "\u0633\u064A"]
        # Siin Alif, Siin Noon
        self.verb_pr22 = ["\u0633\u0627", "\u0633\u0646"]
        # Lam Noon, Lam Taa, Lam Yaa, Lam Hamza
        self.verb_pr33 = [
            "\u0644\u0646",
            "\u0644\u062A",
            "\u0644\u064A",
            "\u0644\u0623",
        ]
        # Taa Miim Alif, Taa Noon Shadda
        self.verb_suf3 = ["\u062A\u0645\u0627", "\u062A\u0646\u0651"]
        # Noon Alif, Taa Miim, Taa Alif, Waaw Alif
        self.verb_suf2 = [
            "\u0646\u0627",
            "\u062A\u0645",
            "\u062A\u0627",
            "\u0648\u0627",
        ]
        # Taa, Alif, Noon
        self.verb_suf1 = ["\u062A", "\u0627", "\u0646"]

    def stem1(self, token):
        """
        call this function to get the first stem
        """
        try:
            if token is None:
                raise ValueError(
                    "The word could not be stemmed, because \
                                 it is empty !"
                )
            self.is_verb = False
            # remove Arabic diacritics and replace some letters with others
            token = self.norm(token)
            # strip the common noun prefixes
            pre = self.pref(token)
            if pre is not None:
                token = pre
            # transform the feminine form to masculine form
            fm = self.fem2masc(token)
            if fm is not None:
                return fm
            # strip the adjective affixes
            adj = self.adjective(token)
            if adj is not None:
                return adj
            # strip the suffixes that are common to nouns and verbs
            token = self.suff(token)
            # transform a plural noun to a singular noun
            ps = self.plur2sing(token)
            if ps is None:
                if pre is None:  # if the noun prefixes are not stripped
                    # strip the verb prefixes and suffixes
                    verb = self.verb(token)
                    if verb is not None:
                        self.is_verb = True
                        return verb
            else:
                return ps
            return token
        except ValueError as e:
            print(e)

    def stem(self, token):
        # stem the input word
        try:
            if token is None:
                raise ValueError(
                    "The word could not be stemmed, because \
                                 it is empty !"
                )
            # run the first round of stemming
            token = self.stem1(token)
            # check if there is some additional noun affixes
            if len(token) > 4:
                # ^Taa, $Yaa + char
                if token.startswith("\u062A") and token[-2] == "\u064A":
                    token = token[1:-2] + token[-1]
                    return token
                # ^Miim, $Waaw + char
                if token.startswith("\u0645") and token[-2] == "\u0648":
                    token = token[1:-2] + token[-1]
                    return token
            if len(token) > 3:
                # !^Alif, $Yaa
                if not token.startswith("\u0627") and token.endswith("\u064A"):
                    token = token[:-1]
                    return token
                # $Laam
                if token.startswith("\u0644"):
                    return token[1:]
            return token
        except ValueError as e:
            print(e)

    def norm(self, token):
        """
        normalize the word by removing diacritics, replace hamzated Alif
        with Alif bare, replace AlifMaqsura with Yaa and remove Waaw at the
        beginning.
        """
        # strip Arabic diacritics
        token = self.re_diacritics.sub("", token)
        # replace Hamzated Alif with Alif bare
        token = self.re_hamzated_alif.sub("\u0627", token)
        # replace alifMaqsura with Yaa
        token = self.re_alifMaqsura.sub("\u064A", token)
        # strip the Waaw from the word beginning if the remaining is
        # tri-literal at least
        if token.startswith("\u0648") and len(token) > 3:
            token = token[1:]
        return token

    def pref(self, token):
        """
        remove prefixes from the words' beginning.
        """
        if len(token) > 5:
            for p3 in self.pr3:
                if token.startswith(p3):
                    return token[3:]
        if len(token) > 6:
            for p4 in self.pr4:
                if token.startswith(p4):
                    return token[4:]
        if len(token) > 5:
            for p3 in self.pr32:
                if token.startswith(p3):
                    return token[3:]
        if len(token) > 4:
            for p2 in self.pr2:
                if token.startswith(p2):
                    return token[2:]

    def adjective(self, token):
        """
        remove the infixes from adjectives
        """
        # ^Alif, Alif, $Yaa
        if len(token) > 5:
            if (
                token.startswith("\u0627")
                and token[-3] == "\u0627"
                and token.endswith("\u064A")
            ):
                return token[:-3] + token[-2]

    def suff(self, token):
        """
        remove the suffixes from the word's ending.
        """
        if token.endswith("\u0643") and len(token) > 3:
            return token[:-1]
        if len(token) > 4:
            for s2 in self.su2:
                if token.endswith(s2):
                    return token[:-2]
        if len(token) > 5:
            for s3 in self.su3:
                if token.endswith(s3):
                    return token[:-3]
        if token.endswith("\u0647") and len(token) > 3:
            token = token[:-1]
            return token
        if len(token) > 4:
            for s2 in self.su22:
                if token.endswith(s2):
                    return token[:-2]
        if len(token) > 5:
            for s3 in self.su32:
                if token.endswith(s3):
                    return token[:-3]
        # $Noon and Alif
        if token.endswith("\u0646\u0627") and len(token) > 4:
            return token[:-2]
        return token

    def fem2masc(self, token):
        """
        transform the word from the feminine form to the masculine form.
        """
        if len(token) > 6:
            # ^Taa, Yaa, $Yaa and Taa Marbuta
            if (
                token.startswith("\u062A")
                and token[-4] == "\u064A"
                and token.endswith("\u064A\u0629")
            ):
                return token[1:-4] + token[-3]
            # ^Alif, Yaa, $Yaa and Taa Marbuta
            if (
                token.startswith("\u0627")
                and token[-4] == "\u0627"
                and token.endswith("\u064A\u0629")
            ):
                return token[:-4] + token[-3]
        # $Alif, Yaa and Taa Marbuta
        if token.endswith("\u0627\u064A\u0629") and len(token) > 5:
            return token[:-2]
        if len(token) > 4:
            # Alif, $Taa Marbuta
            if token[1] == "\u0627" and token.endswith("\u0629"):
                return token[0] + token[2:-1]
            # $Yaa and Taa Marbuta
            if token.endswith("\u064A\u0629"):
                return token[:-2]
        # $Taa Marbuta
        if token.endswith("\u0629") and len(token) > 3:
            return token[:-1]

    def plur2sing(self, token):
        """
        transform the word from the plural form to the singular form.
        """
        # ^Haa, $Noon, Waaw
        if len(token) > 5:
            if token.startswith("\u0645") and token.endswith("\u0648\u0646"):
                return token[1:-2]
        if len(token) > 4:
            for ps2 in self.pl_si2:
                if token.endswith(ps2):
                    return token[:-2]
        if len(token) > 5:
            for ps3 in self.pl_si3:
                if token.endswith(ps3):
                    return token[:-3]
        if len(token) > 4:
            # $Alif, Taa
            if token.endswith("\u0627\u062A"):
                return token[:-2]
            # ^Alif Alif
            if token.startswith("\u0627") and token[2] == "\u0627":
                return token[:2] + token[3:]
            # ^Alif Alif
            if token.startswith("\u0627") and token[-2] == "\u0627":
                return token[1:-2] + token[-1]

    def verb(self, token):
        """
        stem the verb prefixes and suffixes or both
        """
        vb = self.verb_t1(token)
        if vb is not None:
            return vb
        vb = self.verb_t2(token)
        if vb is not None:
            return vb
        vb = self.verb_t3(token)
        if vb is not None:
            return vb
        vb = self.verb_t4(token)
        if vb is not None:
            return vb
        vb = self.verb_t5(token)
        if vb is not None:
            return vb
        vb = self.verb_t6(token)
        return vb

    def verb_t1(self, token):
        """
        stem the present tense co-occurred prefixes and suffixes
        """
        if len(token) > 5 and token.startswith("\u062A"):  # Taa
            for s2 in self.pl_si2:
                if token.endswith(s2):
                    return token[1:-2]
        if len(token) > 5 and token.startswith("\u064A"):  # Yaa
            for s2 in self.verb_su2:
                if token.endswith(s2):
                    return token[1:-2]
        if len(token) > 4 and token.startswith("\u0627"):  # Alif
            # Waaw Alif
            if len(token) > 5 and token.endswith("\u0648\u0627"):
                return token[1:-2]
            # Yaa
            if token.endswith("\u064A"):
                return token[1:-1]
            # Alif
            if token.endswith("\u0627"):
                return token[1:-1]
            # Noon
            if token.endswith("\u0646"):
                return token[1:-1]
        # ^Yaa, Noon$
        if len(token) > 4 and token.startswith("\u064A") and token.endswith("\u0646"):
            return token[1:-1]
        # ^Taa, Noon$
        if len(token) > 4 and token.startswith("\u062A") and token.endswith("\u0646"):
            return token[1:-1]

    def verb_t2(self, token):
        """
        stem the future tense co-occurred prefixes and suffixes
        """
        if len(token) > 6:
            for s2 in self.pl_si2:
                # ^Siin Taa
                if token.startswith(self.verb_pr2[0]) and token.endswith(s2):
                    return token[2:-2]
            # ^Siin Yaa, Alif Noon$
            if token.startswith(self.verb_pr2[1]) and token.endswith(self.pl_si2[0]):
                return token[2:-2]
            # ^Siin Yaa, Waaw Noon$
            if token.startswith(self.verb_pr2[1]) and token.endswith(self.pl_si2[2]):
                return token[2:-2]
        # ^Siin Taa, Noon$
        if (
            len(token) > 5
            and token.startswith(self.verb_pr2[0])
            and token.endswith("\u0646")
        ):
            return token[2:-1]
        # ^Siin Yaa, Noon$
        if (
            len(token) > 5
            and token.startswith(self.verb_pr2[1])
            and token.endswith("\u0646")
        ):
            return token[2:-1]

    def verb_t3(self, token):
        """
        stem the present tense suffixes
        """
        if len(token) > 5:
            for su3 in self.verb_suf3:
                if token.endswith(su3):
                    return token[:-3]
        if len(token) > 4:
            for su2 in self.verb_suf2:
                if token.endswith(su2):
                    return token[:-2]
        if len(token) > 3:
            for su1 in self.verb_suf1:
                if token.endswith(su1):
                    return token[:-1]

    def verb_t4(self, token):
        """
        stem the present tense prefixes
        """
        if len(token) > 3:
            for pr1 in self.verb_suf1:
                if token.startswith(pr1):
                    return token[1:]
            if token.startswith("\u064A"):
                return token[1:]

    def verb_t5(self, token):
        """
        stem the future tense prefixes
        """
        if len(token) > 4:
            for pr2 in self.verb_pr22:
                if token.startswith(pr2):
                    return token[2:]
            for pr2 in self.verb_pr2:
                if token.startswith(pr2):
                    return token[2:]

    def verb_t6(self, token):
        """
        stem the imperative tense prefixes
        """
        if len(token) > 4:
            for pr3 in self.verb_pr33:
                if token.startswith(pr3):
                    return token[2:]

        return token
