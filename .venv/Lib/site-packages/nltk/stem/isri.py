#
# Natural Language Toolkit: The ISRI Arabic Stemmer
#
# Copyright (C) 2001-2023 NLTK Project
# Algorithm: Kazem Taghva, Rania Elkhoury, and Jeffrey Coombs (2005)
# Author: Hosam Algasaier <hosam_hme@yahoo.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
ISRI Arabic Stemmer

The algorithm for this stemmer is described in:

Taghva, K., Elkoury, R., and Coombs, J. 2005. Arabic Stemming without a root dictionary.
Information Science Research Institute. University of Nevada, Las Vegas, USA.

The Information Science Research Institute’s (ISRI) Arabic stemmer shares many features
with the Khoja stemmer. However, the main difference is that ISRI stemmer does not use root
dictionary. Also, if a root is not found, ISRI stemmer returned normalized form, rather than
returning the original unmodified word.

Additional adjustments were made to improve the algorithm:

1- Adding 60 stop words.
2- Adding the pattern (تفاعيل) to ISRI pattern set.
3- The step 2 in the original algorithm was normalizing all hamza. This step is discarded because it
increases the word ambiguities and changes the original root.

"""
import re

from nltk.stem.api import StemmerI


class ISRIStemmer(StemmerI):
    """
    ISRI Arabic stemmer based on algorithm: Arabic Stemming without a root dictionary.
    Information Science Research Institute. University of Nevada, Las Vegas, USA.

    A few minor modifications have been made to ISRI basic algorithm.
    See the source code of this module for more information.

    isri.stem(token) returns Arabic root for the given token.

    The ISRI Stemmer requires that all tokens have Unicode string types.
    If you use Python IDLE on Arabic Windows you have to decode text first
    using Arabic '1256' coding.
    """

    def __init__(self):
        # length three prefixes
        self.p3 = [
            "\u0643\u0627\u0644",
            "\u0628\u0627\u0644",
            "\u0648\u0644\u0644",
            "\u0648\u0627\u0644",
        ]

        # length two prefixes
        self.p2 = ["\u0627\u0644", "\u0644\u0644"]

        # length one prefixes
        self.p1 = [
            "\u0644",
            "\u0628",
            "\u0641",
            "\u0633",
            "\u0648",
            "\u064a",
            "\u062a",
            "\u0646",
            "\u0627",
        ]

        # length three suffixes
        self.s3 = [
            "\u062a\u0645\u0644",
            "\u0647\u0645\u0644",
            "\u062a\u0627\u0646",
            "\u062a\u064a\u0646",
            "\u0643\u0645\u0644",
        ]

        # length two suffixes
        self.s2 = [
            "\u0648\u0646",
            "\u0627\u062a",
            "\u0627\u0646",
            "\u064a\u0646",
            "\u062a\u0646",
            "\u0643\u0645",
            "\u0647\u0646",
            "\u0646\u0627",
            "\u064a\u0627",
            "\u0647\u0627",
            "\u062a\u0645",
            "\u0643\u0646",
            "\u0646\u064a",
            "\u0648\u0627",
            "\u0645\u0627",
            "\u0647\u0645",
        ]

        # length one suffixes
        self.s1 = ["\u0629", "\u0647", "\u064a", "\u0643", "\u062a", "\u0627", "\u0646"]

        # groups of length four patterns
        self.pr4 = {
            0: ["\u0645"],
            1: ["\u0627"],
            2: ["\u0627", "\u0648", "\u064A"],
            3: ["\u0629"],
        }

        # Groups of length five patterns and length three roots
        self.pr53 = {
            0: ["\u0627", "\u062a"],
            1: ["\u0627", "\u064a", "\u0648"],
            2: ["\u0627", "\u062a", "\u0645"],
            3: ["\u0645", "\u064a", "\u062a"],
            4: ["\u0645", "\u062a"],
            5: ["\u0627", "\u0648"],
            6: ["\u0627", "\u0645"],
        }

        self.re_short_vowels = re.compile(r"[\u064B-\u0652]")
        self.re_hamza = re.compile(r"[\u0621\u0624\u0626]")
        self.re_initial_hamza = re.compile(r"^[\u0622\u0623\u0625]")

        self.stop_words = [
            "\u064a\u0643\u0648\u0646",
            "\u0648\u0644\u064a\u0633",
            "\u0648\u0643\u0627\u0646",
            "\u0643\u0630\u0644\u0643",
            "\u0627\u0644\u062a\u064a",
            "\u0648\u0628\u064a\u0646",
            "\u0639\u0644\u064a\u0647\u0627",
            "\u0645\u0633\u0627\u0621",
            "\u0627\u0644\u0630\u064a",
            "\u0648\u0643\u0627\u0646\u062a",
            "\u0648\u0644\u0643\u0646",
            "\u0648\u0627\u0644\u062a\u064a",
            "\u062a\u0643\u0648\u0646",
            "\u0627\u0644\u064a\u0648\u0645",
            "\u0627\u0644\u0644\u0630\u064a\u0646",
            "\u0639\u0644\u064a\u0647",
            "\u0643\u0627\u0646\u062a",
            "\u0644\u0630\u0644\u0643",
            "\u0623\u0645\u0627\u0645",
            "\u0647\u0646\u0627\u0643",
            "\u0645\u0646\u0647\u0627",
            "\u0645\u0627\u0632\u0627\u0644",
            "\u0644\u0627\u0632\u0627\u0644",
            "\u0644\u0627\u064a\u0632\u0627\u0644",
            "\u0645\u0627\u064a\u0632\u0627\u0644",
            "\u0627\u0635\u0628\u062d",
            "\u0623\u0635\u0628\u062d",
            "\u0623\u0645\u0633\u0649",
            "\u0627\u0645\u0633\u0649",
            "\u0623\u0636\u062d\u0649",
            "\u0627\u0636\u062d\u0649",
            "\u0645\u0627\u0628\u0631\u062d",
            "\u0645\u0627\u0641\u062a\u0626",
            "\u0645\u0627\u0627\u0646\u0641\u0643",
            "\u0644\u0627\u0633\u064a\u0645\u0627",
            "\u0648\u0644\u0627\u064a\u0632\u0627\u0644",
            "\u0627\u0644\u062d\u0627\u0644\u064a",
            "\u0627\u0644\u064a\u0647\u0627",
            "\u0627\u0644\u0630\u064a\u0646",
            "\u0641\u0627\u0646\u0647",
            "\u0648\u0627\u0644\u0630\u064a",
            "\u0648\u0647\u0630\u0627",
            "\u0644\u0647\u0630\u0627",
            "\u0641\u0643\u0627\u0646",
            "\u0633\u062a\u0643\u0648\u0646",
            "\u0627\u0644\u064a\u0647",
            "\u064a\u0645\u0643\u0646",
            "\u0628\u0647\u0630\u0627",
            "\u0627\u0644\u0630\u0649",
        ]

    def stem(self, token):
        """
        Stemming a word token using the ISRI stemmer.
        """
        token = self.norm(
            token, 1
        )  # remove diacritics which representing Arabic short vowels
        if token in self.stop_words:
            return token  # exclude stop words from being processed
        token = self.pre32(
            token
        )  # remove length three and length two prefixes in this order
        token = self.suf32(
            token
        )  # remove length three and length two suffixes in this order
        token = self.waw(
            token
        )  # remove connective ‘و’ if it precedes a word beginning with ‘و’
        token = self.norm(token, 2)  # normalize initial hamza to bare alif
        # if 4 <= word length <= 7, then stem; otherwise, no stemming
        if len(token) == 4:  # length 4 word
            token = self.pro_w4(token)
        elif len(token) == 5:  # length 5 word
            token = self.pro_w53(token)
            token = self.end_w5(token)
        elif len(token) == 6:  # length 6 word
            token = self.pro_w6(token)
            token = self.end_w6(token)
        elif len(token) == 7:  # length 7 word
            token = self.suf1(token)
            if len(token) == 7:
                token = self.pre1(token)
            if len(token) == 6:
                token = self.pro_w6(token)
                token = self.end_w6(token)
        return token

    def norm(self, word, num=3):
        """
        normalization:
        num=1  normalize diacritics
        num=2  normalize initial hamza
        num=3  both 1&2
        """
        if num == 1:
            word = self.re_short_vowels.sub("", word)
        elif num == 2:
            word = self.re_initial_hamza.sub("\u0627", word)
        elif num == 3:
            word = self.re_short_vowels.sub("", word)
            word = self.re_initial_hamza.sub("\u0627", word)
        return word

    def pre32(self, word):
        """remove length three and length two prefixes in this order"""
        if len(word) >= 6:
            for pre3 in self.p3:
                if word.startswith(pre3):
                    return word[3:]
        if len(word) >= 5:
            for pre2 in self.p2:
                if word.startswith(pre2):
                    return word[2:]
        return word

    def suf32(self, word):
        """remove length three and length two suffixes in this order"""
        if len(word) >= 6:
            for suf3 in self.s3:
                if word.endswith(suf3):
                    return word[:-3]
        if len(word) >= 5:
            for suf2 in self.s2:
                if word.endswith(suf2):
                    return word[:-2]
        return word

    def waw(self, word):
        """remove connective ‘و’ if it precedes a word beginning with ‘و’"""
        if len(word) >= 4 and word[:2] == "\u0648\u0648":
            word = word[1:]
        return word

    def pro_w4(self, word):
        """process length four patterns and extract length three roots"""
        if word[0] in self.pr4[0]:  # مفعل
            word = word[1:]
        elif word[1] in self.pr4[1]:  # فاعل
            word = word[:1] + word[2:]
        elif word[2] in self.pr4[2]:  # فعال - فعول - فعيل
            word = word[:2] + word[3]
        elif word[3] in self.pr4[3]:  # فعلة
            word = word[:-1]
        else:
            word = self.suf1(word)  # do - normalize short sufix
            if len(word) == 4:
                word = self.pre1(word)  # do - normalize short prefix
        return word

    def pro_w53(self, word):
        """process length five patterns and extract length three roots"""
        if word[2] in self.pr53[0] and word[0] == "\u0627":  # افتعل - افاعل
            word = word[1] + word[3:]
        elif word[3] in self.pr53[1] and word[0] == "\u0645":  # مفعول - مفعال - مفعيل
            word = word[1:3] + word[4]
        elif word[0] in self.pr53[2] and word[4] == "\u0629":  # مفعلة - تفعلة - افعلة
            word = word[1:4]
        elif word[0] in self.pr53[3] and word[2] == "\u062a":  # مفتعل - يفتعل - تفتعل
            word = word[1] + word[3:]
        elif word[0] in self.pr53[4] and word[2] == "\u0627":  # مفاعل - تفاعل
            word = word[1] + word[3:]
        elif word[2] in self.pr53[5] and word[4] == "\u0629":  # فعولة - فعالة
            word = word[:2] + word[3]
        elif word[0] in self.pr53[6] and word[1] == "\u0646":  # انفعل - منفعل
            word = word[2:]
        elif word[3] == "\u0627" and word[0] == "\u0627":  # افعال
            word = word[1:3] + word[4]
        elif word[4] == "\u0646" and word[3] == "\u0627":  # فعلان
            word = word[:3]
        elif word[3] == "\u064a" and word[0] == "\u062a":  # تفعيل
            word = word[1:3] + word[4]
        elif word[3] == "\u0648" and word[1] == "\u0627":  # فاعول
            word = word[0] + word[2] + word[4]
        elif word[2] == "\u0627" and word[1] == "\u0648":  # فواعل
            word = word[0] + word[3:]
        elif word[3] == "\u0626" and word[2] == "\u0627":  # فعائل
            word = word[:2] + word[4]
        elif word[4] == "\u0629" and word[1] == "\u0627":  # فاعلة
            word = word[0] + word[2:4]
        elif word[4] == "\u064a" and word[2] == "\u0627":  # فعالي
            word = word[:2] + word[3]
        else:
            word = self.suf1(word)  # do - normalize short sufix
            if len(word) == 5:
                word = self.pre1(word)  # do - normalize short prefix
        return word

    def pro_w54(self, word):
        """process length five patterns and extract length four roots"""
        if word[0] in self.pr53[2]:  # تفعلل - افعلل - مفعلل
            word = word[1:]
        elif word[4] == "\u0629":  # فعللة
            word = word[:4]
        elif word[2] == "\u0627":  # فعالل
            word = word[:2] + word[3:]
        return word

    def end_w5(self, word):
        """ending step (word of length five)"""
        if len(word) == 4:
            word = self.pro_w4(word)
        elif len(word) == 5:
            word = self.pro_w54(word)
        return word

    def pro_w6(self, word):
        """process length six patterns and extract length three roots"""
        if word.startswith("\u0627\u0633\u062a") or word.startswith(
            "\u0645\u0633\u062a"
        ):  # مستفعل - استفعل
            word = word[3:]
        elif (
            word[0] == "\u0645" and word[3] == "\u0627" and word[5] == "\u0629"
        ):  # مفعالة
            word = word[1:3] + word[4]
        elif (
            word[0] == "\u0627" and word[2] == "\u062a" and word[4] == "\u0627"
        ):  # افتعال
            word = word[1] + word[3] + word[5]
        elif (
            word[0] == "\u0627" and word[3] == "\u0648" and word[2] == word[4]
        ):  # افعوعل
            word = word[1] + word[4:]
        elif (
            word[0] == "\u062a" and word[2] == "\u0627" and word[4] == "\u064a"
        ):  # تفاعيل   new pattern
            word = word[1] + word[3] + word[5]
        else:
            word = self.suf1(word)  # do - normalize short sufix
            if len(word) == 6:
                word = self.pre1(word)  # do - normalize short prefix
        return word

    def pro_w64(self, word):
        """process length six patterns and extract length four roots"""
        if word[0] == "\u0627" and word[4] == "\u0627":  # افعلال
            word = word[1:4] + word[5]
        elif word.startswith("\u0645\u062a"):  # متفعلل
            word = word[2:]
        return word

    def end_w6(self, word):
        """ending step (word of length six)"""
        if len(word) == 5:
            word = self.pro_w53(word)
            word = self.end_w5(word)
        elif len(word) == 6:
            word = self.pro_w64(word)
        return word

    def suf1(self, word):
        """normalize short sufix"""
        for sf1 in self.s1:
            if word.endswith(sf1):
                return word[:-1]
        return word

    def pre1(self, word):
        """normalize short prefix"""
        for sp1 in self.p1:
            if word.startswith(sp1):
                return word[1:]
        return word
