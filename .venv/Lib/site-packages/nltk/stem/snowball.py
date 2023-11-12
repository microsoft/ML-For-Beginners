#
# Natural Language Toolkit: Snowball Stemmer
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Peter Michael Stahl <pemistahl@gmail.com>
#         Peter Ljunglof <peter.ljunglof@heatherleaf.se> (revisions)
#         Lakhdar Benzahia <lakhdar.benzahia@gmail.com>  (co-writer)
#         Assem Chelli <assem.ch@gmail.com>  (reviewer arabicstemmer)
#         Abdelkrim Aries <ab_aries@esi.dz> (reviewer arabicstemmer)
# Algorithms: Dr Martin Porter <martin@tartarus.org>
#             Assem Chelli <assem.ch@gmail.com>  arabic stemming algorithm
#             Benzahia Lakhdar <lakhdar.benzahia@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Snowball stemmers

This module provides a port of the Snowball stemmers
developed by Martin Porter.

There is also a demo function: `snowball.demo()`.

"""

import re

from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace


class SnowballStemmer(StemmerI):

    """
    Snowball Stemmer

    The following languages are supported:
    Arabic, Danish, Dutch, English, Finnish, French, German,
    Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian,
    Spanish and Swedish.

    The algorithm for English is documented here:

        Porter, M. \"An algorithm for suffix stripping.\"
        Program 14.3 (1980): 130-137.

    The algorithms have been developed by Martin Porter.
    These stemmers are called Snowball, because Porter created
    a programming language with this name for creating
    new stemming algorithms. There is more information available
    at http://snowball.tartarus.org/

    The stemmer is invoked as shown below:

    >>> from nltk.stem import SnowballStemmer # See which languages are supported
    >>> print(" ".join(SnowballStemmer.languages)) # doctest: +NORMALIZE_WHITESPACE
    arabic danish dutch english finnish french german hungarian
    italian norwegian porter portuguese romanian russian
    spanish swedish
    >>> stemmer = SnowballStemmer("german") # Choose a language
    >>> stemmer.stem("Autobahnen") # Stem a word
    'autobahn'

    Invoking the stemmers that way is useful if you do not know the
    language to be stemmed at runtime. Alternatively, if you already know
    the language, then you can invoke the language specific stemmer directly:

    >>> from nltk.stem.snowball import GermanStemmer
    >>> stemmer = GermanStemmer()
    >>> stemmer.stem("Autobahnen")
    'autobahn'

    :param language: The language whose subclass is instantiated.
    :type language: str or unicode
    :param ignore_stopwords: If set to True, stopwords are
                             not stemmed and returned unchanged.
                             Set to False by default.
    :type ignore_stopwords: bool
    :raise ValueError: If there is no stemmer for the specified
                           language, a ValueError is raised.
    """

    languages = (
        "arabic",
        "danish",
        "dutch",
        "english",
        "finnish",
        "french",
        "german",
        "hungarian",
        "italian",
        "norwegian",
        "porter",
        "portuguese",
        "romanian",
        "russian",
        "spanish",
        "swedish",
    )

    def __init__(self, language, ignore_stopwords=False):
        if language not in self.languages:
            raise ValueError(f"The language '{language}' is not supported.")
        stemmerclass = globals()[language.capitalize() + "Stemmer"]
        self.stemmer = stemmerclass(ignore_stopwords)
        self.stem = self.stemmer.stem
        self.stopwords = self.stemmer.stopwords

    def stem(self, token):
        return self.stemmer.stem(self, token)


class _LanguageSpecificStemmer(StemmerI):

    """
    This helper subclass offers the possibility
    to invoke a specific stemmer directly.
    This is useful if you already know the language to be stemmed at runtime.

    Create an instance of the Snowball stemmer.

    :param ignore_stopwords: If set to True, stopwords are
                             not stemmed and returned unchanged.
                             Set to False by default.
    :type ignore_stopwords: bool
    """

    def __init__(self, ignore_stopwords=False):
        # The language is the name of the class, minus the final "Stemmer".
        language = type(self).__name__.lower()
        if language.endswith("stemmer"):
            language = language[:-7]

        self.stopwords = set()
        if ignore_stopwords:
            try:
                for word in stopwords.words(language):
                    self.stopwords.add(word)
            except OSError as e:
                raise ValueError(
                    "{!r} has no list of stopwords. Please set"
                    " 'ignore_stopwords' to 'False'.".format(self)
                ) from e

    def __repr__(self):
        """
        Print out the string representation of the respective class.

        """
        return f"<{type(self).__name__}>"


class PorterStemmer(_LanguageSpecificStemmer, porter.PorterStemmer):
    """
    A word stemmer based on the original Porter stemming algorithm.

        Porter, M. \"An algorithm for suffix stripping.\"
        Program 14.3 (1980): 130-137.

    A few minor modifications have been made to Porter's basic
    algorithm.  See the source code of the module
    nltk.stem.porter for more information.

    """

    def __init__(self, ignore_stopwords=False):
        _LanguageSpecificStemmer.__init__(self, ignore_stopwords)
        porter.PorterStemmer.__init__(self)


class _ScandinavianStemmer(_LanguageSpecificStemmer):

    """
    This subclass encapsulates a method for defining the string region R1.
    It is used by the Danish, Norwegian, and Swedish stemmer.

    """

    def _r1_scandinavian(self, word, vowels):
        """
        Return the region R1 that is used by the Scandinavian stemmers.

        R1 is the region after the first non-vowel following a vowel,
        or is the null region at the end of the word if there is no
        such non-vowel. But then R1 is adjusted so that the region
        before it contains at least three letters.

        :param word: The word whose region R1 is determined.
        :type word: str or unicode
        :param vowels: The vowels of the respective language that are
                       used to determine the region R1.
        :type vowels: unicode
        :return: the region R1 for the respective word.
        :rtype: unicode
        :note: This helper method is invoked by the respective stem method of
               the subclasses DanishStemmer, NorwegianStemmer, and
               SwedishStemmer. It is not to be invoked directly!

        """
        r1 = ""
        for i in range(1, len(word)):
            if word[i] not in vowels and word[i - 1] in vowels:
                if 3 > len(word[: i + 1]) > 0:
                    r1 = word[3:]
                elif len(word[: i + 1]) >= 3:
                    r1 = word[i + 1 :]
                else:
                    return word
                break

        return r1


class _StandardStemmer(_LanguageSpecificStemmer):

    """
    This subclass encapsulates two methods for defining the standard versions
    of the string regions R1, R2, and RV.

    """

    def _r1r2_standard(self, word, vowels):
        """
        Return the standard interpretations of the string regions R1 and R2.

        R1 is the region after the first non-vowel following a vowel,
        or is the null region at the end of the word if there is no
        such non-vowel.

        R2 is the region after the first non-vowel following a vowel
        in R1, or is the null region at the end of the word if there
        is no such non-vowel.

        :param word: The word whose regions R1 and R2 are determined.
        :type word: str or unicode
        :param vowels: The vowels of the respective language that are
                       used to determine the regions R1 and R2.
        :type vowels: unicode
        :return: (r1,r2), the regions R1 and R2 for the respective word.
        :rtype: tuple
        :note: This helper method is invoked by the respective stem method of
               the subclasses DutchStemmer, FinnishStemmer,
               FrenchStemmer, GermanStemmer, ItalianStemmer,
               PortugueseStemmer, RomanianStemmer, and SpanishStemmer.
               It is not to be invoked directly!
        :note: A detailed description of how to define R1 and R2
               can be found at http://snowball.tartarus.org/texts/r1r2.html

        """
        r1 = ""
        r2 = ""
        for i in range(1, len(word)):
            if word[i] not in vowels and word[i - 1] in vowels:
                r1 = word[i + 1 :]
                break

        for i in range(1, len(r1)):
            if r1[i] not in vowels and r1[i - 1] in vowels:
                r2 = r1[i + 1 :]
                break

        return (r1, r2)

    def _rv_standard(self, word, vowels):
        """
        Return the standard interpretation of the string region RV.

        If the second letter is a consonant, RV is the region after the
        next following vowel. If the first two letters are vowels, RV is
        the region after the next following consonant. Otherwise, RV is
        the region after the third letter.

        :param word: The word whose region RV is determined.
        :type word: str or unicode
        :param vowels: The vowels of the respective language that are
                       used to determine the region RV.
        :type vowels: unicode
        :return: the region RV for the respective word.
        :rtype: unicode
        :note: This helper method is invoked by the respective stem method of
               the subclasses ItalianStemmer, PortugueseStemmer,
               RomanianStemmer, and SpanishStemmer. It is not to be
               invoked directly!

        """
        rv = ""
        if len(word) >= 2:
            if word[1] not in vowels:
                for i in range(2, len(word)):
                    if word[i] in vowels:
                        rv = word[i + 1 :]
                        break

            elif word[0] in vowels and word[1] in vowels:
                for i in range(2, len(word)):
                    if word[i] not in vowels:
                        rv = word[i + 1 :]
                        break
            else:
                rv = word[3:]

        return rv


class ArabicStemmer(_StandardStemmer):
    """
    https://github.com/snowballstem/snowball/blob/master/algorithms/arabic/stem_Unicode.sbl (Original Algorithm)
    The Snowball Arabic light Stemmer
    Algorithm:

    - Assem Chelli
    - Abdelkrim Aries
    - Lakhdar Benzahia

    NLTK Version Author:

    - Lakhdar Benzahia
    """

    # Normalize_pre stes
    __vocalization = re.compile(
        r"[\u064b-\u064c-\u064d-\u064e-\u064f-\u0650-\u0651-\u0652]"
    )  # ً، ٌ، ٍ، َ، ُ، ِ، ّ، ْ

    __kasheeda = re.compile(r"[\u0640]")  # ـ tatweel/kasheeda

    __arabic_punctuation_marks = re.compile(r"[\u060C-\u061B-\u061F]")  #  ؛ ، ؟

    # Normalize_post
    __last_hamzat = ("\u0623", "\u0625", "\u0622", "\u0624", "\u0626")  # أ، إ، آ، ؤ، ئ

    # normalize other hamza's
    __initial_hamzat = re.compile(r"^[\u0622\u0623\u0625]")  #  أ، إ، آ

    __waw_hamza = re.compile(r"[\u0624]")  # ؤ

    __yeh_hamza = re.compile(r"[\u0626]")  # ئ

    __alefat = re.compile(r"[\u0623\u0622\u0625]")  #  أ، إ، آ

    # Checks
    __checks1 = (
        "\u0643\u0627\u0644",
        "\u0628\u0627\u0644",  # بال، كال
        "\u0627\u0644",
        "\u0644\u0644",  # لل، ال
    )

    __checks2 = ("\u0629", "\u0627\u062a")  # ة  #  female plural ات

    # Suffixes
    __suffix_noun_step1a = (
        "\u064a",
        "\u0643",
        "\u0647",  # ي، ك، ه
        "\u0646\u0627",
        "\u0643\u0645",
        "\u0647\u0627",
        "\u0647\u0646",
        "\u0647\u0645",  # نا، كم، ها، هن، هم
        "\u0643\u0645\u0627",
        "\u0647\u0645\u0627",  # كما، هما
    )

    __suffix_noun_step1b = "\u0646"  # ن

    __suffix_noun_step2a = ("\u0627", "\u064a", "\u0648")  # ا، ي، و

    __suffix_noun_step2b = "\u0627\u062a"  # ات

    __suffix_noun_step2c1 = "\u062a"  # ت

    __suffix_noun_step2c2 = "\u0629"  # ة

    __suffix_noun_step3 = "\u064a"  # ي

    __suffix_verb_step1 = (
        "\u0647",
        "\u0643",  # ه، ك
        "\u0646\u064a",
        "\u0646\u0627",
        "\u0647\u0627",
        "\u0647\u0645",  # ني، نا، ها، هم
        "\u0647\u0646",
        "\u0643\u0645",
        "\u0643\u0646",  # هن، كم، كن
        "\u0647\u0645\u0627",
        "\u0643\u0645\u0627",
        "\u0643\u0645\u0648",  # هما، كما، كمو
    )

    __suffix_verb_step2a = (
        "\u062a",
        "\u0627",
        "\u0646",
        "\u064a",  # ت، ا، ن، ي
        "\u0646\u0627",
        "\u062a\u0627",
        "\u062a\u0646",  # نا، تا، تن Past
        "\u0627\u0646",
        "\u0648\u0646",
        "\u064a\u0646",  # ان، هن، ين Present
        "\u062a\u0645\u0627",  # تما
    )

    __suffix_verb_step2b = ("\u0648\u0627", "\u062a\u0645")  # وا، تم

    __suffix_verb_step2c = ("\u0648", "\u062a\u0645\u0648")  # و  # تمو

    __suffix_all_alef_maqsura = "\u0649"  # ى

    # Prefixes
    __prefix_step1 = (
        "\u0623",  # أ
        "\u0623\u0623",
        "\u0623\u0622",
        "\u0623\u0624",
        "\u0623\u0627",
        "\u0623\u0625",  # أأ، أآ، أؤ، أا، أإ
    )

    __prefix_step2a = ("\u0641\u0627\u0644", "\u0648\u0627\u0644")  # فال، وال

    __prefix_step2b = ("\u0641", "\u0648")  # ف، و

    __prefix_step3a_noun = (
        "\u0627\u0644",
        "\u0644\u0644",  # لل، ال
        "\u0643\u0627\u0644",
        "\u0628\u0627\u0644",  # بال، كال
    )

    __prefix_step3b_noun = (
        "\u0628",
        "\u0643",
        "\u0644",  # ب، ك، ل
        "\u0628\u0628",
        "\u0643\u0643",  # بب، كك
    )

    __prefix_step3_verb = (
        "\u0633\u064a",
        "\u0633\u062a",
        "\u0633\u0646",
        "\u0633\u0623",
    )  # سي، ست، سن، سأ

    __prefix_step4_verb = (
        "\u064a\u0633\u062a",
        "\u0646\u0633\u062a",
        "\u062a\u0633\u062a",
    )  # يست، نست، تست

    # Suffixes added due to Conjugation Verbs
    __conjugation_suffix_verb_1 = ("\u0647", "\u0643")  # ه، ك

    __conjugation_suffix_verb_2 = (
        "\u0646\u064a",
        "\u0646\u0627",
        "\u0647\u0627",  # ني، نا، ها
        "\u0647\u0645",
        "\u0647\u0646",
        "\u0643\u0645",  # هم، هن، كم
        "\u0643\u0646",  # كن
    )
    __conjugation_suffix_verb_3 = (
        "\u0647\u0645\u0627",
        "\u0643\u0645\u0627",
        "\u0643\u0645\u0648",
    )  # هما، كما، كمو

    __conjugation_suffix_verb_4 = ("\u0627", "\u0646", "\u064a")  # ا، ن، ي

    __conjugation_suffix_verb_past = (
        "\u0646\u0627",
        "\u062a\u0627",
        "\u062a\u0646",
    )  # نا، تا، تن

    __conjugation_suffix_verb_present = (
        "\u0627\u0646",
        "\u0648\u0646",
        "\u064a\u0646",
    )  # ان، ون، ين

    # Suffixes added due to derivation Names
    __conjugation_suffix_noun_1 = ("\u064a", "\u0643", "\u0647")  # ي، ك، ه

    __conjugation_suffix_noun_2 = (
        "\u0646\u0627",
        "\u0643\u0645",  # نا، كم
        "\u0647\u0627",
        "\u0647\u0646",
        "\u0647\u0645",  # ها، هن، هم
    )

    __conjugation_suffix_noun_3 = (
        "\u0643\u0645\u0627",
        "\u0647\u0645\u0627",
    )  # كما، هما

    # Prefixes added due to derivation Names
    __prefixes1 = ("\u0648\u0627", "\u0641\u0627")  # فا، وا

    __articles_3len = ("\u0643\u0627\u0644", "\u0628\u0627\u0644")  # بال كال

    __articles_2len = ("\u0627\u0644", "\u0644\u0644")  # ال لل

    # Prepositions letters
    __prepositions1 = ("\u0643", "\u0644")  # ك، ل
    __prepositions2 = ("\u0628\u0628", "\u0643\u0643")  # بب، كك

    is_verb = True
    is_noun = True
    is_defined = False

    suffixes_verb_step1_success = False
    suffix_verb_step2a_success = False
    suffix_verb_step2b_success = False
    suffix_noun_step2c2_success = False
    suffix_noun_step1a_success = False
    suffix_noun_step2a_success = False
    suffix_noun_step2b_success = False
    suffixe_noun_step1b_success = False
    prefix_step2a_success = False
    prefix_step3a_noun_success = False
    prefix_step3b_noun_success = False

    def __normalize_pre(self, token):
        """
        :param token: string
        :return: normalized token type string
        """
        # strip diacritics
        token = self.__vocalization.sub("", token)
        # strip kasheeda
        token = self.__kasheeda.sub("", token)
        # strip punctuation marks
        token = self.__arabic_punctuation_marks.sub("", token)
        return token

    def __normalize_post(self, token):
        # normalize last hamza
        for hamza in self.__last_hamzat:
            if token.endswith(hamza):
                token = suffix_replace(token, hamza, "\u0621")
                break
        # normalize other hamzat
        token = self.__initial_hamzat.sub("\u0627", token)
        token = self.__waw_hamza.sub("\u0648", token)
        token = self.__yeh_hamza.sub("\u064a", token)
        token = self.__alefat.sub("\u0627", token)
        return token

    def __checks_1(self, token):
        for prefix in self.__checks1:
            if token.startswith(prefix):
                if prefix in self.__articles_3len and len(token) > 4:
                    self.is_noun = True
                    self.is_verb = False
                    self.is_defined = True
                    break

                if prefix in self.__articles_2len and len(token) > 3:
                    self.is_noun = True
                    self.is_verb = False
                    self.is_defined = True
                    break

    def __checks_2(self, token):
        for suffix in self.__checks2:
            if token.endswith(suffix):
                if suffix == "\u0629" and len(token) > 2:
                    self.is_noun = True
                    self.is_verb = False
                    break

                if suffix == "\u0627\u062a" and len(token) > 3:
                    self.is_noun = True
                    self.is_verb = False
                    break

    def __Suffix_Verb_Step1(self, token):
        for suffix in self.__suffix_verb_step1:
            if token.endswith(suffix):
                if suffix in self.__conjugation_suffix_verb_1 and len(token) >= 4:
                    token = token[:-1]
                    self.suffixes_verb_step1_success = True
                    break

                if suffix in self.__conjugation_suffix_verb_2 and len(token) >= 5:
                    token = token[:-2]
                    self.suffixes_verb_step1_success = True
                    break

                if suffix in self.__conjugation_suffix_verb_3 and len(token) >= 6:
                    token = token[:-3]
                    self.suffixes_verb_step1_success = True
                    break
        return token

    def __Suffix_Verb_Step2a(self, token):
        for suffix in self.__suffix_verb_step2a:
            if token.endswith(suffix) and len(token) > 3:
                if suffix == "\u062a" and len(token) >= 4:
                    token = token[:-1]
                    self.suffix_verb_step2a_success = True
                    break

                if suffix in self.__conjugation_suffix_verb_4 and len(token) >= 4:
                    token = token[:-1]
                    self.suffix_verb_step2a_success = True
                    break

                if suffix in self.__conjugation_suffix_verb_past and len(token) >= 5:
                    token = token[:-2]  # past
                    self.suffix_verb_step2a_success = True
                    break

                if suffix in self.__conjugation_suffix_verb_present and len(token) > 5:
                    token = token[:-2]  # present
                    self.suffix_verb_step2a_success = True
                    break

                if suffix == "\u062a\u0645\u0627" and len(token) >= 6:
                    token = token[:-3]
                    self.suffix_verb_step2a_success = True
                    break
        return token

    def __Suffix_Verb_Step2c(self, token):
        for suffix in self.__suffix_verb_step2c:
            if token.endswith(suffix):
                if suffix == "\u062a\u0645\u0648" and len(token) >= 6:
                    token = token[:-3]
                    break

                if suffix == "\u0648" and len(token) >= 4:
                    token = token[:-1]
                    break
        return token

    def __Suffix_Verb_Step2b(self, token):
        for suffix in self.__suffix_verb_step2b:
            if token.endswith(suffix) and len(token) >= 5:
                token = token[:-2]
                self.suffix_verb_step2b_success = True
                break
        return token

    def __Suffix_Noun_Step2c2(self, token):
        for suffix in self.__suffix_noun_step2c2:
            if token.endswith(suffix) and len(token) >= 3:
                token = token[:-1]
                self.suffix_noun_step2c2_success = True
                break
        return token

    def __Suffix_Noun_Step1a(self, token):
        for suffix in self.__suffix_noun_step1a:
            if token.endswith(suffix):
                if suffix in self.__conjugation_suffix_noun_1 and len(token) >= 4:
                    token = token[:-1]
                    self.suffix_noun_step1a_success = True
                    break

                if suffix in self.__conjugation_suffix_noun_2 and len(token) >= 5:
                    token = token[:-2]
                    self.suffix_noun_step1a_success = True
                    break

                if suffix in self.__conjugation_suffix_noun_3 and len(token) >= 6:
                    token = token[:-3]
                    self.suffix_noun_step1a_success = True
                    break
        return token

    def __Suffix_Noun_Step2a(self, token):
        for suffix in self.__suffix_noun_step2a:
            if token.endswith(suffix) and len(token) > 4:
                token = token[:-1]
                self.suffix_noun_step2a_success = True
                break
        return token

    def __Suffix_Noun_Step2b(self, token):
        for suffix in self.__suffix_noun_step2b:
            if token.endswith(suffix) and len(token) >= 5:
                token = token[:-2]
                self.suffix_noun_step2b_success = True
                break
        return token

    def __Suffix_Noun_Step2c1(self, token):
        for suffix in self.__suffix_noun_step2c1:
            if token.endswith(suffix) and len(token) >= 4:
                token = token[:-1]
                break
        return token

    def __Suffix_Noun_Step1b(self, token):
        for suffix in self.__suffix_noun_step1b:
            if token.endswith(suffix) and len(token) > 5:
                token = token[:-1]
                self.suffixe_noun_step1b_success = True
                break
        return token

    def __Suffix_Noun_Step3(self, token):
        for suffix in self.__suffix_noun_step3:
            if token.endswith(suffix) and len(token) >= 3:
                token = token[:-1]  # ya' nisbiya
                break
        return token

    def __Suffix_All_alef_maqsura(self, token):
        for suffix in self.__suffix_all_alef_maqsura:
            if token.endswith(suffix):
                token = suffix_replace(token, suffix, "\u064a")
        return token

    def __Prefix_Step1(self, token):
        for prefix in self.__prefix_step1:
            if token.startswith(prefix) and len(token) > 3:
                if prefix == "\u0623\u0623":
                    token = prefix_replace(token, prefix, "\u0623")
                    break

                elif prefix == "\u0623\u0622":
                    token = prefix_replace(token, prefix, "\u0622")
                    break

                elif prefix == "\u0623\u0624":
                    token = prefix_replace(token, prefix, "\u0624")
                    break

                elif prefix == "\u0623\u0627":
                    token = prefix_replace(token, prefix, "\u0627")
                    break

                elif prefix == "\u0623\u0625":
                    token = prefix_replace(token, prefix, "\u0625")
                    break
        return token

    def __Prefix_Step2a(self, token):
        for prefix in self.__prefix_step2a:
            if token.startswith(prefix) and len(token) > 5:
                token = token[len(prefix) :]
                self.prefix_step2a_success = True
                break
        return token

    def __Prefix_Step2b(self, token):
        for prefix in self.__prefix_step2b:
            if token.startswith(prefix) and len(token) > 3:
                if token[:2] not in self.__prefixes1:
                    token = token[len(prefix) :]
                    break
        return token

    def __Prefix_Step3a_Noun(self, token):
        for prefix in self.__prefix_step3a_noun:
            if token.startswith(prefix):
                if prefix in self.__articles_2len and len(token) > 4:
                    token = token[len(prefix) :]
                    self.prefix_step3a_noun_success = True
                    break
                if prefix in self.__articles_3len and len(token) > 5:
                    token = token[len(prefix) :]
                    break
        return token

    def __Prefix_Step3b_Noun(self, token):
        for prefix in self.__prefix_step3b_noun:
            if token.startswith(prefix):
                if len(token) > 3:
                    if prefix == "\u0628":
                        token = token[len(prefix) :]
                        self.prefix_step3b_noun_success = True
                        break

                    if prefix in self.__prepositions2:
                        token = prefix_replace(token, prefix, prefix[1])
                        self.prefix_step3b_noun_success = True
                        break

                if prefix in self.__prepositions1 and len(token) > 4:
                    token = token[len(prefix) :]  # BUG: cause confusion
                    self.prefix_step3b_noun_success = True
                    break
        return token

    def __Prefix_Step3_Verb(self, token):
        for prefix in self.__prefix_step3_verb:
            if token.startswith(prefix) and len(token) > 4:
                token = prefix_replace(token, prefix, prefix[1])
                break
        return token

    def __Prefix_Step4_Verb(self, token):
        for prefix in self.__prefix_step4_verb:
            if token.startswith(prefix) and len(token) > 4:
                token = prefix_replace(token, prefix, "\u0627\u0633\u062a")
                self.is_verb = True
                self.is_noun = False
                break
        return token

    def stem(self, word):
        """
        Stem an Arabic word and return the stemmed form.

        :param word: string
        :return: string
        """
        # set initial values
        self.is_verb = True
        self.is_noun = True
        self.is_defined = False

        self.suffix_verb_step2a_success = False
        self.suffix_verb_step2b_success = False
        self.suffix_noun_step2c2_success = False
        self.suffix_noun_step1a_success = False
        self.suffix_noun_step2a_success = False
        self.suffix_noun_step2b_success = False
        self.suffixe_noun_step1b_success = False
        self.prefix_step2a_success = False
        self.prefix_step3a_noun_success = False
        self.prefix_step3b_noun_success = False

        modified_word = word
        # guess type and properties
        # checks1
        self.__checks_1(modified_word)
        # checks2
        self.__checks_2(modified_word)
        # Pre_Normalization
        modified_word = self.__normalize_pre(modified_word)
        # Avoid stopwords
        if modified_word in self.stopwords or len(modified_word) <= 2:
            return modified_word
        # Start stemming
        if self.is_verb:
            modified_word = self.__Suffix_Verb_Step1(modified_word)
            if self.suffixes_verb_step1_success:
                modified_word = self.__Suffix_Verb_Step2a(modified_word)
                if not self.suffix_verb_step2a_success:
                    modified_word = self.__Suffix_Verb_Step2c(modified_word)
                # or next TODO: How to deal with or next instruction
            else:
                modified_word = self.__Suffix_Verb_Step2b(modified_word)
                if not self.suffix_verb_step2b_success:
                    modified_word = self.__Suffix_Verb_Step2a(modified_word)
        if self.is_noun:
            modified_word = self.__Suffix_Noun_Step2c2(modified_word)
            if not self.suffix_noun_step2c2_success:
                if not self.is_defined:
                    modified_word = self.__Suffix_Noun_Step1a(modified_word)
                    # if self.suffix_noun_step1a_success:
                    modified_word = self.__Suffix_Noun_Step2a(modified_word)
                    if not self.suffix_noun_step2a_success:
                        modified_word = self.__Suffix_Noun_Step2b(modified_word)
                    if (
                        not self.suffix_noun_step2b_success
                        and not self.suffix_noun_step2a_success
                    ):
                        modified_word = self.__Suffix_Noun_Step2c1(modified_word)
                    # or next ? todo : how to deal with or next
                else:
                    modified_word = self.__Suffix_Noun_Step1b(modified_word)
                    if self.suffixe_noun_step1b_success:
                        modified_word = self.__Suffix_Noun_Step2a(modified_word)
                        if not self.suffix_noun_step2a_success:
                            modified_word = self.__Suffix_Noun_Step2b(modified_word)
                        if (
                            not self.suffix_noun_step2b_success
                            and not self.suffix_noun_step2a_success
                        ):
                            modified_word = self.__Suffix_Noun_Step2c1(modified_word)
                    else:
                        if not self.is_defined:
                            modified_word = self.__Suffix_Noun_Step2a(modified_word)
                        modified_word = self.__Suffix_Noun_Step2b(modified_word)
            modified_word = self.__Suffix_Noun_Step3(modified_word)
        if not self.is_noun and self.is_verb:
            modified_word = self.__Suffix_All_alef_maqsura(modified_word)

        # prefixes
        modified_word = self.__Prefix_Step1(modified_word)
        modified_word = self.__Prefix_Step2a(modified_word)
        if not self.prefix_step2a_success:
            modified_word = self.__Prefix_Step2b(modified_word)
        modified_word = self.__Prefix_Step3a_Noun(modified_word)
        if not self.prefix_step3a_noun_success and self.is_noun:
            modified_word = self.__Prefix_Step3b_Noun(modified_word)
        else:
            if not self.prefix_step3b_noun_success and self.is_verb:
                modified_word = self.__Prefix_Step3_Verb(modified_word)
                modified_word = self.__Prefix_Step4_Verb(modified_word)

        # post normalization stemming
        modified_word = self.__normalize_post(modified_word)
        stemmed_word = modified_word
        return stemmed_word


class DanishStemmer(_ScandinavianStemmer):

    """
    The Danish Snowball stemmer.

    :cvar __vowels: The Danish vowels.
    :type __vowels: unicode
    :cvar __consonants: The Danish consonants.
    :type __consonants: unicode
    :cvar __double_consonants: The Danish double consonants.
    :type __double_consonants: tuple
    :cvar __s_ending: Letters that may directly appear before a word final 's'.
    :type __s_ending: unicode
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :note: A detailed description of the Danish
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/danish/stemmer.html

    """

    # The language's vowels and other important characters are defined.
    __vowels = "aeiouy\xE6\xE5\xF8"
    __consonants = "bcdfghjklmnpqrstvwxz"
    __double_consonants = (
        "bb",
        "cc",
        "dd",
        "ff",
        "gg",
        "hh",
        "jj",
        "kk",
        "ll",
        "mm",
        "nn",
        "pp",
        "qq",
        "rr",
        "ss",
        "tt",
        "vv",
        "ww",
        "xx",
        "zz",
    )
    __s_ending = "abcdfghjklmnoprtvyz\xE5"

    # The different suffixes, divided into the algorithm's steps
    # and organized by length, are listed in tuples.
    __step1_suffixes = (
        "erendes",
        "erende",
        "hedens",
        "ethed",
        "erede",
        "heden",
        "heder",
        "endes",
        "ernes",
        "erens",
        "erets",
        "ered",
        "ende",
        "erne",
        "eren",
        "erer",
        "heds",
        "enes",
        "eres",
        "eret",
        "hed",
        "ene",
        "ere",
        "ens",
        "ers",
        "ets",
        "en",
        "er",
        "es",
        "et",
        "e",
        "s",
    )
    __step2_suffixes = ("gd", "dt", "gt", "kt")
    __step3_suffixes = ("elig", "l\xF8st", "lig", "els", "ig")

    def stem(self, word):
        """
        Stem a Danish word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        # Every word is put into lower case for normalization.
        word = word.lower()

        if word in self.stopwords:
            return word

        # After this, the required regions are generated
        # by the respective helper method.
        r1 = self._r1_scandinavian(word, self.__vowels)

        # Then the actual stemming process starts.
        # Every new step is explicitly indicated
        # according to the descriptions on the Snowball website.

        # STEP 1
        for suffix in self.__step1_suffixes:
            if r1.endswith(suffix):
                if suffix == "s":
                    if word[-2] in self.__s_ending:
                        word = word[:-1]
                        r1 = r1[:-1]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                break

        # STEP 2
        for suffix in self.__step2_suffixes:
            if r1.endswith(suffix):
                word = word[:-1]
                r1 = r1[:-1]
                break

        # STEP 3
        if r1.endswith("igst"):
            word = word[:-2]
            r1 = r1[:-2]

        for suffix in self.__step3_suffixes:
            if r1.endswith(suffix):
                if suffix == "l\xF8st":
                    word = word[:-1]
                    r1 = r1[:-1]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]

                    if r1.endswith(self.__step2_suffixes):
                        word = word[:-1]
                        r1 = r1[:-1]
                break

        # STEP 4: Undouble
        for double_cons in self.__double_consonants:
            if word.endswith(double_cons) and len(word) > 3:
                word = word[:-1]
                break

        return word


class DutchStemmer(_StandardStemmer):

    """
    The Dutch Snowball stemmer.

    :cvar __vowels: The Dutch vowels.
    :type __vowels: unicode
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step3b_suffixes: Suffixes to be deleted in step 3b of the algorithm.
    :type __step3b_suffixes: tuple
    :note: A detailed description of the Dutch
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/dutch/stemmer.html

    """

    __vowels = "aeiouy\xE8"
    __step1_suffixes = ("heden", "ene", "en", "se", "s")
    __step3b_suffixes = ("baar", "lijk", "bar", "end", "ing", "ig")

    def stem(self, word):
        """
        Stem a Dutch word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        step2_success = False

        # Vowel accents are removed.
        word = (
            word.replace("\xE4", "a")
            .replace("\xE1", "a")
            .replace("\xEB", "e")
            .replace("\xE9", "e")
            .replace("\xED", "i")
            .replace("\xEF", "i")
            .replace("\xF6", "o")
            .replace("\xF3", "o")
            .replace("\xFC", "u")
            .replace("\xFA", "u")
        )

        # An initial 'y', a 'y' after a vowel,
        # and an 'i' between self.__vowels is put into upper case.
        # As from now these are treated as consonants.
        if word.startswith("y"):
            word = "".join(("Y", word[1:]))

        for i in range(1, len(word)):
            if word[i - 1] in self.__vowels and word[i] == "y":
                word = "".join((word[:i], "Y", word[i + 1 :]))

        for i in range(1, len(word) - 1):
            if (
                word[i - 1] in self.__vowels
                and word[i] == "i"
                and word[i + 1] in self.__vowels
            ):
                word = "".join((word[:i], "I", word[i + 1 :]))

        r1, r2 = self._r1r2_standard(word, self.__vowels)

        # R1 is adjusted so that the region before it
        # contains at least 3 letters.
        for i in range(1, len(word)):
            if word[i] not in self.__vowels and word[i - 1] in self.__vowels:
                if 3 > len(word[: i + 1]) > 0:
                    r1 = word[3:]
                elif len(word[: i + 1]) == 0:
                    return word
                break

        # STEP 1
        for suffix in self.__step1_suffixes:
            if r1.endswith(suffix):
                if suffix == "heden":
                    word = suffix_replace(word, suffix, "heid")
                    r1 = suffix_replace(r1, suffix, "heid")
                    if r2.endswith("heden"):
                        r2 = suffix_replace(r2, suffix, "heid")

                elif (
                    suffix in ("ene", "en")
                    and not word.endswith("heden")
                    and word[-len(suffix) - 1] not in self.__vowels
                    and word[-len(suffix) - 3 : -len(suffix)] != "gem"
                ):
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    if word.endswith(("kk", "dd", "tt")):
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]

                elif (
                    suffix in ("se", "s")
                    and word[-len(suffix) - 1] not in self.__vowels
                    and word[-len(suffix) - 1] != "j"
                ):
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                break

        # STEP 2
        if r1.endswith("e") and word[-2] not in self.__vowels:
            step2_success = True
            word = word[:-1]
            r1 = r1[:-1]
            r2 = r2[:-1]

            if word.endswith(("kk", "dd", "tt")):
                word = word[:-1]
                r1 = r1[:-1]
                r2 = r2[:-1]

        # STEP 3a
        if r2.endswith("heid") and word[-5] != "c":
            word = word[:-4]
            r1 = r1[:-4]
            r2 = r2[:-4]

            if (
                r1.endswith("en")
                and word[-3] not in self.__vowels
                and word[-5:-2] != "gem"
            ):
                word = word[:-2]
                r1 = r1[:-2]
                r2 = r2[:-2]

                if word.endswith(("kk", "dd", "tt")):
                    word = word[:-1]
                    r1 = r1[:-1]
                    r2 = r2[:-1]

        # STEP 3b: Derivational suffixes
        for suffix in self.__step3b_suffixes:
            if r2.endswith(suffix):
                if suffix in ("end", "ing"):
                    word = word[:-3]
                    r2 = r2[:-3]

                    if r2.endswith("ig") and word[-3] != "e":
                        word = word[:-2]
                    else:
                        if word.endswith(("kk", "dd", "tt")):
                            word = word[:-1]

                elif suffix == "ig" and word[-3] != "e":
                    word = word[:-2]

                elif suffix == "lijk":
                    word = word[:-4]
                    r1 = r1[:-4]

                    if r1.endswith("e") and word[-2] not in self.__vowels:
                        word = word[:-1]
                        if word.endswith(("kk", "dd", "tt")):
                            word = word[:-1]

                elif suffix == "baar":
                    word = word[:-4]

                elif suffix == "bar" and step2_success:
                    word = word[:-3]
                break

        # STEP 4: Undouble vowel
        if len(word) >= 4:
            if word[-1] not in self.__vowels and word[-1] != "I":
                if word[-3:-1] in ("aa", "ee", "oo", "uu"):
                    if word[-4] not in self.__vowels:
                        word = "".join((word[:-3], word[-3], word[-1]))

        # All occurrences of 'I' and 'Y' are put back into lower case.
        word = word.replace("I", "i").replace("Y", "y")

        return word


class EnglishStemmer(_StandardStemmer):

    """
    The English Snowball stemmer.

    :cvar __vowels: The English vowels.
    :type __vowels: unicode
    :cvar __double_consonants: The English double consonants.
    :type __double_consonants: tuple
    :cvar __li_ending: Letters that may directly appear before a word final 'li'.
    :type __li_ending: unicode
    :cvar __step0_suffixes: Suffixes to be deleted in step 0 of the algorithm.
    :type __step0_suffixes: tuple
    :cvar __step1a_suffixes: Suffixes to be deleted in step 1a of the algorithm.
    :type __step1a_suffixes: tuple
    :cvar __step1b_suffixes: Suffixes to be deleted in step 1b of the algorithm.
    :type __step1b_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :cvar __step4_suffixes: Suffixes to be deleted in step 4 of the algorithm.
    :type __step4_suffixes: tuple
    :cvar __step5_suffixes: Suffixes to be deleted in step 5 of the algorithm.
    :type __step5_suffixes: tuple
    :cvar __special_words: A dictionary containing words
                           which have to be stemmed specially.
    :type __special_words: dict
    :note: A detailed description of the English
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/english/stemmer.html
    """

    __vowels = "aeiouy"
    __double_consonants = ("bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt")
    __li_ending = "cdeghkmnrt"
    __step0_suffixes = ("'s'", "'s", "'")
    __step1a_suffixes = ("sses", "ied", "ies", "us", "ss", "s")
    __step1b_suffixes = ("eedly", "ingly", "edly", "eed", "ing", "ed")
    __step2_suffixes = (
        "ization",
        "ational",
        "fulness",
        "ousness",
        "iveness",
        "tional",
        "biliti",
        "lessli",
        "entli",
        "ation",
        "alism",
        "aliti",
        "ousli",
        "iviti",
        "fulli",
        "enci",
        "anci",
        "abli",
        "izer",
        "ator",
        "alli",
        "bli",
        "ogi",
        "li",
    )
    __step3_suffixes = (
        "ational",
        "tional",
        "alize",
        "icate",
        "iciti",
        "ative",
        "ical",
        "ness",
        "ful",
    )
    __step4_suffixes = (
        "ement",
        "ance",
        "ence",
        "able",
        "ible",
        "ment",
        "ant",
        "ent",
        "ism",
        "ate",
        "iti",
        "ous",
        "ive",
        "ize",
        "ion",
        "al",
        "er",
        "ic",
    )
    __step5_suffixes = ("e", "l")
    __special_words = {
        "skis": "ski",
        "skies": "sky",
        "dying": "die",
        "lying": "lie",
        "tying": "tie",
        "idly": "idl",
        "gently": "gentl",
        "ugly": "ugli",
        "early": "earli",
        "only": "onli",
        "singly": "singl",
        "sky": "sky",
        "news": "news",
        "howe": "howe",
        "atlas": "atlas",
        "cosmos": "cosmos",
        "bias": "bias",
        "andes": "andes",
        "inning": "inning",
        "innings": "inning",
        "outing": "outing",
        "outings": "outing",
        "canning": "canning",
        "cannings": "canning",
        "herring": "herring",
        "herrings": "herring",
        "earring": "earring",
        "earrings": "earring",
        "proceed": "proceed",
        "proceeds": "proceed",
        "proceeded": "proceed",
        "proceeding": "proceed",
        "exceed": "exceed",
        "exceeds": "exceed",
        "exceeded": "exceed",
        "exceeding": "exceed",
        "succeed": "succeed",
        "succeeds": "succeed",
        "succeeded": "succeed",
        "succeeding": "succeed",
    }

    def stem(self, word):

        """
        Stem an English word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords or len(word) <= 2:
            return word

        elif word in self.__special_words:
            return self.__special_words[word]

        # Map the different apostrophe characters to a single consistent one
        word = (
            word.replace("\u2019", "\x27")
            .replace("\u2018", "\x27")
            .replace("\u201B", "\x27")
        )

        if word.startswith("\x27"):
            word = word[1:]

        if word.startswith("y"):
            word = "".join(("Y", word[1:]))

        for i in range(1, len(word)):
            if word[i - 1] in self.__vowels and word[i] == "y":
                word = "".join((word[:i], "Y", word[i + 1 :]))

        step1a_vowel_found = False
        step1b_vowel_found = False

        r1 = ""
        r2 = ""

        if word.startswith(("gener", "commun", "arsen")):
            if word.startswith(("gener", "arsen")):
                r1 = word[5:]
            else:
                r1 = word[6:]

            for i in range(1, len(r1)):
                if r1[i] not in self.__vowels and r1[i - 1] in self.__vowels:
                    r2 = r1[i + 1 :]
                    break
        else:
            r1, r2 = self._r1r2_standard(word, self.__vowels)

        # STEP 0
        for suffix in self.__step0_suffixes:
            if word.endswith(suffix):
                word = word[: -len(suffix)]
                r1 = r1[: -len(suffix)]
                r2 = r2[: -len(suffix)]
                break

        # STEP 1a
        for suffix in self.__step1a_suffixes:
            if word.endswith(suffix):

                if suffix == "sses":
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]

                elif suffix in ("ied", "ies"):
                    if len(word[: -len(suffix)]) > 1:
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                    else:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]

                elif suffix == "s":
                    for letter in word[:-2]:
                        if letter in self.__vowels:
                            step1a_vowel_found = True
                            break

                    if step1a_vowel_found:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                break

        # STEP 1b
        for suffix in self.__step1b_suffixes:
            if word.endswith(suffix):
                if suffix in ("eed", "eedly"):

                    if r1.endswith(suffix):
                        word = suffix_replace(word, suffix, "ee")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ee")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ee")
                        else:
                            r2 = ""
                else:
                    for letter in word[: -len(suffix)]:
                        if letter in self.__vowels:
                            step1b_vowel_found = True
                            break

                    if step1b_vowel_found:
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]

                        if word.endswith(("at", "bl", "iz")):
                            word = "".join((word, "e"))
                            r1 = "".join((r1, "e"))

                            if len(word) > 5 or len(r1) >= 3:
                                r2 = "".join((r2, "e"))

                        elif word.endswith(self.__double_consonants):
                            word = word[:-1]
                            r1 = r1[:-1]
                            r2 = r2[:-1]

                        elif (
                            r1 == ""
                            and len(word) >= 3
                            and word[-1] not in self.__vowels
                            and word[-1] not in "wxY"
                            and word[-2] in self.__vowels
                            and word[-3] not in self.__vowels
                        ) or (
                            r1 == ""
                            and len(word) == 2
                            and word[0] in self.__vowels
                            and word[1] not in self.__vowels
                        ):

                            word = "".join((word, "e"))

                            if len(r1) > 0:
                                r1 = "".join((r1, "e"))

                            if len(r2) > 0:
                                r2 = "".join((r2, "e"))
                break

        # STEP 1c
        if len(word) > 2 and word[-1] in "yY" and word[-2] not in self.__vowels:
            word = "".join((word[:-1], "i"))
            if len(r1) >= 1:
                r1 = "".join((r1[:-1], "i"))
            else:
                r1 = ""

            if len(r2) >= 1:
                r2 = "".join((r2[:-1], "i"))
            else:
                r2 = ""

        # STEP 2
        for suffix in self.__step2_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix == "tional":
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                    elif suffix in ("enci", "anci", "abli"):
                        word = "".join((word[:-1], "e"))

                        if len(r1) >= 1:
                            r1 = "".join((r1[:-1], "e"))
                        else:
                            r1 = ""

                        if len(r2) >= 1:
                            r2 = "".join((r2[:-1], "e"))
                        else:
                            r2 = ""

                    elif suffix == "entli":
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                    elif suffix in ("izer", "ization"):
                        word = suffix_replace(word, suffix, "ize")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ize")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ize")
                        else:
                            r2 = ""

                    elif suffix in ("ational", "ation", "ator"):
                        word = suffix_replace(word, suffix, "ate")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ate")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ate")
                        else:
                            r2 = "e"

                    elif suffix in ("alism", "aliti", "alli"):
                        word = suffix_replace(word, suffix, "al")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "al")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "al")
                        else:
                            r2 = ""

                    elif suffix == "fulness":
                        word = word[:-4]
                        r1 = r1[:-4]
                        r2 = r2[:-4]

                    elif suffix in ("ousli", "ousness"):
                        word = suffix_replace(word, suffix, "ous")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ous")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ous")
                        else:
                            r2 = ""

                    elif suffix in ("iveness", "iviti"):
                        word = suffix_replace(word, suffix, "ive")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ive")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ive")
                        else:
                            r2 = "e"

                    elif suffix in ("biliti", "bli"):
                        word = suffix_replace(word, suffix, "ble")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ble")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ble")
                        else:
                            r2 = ""

                    elif suffix == "ogi" and word[-4] == "l":
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]

                    elif suffix in ("fulli", "lessli"):
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                    elif suffix == "li" and word[-3] in self.__li_ending:
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                break

        # STEP 3
        for suffix in self.__step3_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix == "tional":
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                    elif suffix == "ational":
                        word = suffix_replace(word, suffix, "ate")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ate")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ate")
                        else:
                            r2 = ""

                    elif suffix == "alize":
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]

                    elif suffix in ("icate", "iciti", "ical"):
                        word = suffix_replace(word, suffix, "ic")

                        if len(r1) >= len(suffix):
                            r1 = suffix_replace(r1, suffix, "ic")
                        else:
                            r1 = ""

                        if len(r2) >= len(suffix):
                            r2 = suffix_replace(r2, suffix, "ic")
                        else:
                            r2 = ""

                    elif suffix in ("ful", "ness"):
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]

                    elif suffix == "ative" and r2.endswith(suffix):
                        word = word[:-5]
                        r1 = r1[:-5]
                        r2 = r2[:-5]
                break

        # STEP 4
        for suffix in self.__step4_suffixes:
            if word.endswith(suffix):
                if r2.endswith(suffix):
                    if suffix == "ion":
                        if word[-4] in "st":
                            word = word[:-3]
                            r1 = r1[:-3]
                            r2 = r2[:-3]
                    else:
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                break

        # STEP 5
        if r2.endswith("l") and word[-2] == "l":
            word = word[:-1]
        elif r2.endswith("e"):
            word = word[:-1]
        elif r1.endswith("e"):
            if len(word) >= 4 and (
                word[-2] in self.__vowels
                or word[-2] in "wxY"
                or word[-3] not in self.__vowels
                or word[-4] in self.__vowels
            ):
                word = word[:-1]

        word = word.replace("Y", "y")

        return word


class FinnishStemmer(_StandardStemmer):

    """
    The Finnish Snowball stemmer.

    :cvar __vowels: The Finnish vowels.
    :type __vowels: unicode
    :cvar __restricted_vowels: A subset of the Finnish vowels.
    :type __restricted_vowels: unicode
    :cvar __long_vowels: The Finnish vowels in their long forms.
    :type __long_vowels: tuple
    :cvar __consonants: The Finnish consonants.
    :type __consonants: unicode
    :cvar __double_consonants: The Finnish double consonants.
    :type __double_consonants: tuple
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :cvar __step4_suffixes: Suffixes to be deleted in step 4 of the algorithm.
    :type __step4_suffixes: tuple
    :note: A detailed description of the Finnish
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/finnish/stemmer.html
    """

    __vowels = "aeiouy\xE4\xF6"
    __restricted_vowels = "aeiou\xE4\xF6"
    __long_vowels = ("aa", "ee", "ii", "oo", "uu", "\xE4\xE4", "\xF6\xF6")
    __consonants = "bcdfghjklmnpqrstvwxz"
    __double_consonants = (
        "bb",
        "cc",
        "dd",
        "ff",
        "gg",
        "hh",
        "jj",
        "kk",
        "ll",
        "mm",
        "nn",
        "pp",
        "qq",
        "rr",
        "ss",
        "tt",
        "vv",
        "ww",
        "xx",
        "zz",
    )
    __step1_suffixes = (
        "kaan",
        "k\xE4\xE4n",
        "sti",
        "kin",
        "han",
        "h\xE4n",
        "ko",
        "k\xF6",
        "pa",
        "p\xE4",
    )
    __step2_suffixes = ("nsa", "ns\xE4", "mme", "nne", "si", "ni", "an", "\xE4n", "en")
    __step3_suffixes = (
        "siin",
        "tten",
        "seen",
        "han",
        "hen",
        "hin",
        "hon",
        "h\xE4n",
        "h\xF6n",
        "den",
        "tta",
        "tt\xE4",
        "ssa",
        "ss\xE4",
        "sta",
        "st\xE4",
        "lla",
        "ll\xE4",
        "lta",
        "lt\xE4",
        "lle",
        "ksi",
        "ine",
        "ta",
        "t\xE4",
        "na",
        "n\xE4",
        "a",
        "\xE4",
        "n",
    )
    __step4_suffixes = (
        "impi",
        "impa",
        "imp\xE4",
        "immi",
        "imma",
        "imm\xE4",
        "mpi",
        "mpa",
        "mp\xE4",
        "mmi",
        "mma",
        "mm\xE4",
        "eja",
        "ej\xE4",
    )

    def stem(self, word):
        """
        Stem a Finnish word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        step3_success = False

        r1, r2 = self._r1r2_standard(word, self.__vowels)

        # STEP 1: Particles etc.
        for suffix in self.__step1_suffixes:
            if r1.endswith(suffix):
                if suffix == "sti":
                    if suffix in r2:
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                else:
                    if word[-len(suffix) - 1] in "ntaeiouy\xE4\xF6":
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                break

        # STEP 2: Possessives
        for suffix in self.__step2_suffixes:
            if r1.endswith(suffix):
                if suffix == "si":
                    if word[-3] != "k":
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                elif suffix == "ni":
                    word = word[:-2]
                    r1 = r1[:-2]
                    r2 = r2[:-2]
                    if word.endswith("kse"):
                        word = suffix_replace(word, "kse", "ksi")

                    if r1.endswith("kse"):
                        r1 = suffix_replace(r1, "kse", "ksi")

                    if r2.endswith("kse"):
                        r2 = suffix_replace(r2, "kse", "ksi")

                elif suffix == "an":
                    if word[-4:-2] in ("ta", "na") or word[-5:-2] in (
                        "ssa",
                        "sta",
                        "lla",
                        "lta",
                    ):
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                elif suffix == "\xE4n":
                    if word[-4:-2] in ("t\xE4", "n\xE4") or word[-5:-2] in (
                        "ss\xE4",
                        "st\xE4",
                        "ll\xE4",
                        "lt\xE4",
                    ):
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]

                elif suffix == "en":
                    if word[-5:-2] in ("lle", "ine"):
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                else:
                    word = word[:-3]
                    r1 = r1[:-3]
                    r2 = r2[:-3]
                break

        # STEP 3: Cases
        for suffix in self.__step3_suffixes:
            if r1.endswith(suffix):
                if suffix in ("han", "hen", "hin", "hon", "h\xE4n", "h\xF6n"):
                    if (
                        (suffix == "han" and word[-4] == "a")
                        or (suffix == "hen" and word[-4] == "e")
                        or (suffix == "hin" and word[-4] == "i")
                        or (suffix == "hon" and word[-4] == "o")
                        or (suffix == "h\xE4n" and word[-4] == "\xE4")
                        or (suffix == "h\xF6n" and word[-4] == "\xF6")
                    ):
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                        step3_success = True

                elif suffix in ("siin", "den", "tten"):
                    if (
                        word[-len(suffix) - 1] == "i"
                        and word[-len(suffix) - 2] in self.__restricted_vowels
                    ):
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        step3_success = True
                    else:
                        continue

                elif suffix == "seen":
                    if word[-6:-4] in self.__long_vowels:
                        word = word[:-4]
                        r1 = r1[:-4]
                        r2 = r2[:-4]
                        step3_success = True
                    else:
                        continue

                elif suffix in ("a", "\xE4"):
                    if word[-2] in self.__vowels and word[-3] in self.__consonants:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                        step3_success = True

                elif suffix in ("tta", "tt\xE4"):
                    if word[-4] == "e":
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                        step3_success = True

                elif suffix == "n":
                    word = word[:-1]
                    r1 = r1[:-1]
                    r2 = r2[:-1]
                    step3_success = True

                    if word[-2:] == "ie" or word[-2:] in self.__long_vowels:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    step3_success = True
                break

        # STEP 4: Other endings
        for suffix in self.__step4_suffixes:
            if r2.endswith(suffix):
                if suffix in ("mpi", "mpa", "mp\xE4", "mmi", "mma", "mm\xE4"):
                    if word[-5:-3] != "po":
                        word = word[:-3]
                        r1 = r1[:-3]
                        r2 = r2[:-3]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                break

        # STEP 5: Plurals
        if step3_success and len(r1) >= 1 and r1[-1] in "ij":
            word = word[:-1]
            r1 = r1[:-1]

        elif (
            not step3_success
            and len(r1) >= 2
            and r1[-1] == "t"
            and r1[-2] in self.__vowels
        ):
            word = word[:-1]
            r1 = r1[:-1]
            r2 = r2[:-1]
            if r2.endswith("imma"):
                word = word[:-4]
                r1 = r1[:-4]
            elif r2.endswith("mma") and r2[-5:-3] != "po":
                word = word[:-3]
                r1 = r1[:-3]

        # STEP 6: Tidying up
        if r1[-2:] in self.__long_vowels:
            word = word[:-1]
            r1 = r1[:-1]

        if len(r1) >= 2 and r1[-2] in self.__consonants and r1[-1] in "a\xE4ei":
            word = word[:-1]
            r1 = r1[:-1]

        if r1.endswith(("oj", "uj")):
            word = word[:-1]
            r1 = r1[:-1]

        if r1.endswith("jo"):
            word = word[:-1]
            r1 = r1[:-1]

        # If the word ends with a double consonant
        # followed by zero or more vowels, the last consonant is removed.
        for i in range(1, len(word)):
            if word[-i] in self.__vowels:
                continue
            else:
                if i == 1:
                    if word[-i - 1 :] in self.__double_consonants:
                        word = word[:-1]
                else:
                    if word[-i - 1 : -i + 1] in self.__double_consonants:
                        word = "".join((word[:-i], word[-i + 1 :]))
                break

        return word


class FrenchStemmer(_StandardStemmer):

    """
    The French Snowball stemmer.

    :cvar __vowels: The French vowels.
    :type __vowels: unicode
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2a_suffixes: Suffixes to be deleted in step 2a of the algorithm.
    :type __step2a_suffixes: tuple
    :cvar __step2b_suffixes: Suffixes to be deleted in step 2b of the algorithm.
    :type __step2b_suffixes: tuple
    :cvar __step4_suffixes: Suffixes to be deleted in step 4 of the algorithm.
    :type __step4_suffixes: tuple
    :note: A detailed description of the French
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/french/stemmer.html
    """

    __vowels = "aeiouy\xE2\xE0\xEB\xE9\xEA\xE8\xEF\xEE\xF4\xFB\xF9"
    __step1_suffixes = (
        "issements",
        "issement",
        "atrices",
        "atrice",
        "ateurs",
        "ations",
        "logies",
        "usions",
        "utions",
        "ements",
        "amment",
        "emment",
        "ances",
        "iqUes",
        "ismes",
        "ables",
        "istes",
        "ateur",
        "ation",
        "logie",
        "usion",
        "ution",
        "ences",
        "ement",
        "euses",
        "ments",
        "ance",
        "iqUe",
        "isme",
        "able",
        "iste",
        "ence",
        "it\xE9s",
        "ives",
        "eaux",
        "euse",
        "ment",
        "eux",
        "it\xE9",
        "ive",
        "ifs",
        "aux",
        "if",
    )
    __step2a_suffixes = (
        "issaIent",
        "issantes",
        "iraIent",
        "issante",
        "issants",
        "issions",
        "irions",
        "issais",
        "issait",
        "issant",
        "issent",
        "issiez",
        "issons",
        "irais",
        "irait",
        "irent",
        "iriez",
        "irons",
        "iront",
        "isses",
        "issez",
        "\xEEmes",
        "\xEEtes",
        "irai",
        "iras",
        "irez",
        "isse",
        "ies",
        "ira",
        "\xEEt",
        "ie",
        "ir",
        "is",
        "it",
        "i",
    )
    __step2b_suffixes = (
        "eraIent",
        "assions",
        "erions",
        "assent",
        "assiez",
        "\xE8rent",
        "erais",
        "erait",
        "eriez",
        "erons",
        "eront",
        "aIent",
        "antes",
        "asses",
        "ions",
        "erai",
        "eras",
        "erez",
        "\xE2mes",
        "\xE2tes",
        "ante",
        "ants",
        "asse",
        "\xE9es",
        "era",
        "iez",
        "ais",
        "ait",
        "ant",
        "\xE9e",
        "\xE9s",
        "er",
        "ez",
        "\xE2t",
        "ai",
        "as",
        "\xE9",
        "a",
    )
    __step4_suffixes = ("i\xE8re", "I\xE8re", "ion", "ier", "Ier", "e", "\xEB")

    def stem(self, word):
        """
        Stem a French word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        step1_success = False
        rv_ending_found = False
        step2a_success = False
        step2b_success = False

        # Every occurrence of 'u' after 'q' is put into upper case.
        for i in range(1, len(word)):
            if word[i - 1] == "q" and word[i] == "u":
                word = "".join((word[:i], "U", word[i + 1 :]))

        # Every occurrence of 'u' and 'i'
        # between vowels is put into upper case.
        # Every occurrence of 'y' preceded or
        # followed by a vowel is also put into upper case.
        for i in range(1, len(word) - 1):
            if word[i - 1] in self.__vowels and word[i + 1] in self.__vowels:
                if word[i] == "u":
                    word = "".join((word[:i], "U", word[i + 1 :]))

                elif word[i] == "i":
                    word = "".join((word[:i], "I", word[i + 1 :]))

            if word[i - 1] in self.__vowels or word[i + 1] in self.__vowels:
                if word[i] == "y":
                    word = "".join((word[:i], "Y", word[i + 1 :]))

        r1, r2 = self._r1r2_standard(word, self.__vowels)
        rv = self.__rv_french(word, self.__vowels)

        # STEP 1: Standard suffix removal
        for suffix in self.__step1_suffixes:
            if word.endswith(suffix):
                if suffix == "eaux":
                    word = word[:-1]
                    step1_success = True

                elif suffix in ("euse", "euses"):
                    if suffix in r2:
                        word = word[: -len(suffix)]
                        step1_success = True

                    elif suffix in r1:
                        word = suffix_replace(word, suffix, "eux")
                        step1_success = True

                elif suffix in ("ement", "ements") and suffix in rv:
                    word = word[: -len(suffix)]
                    step1_success = True

                    if word[-2:] == "iv" and "iv" in r2:
                        word = word[:-2]

                        if word[-2:] == "at" and "at" in r2:
                            word = word[:-2]

                    elif word[-3:] == "eus":
                        if "eus" in r2:
                            word = word[:-3]
                        elif "eus" in r1:
                            word = "".join((word[:-1], "x"))

                    elif word[-3:] in ("abl", "iqU"):
                        if "abl" in r2 or "iqU" in r2:
                            word = word[:-3]

                    elif word[-3:] in ("i\xE8r", "I\xE8r"):
                        if "i\xE8r" in rv or "I\xE8r" in rv:
                            word = "".join((word[:-3], "i"))

                elif suffix == "amment" and suffix in rv:
                    word = suffix_replace(word, "amment", "ant")
                    rv = suffix_replace(rv, "amment", "ant")
                    rv_ending_found = True

                elif suffix == "emment" and suffix in rv:
                    word = suffix_replace(word, "emment", "ent")
                    rv_ending_found = True

                elif (
                    suffix in ("ment", "ments")
                    and suffix in rv
                    and not rv.startswith(suffix)
                    and rv[rv.rindex(suffix) - 1] in self.__vowels
                ):
                    word = word[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    rv_ending_found = True

                elif suffix == "aux" and suffix in r1:
                    word = "".join((word[:-2], "l"))
                    step1_success = True

                elif (
                    suffix in ("issement", "issements")
                    and suffix in r1
                    and word[-len(suffix) - 1] not in self.__vowels
                ):
                    word = word[: -len(suffix)]
                    step1_success = True

                elif (
                    suffix
                    in (
                        "ance",
                        "iqUe",
                        "isme",
                        "able",
                        "iste",
                        "eux",
                        "ances",
                        "iqUes",
                        "ismes",
                        "ables",
                        "istes",
                    )
                    and suffix in r2
                ):
                    word = word[: -len(suffix)]
                    step1_success = True

                elif (
                    suffix
                    in ("atrice", "ateur", "ation", "atrices", "ateurs", "ations")
                    and suffix in r2
                ):
                    word = word[: -len(suffix)]
                    step1_success = True

                    if word[-2:] == "ic":
                        if "ic" in r2:
                            word = word[:-2]
                        else:
                            word = "".join((word[:-2], "iqU"))

                elif suffix in ("logie", "logies") and suffix in r2:
                    word = suffix_replace(word, suffix, "log")
                    step1_success = True

                elif suffix in ("usion", "ution", "usions", "utions") and suffix in r2:
                    word = suffix_replace(word, suffix, "u")
                    step1_success = True

                elif suffix in ("ence", "ences") and suffix in r2:
                    word = suffix_replace(word, suffix, "ent")
                    step1_success = True

                elif suffix in ("it\xE9", "it\xE9s") and suffix in r2:
                    word = word[: -len(suffix)]
                    step1_success = True

                    if word[-4:] == "abil":
                        if "abil" in r2:
                            word = word[:-4]
                        else:
                            word = "".join((word[:-2], "l"))

                    elif word[-2:] == "ic":
                        if "ic" in r2:
                            word = word[:-2]
                        else:
                            word = "".join((word[:-2], "iqU"))

                    elif word[-2:] == "iv":
                        if "iv" in r2:
                            word = word[:-2]

                elif suffix in ("if", "ive", "ifs", "ives") and suffix in r2:
                    word = word[: -len(suffix)]
                    step1_success = True

                    if word[-2:] == "at" and "at" in r2:
                        word = word[:-2]

                        if word[-2:] == "ic":
                            if "ic" in r2:
                                word = word[:-2]
                            else:
                                word = "".join((word[:-2], "iqU"))
                break

        # STEP 2a: Verb suffixes beginning 'i'
        if not step1_success or rv_ending_found:
            for suffix in self.__step2a_suffixes:
                if word.endswith(suffix):
                    if (
                        suffix in rv
                        and len(rv) > len(suffix)
                        and rv[rv.rindex(suffix) - 1] not in self.__vowels
                    ):
                        word = word[: -len(suffix)]
                        step2a_success = True
                    break

            # STEP 2b: Other verb suffixes
            if not step2a_success:
                for suffix in self.__step2b_suffixes:
                    if rv.endswith(suffix):
                        if suffix == "ions" and "ions" in r2:
                            word = word[:-4]
                            step2b_success = True

                        elif suffix in (
                            "eraIent",
                            "erions",
                            "\xE8rent",
                            "erais",
                            "erait",
                            "eriez",
                            "erons",
                            "eront",
                            "erai",
                            "eras",
                            "erez",
                            "\xE9es",
                            "era",
                            "iez",
                            "\xE9e",
                            "\xE9s",
                            "er",
                            "ez",
                            "\xE9",
                        ):
                            word = word[: -len(suffix)]
                            step2b_success = True

                        elif suffix in (
                            "assions",
                            "assent",
                            "assiez",
                            "aIent",
                            "antes",
                            "asses",
                            "\xE2mes",
                            "\xE2tes",
                            "ante",
                            "ants",
                            "asse",
                            "ais",
                            "ait",
                            "ant",
                            "\xE2t",
                            "ai",
                            "as",
                            "a",
                        ):
                            word = word[: -len(suffix)]
                            rv = rv[: -len(suffix)]
                            step2b_success = True
                            if rv.endswith("e"):
                                word = word[:-1]
                        break

        # STEP 3
        if step1_success or step2a_success or step2b_success:
            if word[-1] == "Y":
                word = "".join((word[:-1], "i"))
            elif word[-1] == "\xE7":
                word = "".join((word[:-1], "c"))

        # STEP 4: Residual suffixes
        else:
            if len(word) >= 2 and word[-1] == "s" and word[-2] not in "aiou\xE8s":
                word = word[:-1]

            for suffix in self.__step4_suffixes:
                if word.endswith(suffix):
                    if suffix in rv:
                        if suffix == "ion" and suffix in r2 and rv[-4] in "st":
                            word = word[:-3]

                        elif suffix in ("ier", "i\xE8re", "Ier", "I\xE8re"):
                            word = suffix_replace(word, suffix, "i")

                        elif suffix == "e":
                            word = word[:-1]

                        elif suffix == "\xEB" and word[-3:-1] == "gu":
                            word = word[:-1]
                        break

        # STEP 5: Undouble
        if word.endswith(("enn", "onn", "ett", "ell", "eill")):
            word = word[:-1]

        # STEP 6: Un-accent
        for i in range(1, len(word)):
            if word[-i] not in self.__vowels:
                i += 1
            else:
                if i != 1 and word[-i] in ("\xE9", "\xE8"):
                    word = "".join((word[:-i], "e", word[-i + 1 :]))
                break

        word = word.replace("I", "i").replace("U", "u").replace("Y", "y")

        return word

    def __rv_french(self, word, vowels):
        """
        Return the region RV that is used by the French stemmer.

        If the word begins with two vowels, RV is the region after
        the third letter. Otherwise, it is the region after the first
        vowel not at the beginning of the word, or the end of the word
        if these positions cannot be found. (Exceptionally, u'par',
        u'col' or u'tap' at the beginning of a word is also taken to
        define RV as the region to their right.)

        :param word: The French word whose region RV is determined.
        :type word: str or unicode
        :param vowels: The French vowels that are used to determine
                       the region RV.
        :type vowels: unicode
        :return: the region RV for the respective French word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of
               the subclass FrenchStemmer. It is not to be invoked directly!

        """
        rv = ""
        if len(word) >= 2:
            if word.startswith(("par", "col", "tap")) or (
                word[0] in vowels and word[1] in vowels
            ):
                rv = word[3:]
            else:
                for i in range(1, len(word)):
                    if word[i] in vowels:
                        rv = word[i + 1 :]
                        break

        return rv


class GermanStemmer(_StandardStemmer):

    """
    The German Snowball stemmer.

    :cvar __vowels: The German vowels.
    :type __vowels: unicode
    :cvar __s_ending: Letters that may directly appear before a word final 's'.
    :type __s_ending: unicode
    :cvar __st_ending: Letter that may directly appear before a word final 'st'.
    :type __st_ending: unicode
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :note: A detailed description of the German
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/german/stemmer.html

    """

    __vowels = "aeiouy\xE4\xF6\xFC"
    __s_ending = "bdfghklmnrt"
    __st_ending = "bdfghklmnt"

    __step1_suffixes = ("ern", "em", "er", "en", "es", "e", "s")
    __step2_suffixes = ("est", "en", "er", "st")
    __step3_suffixes = ("isch", "lich", "heit", "keit", "end", "ung", "ig", "ik")

    def stem(self, word):
        """
        Stem a German word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        word = word.replace("\xDF", "ss")

        # Every occurrence of 'u' and 'y'
        # between vowels is put into upper case.
        for i in range(1, len(word) - 1):
            if word[i - 1] in self.__vowels and word[i + 1] in self.__vowels:
                if word[i] == "u":
                    word = "".join((word[:i], "U", word[i + 1 :]))

                elif word[i] == "y":
                    word = "".join((word[:i], "Y", word[i + 1 :]))

        r1, r2 = self._r1r2_standard(word, self.__vowels)

        # R1 is adjusted so that the region before it
        # contains at least 3 letters.
        for i in range(1, len(word)):
            if word[i] not in self.__vowels and word[i - 1] in self.__vowels:
                if 3 > len(word[: i + 1]) > 0:
                    r1 = word[3:]
                elif len(word[: i + 1]) == 0:
                    return word
                break

        # STEP 1
        for suffix in self.__step1_suffixes:
            if r1.endswith(suffix):
                if (
                    suffix in ("en", "es", "e")
                    and word[-len(suffix) - 4 : -len(suffix)] == "niss"
                ):
                    word = word[: -len(suffix) - 1]
                    r1 = r1[: -len(suffix) - 1]
                    r2 = r2[: -len(suffix) - 1]

                elif suffix == "s":
                    if word[-2] in self.__s_ending:
                        word = word[:-1]
                        r1 = r1[:-1]
                        r2 = r2[:-1]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                break

        # STEP 2
        for suffix in self.__step2_suffixes:
            if r1.endswith(suffix):
                if suffix == "st":
                    if word[-3] in self.__st_ending and len(word[:-3]) >= 3:
                        word = word[:-2]
                        r1 = r1[:-2]
                        r2 = r2[:-2]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                break

        # STEP 3: Derivational suffixes
        for suffix in self.__step3_suffixes:
            if r2.endswith(suffix):
                if suffix in ("end", "ung"):
                    if (
                        "ig" in r2[-len(suffix) - 2 : -len(suffix)]
                        and "e" not in r2[-len(suffix) - 3 : -len(suffix) - 2]
                    ):
                        word = word[: -len(suffix) - 2]
                    else:
                        word = word[: -len(suffix)]

                elif (
                    suffix in ("ig", "ik", "isch")
                    and "e" not in r2[-len(suffix) - 1 : -len(suffix)]
                ):
                    word = word[: -len(suffix)]

                elif suffix in ("lich", "heit"):
                    if (
                        "er" in r1[-len(suffix) - 2 : -len(suffix)]
                        or "en" in r1[-len(suffix) - 2 : -len(suffix)]
                    ):
                        word = word[: -len(suffix) - 2]
                    else:
                        word = word[: -len(suffix)]

                elif suffix == "keit":
                    if "lich" in r2[-len(suffix) - 4 : -len(suffix)]:
                        word = word[: -len(suffix) - 4]

                    elif "ig" in r2[-len(suffix) - 2 : -len(suffix)]:
                        word = word[: -len(suffix) - 2]
                    else:
                        word = word[: -len(suffix)]
                break

        # Umlaut accents are removed and
        # 'u' and 'y' are put back into lower case.
        word = (
            word.replace("\xE4", "a")
            .replace("\xF6", "o")
            .replace("\xFC", "u")
            .replace("U", "u")
            .replace("Y", "y")
        )

        return word


class HungarianStemmer(_LanguageSpecificStemmer):

    """
    The Hungarian Snowball stemmer.

    :cvar __vowels: The Hungarian vowels.
    :type __vowels: unicode
    :cvar __digraphs: The Hungarian digraphs.
    :type __digraphs: tuple
    :cvar __double_consonants: The Hungarian double consonants.
    :type __double_consonants: tuple
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :cvar __step4_suffixes: Suffixes to be deleted in step 4 of the algorithm.
    :type __step4_suffixes: tuple
    :cvar __step5_suffixes: Suffixes to be deleted in step 5 of the algorithm.
    :type __step5_suffixes: tuple
    :cvar __step6_suffixes: Suffixes to be deleted in step 6 of the algorithm.
    :type __step6_suffixes: tuple
    :cvar __step7_suffixes: Suffixes to be deleted in step 7 of the algorithm.
    :type __step7_suffixes: tuple
    :cvar __step8_suffixes: Suffixes to be deleted in step 8 of the algorithm.
    :type __step8_suffixes: tuple
    :cvar __step9_suffixes: Suffixes to be deleted in step 9 of the algorithm.
    :type __step9_suffixes: tuple
    :note: A detailed description of the Hungarian
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/hungarian/stemmer.html

    """

    __vowels = "aeiou\xF6\xFC\xE1\xE9\xED\xF3\xF5\xFA\xFB"
    __digraphs = ("cs", "dz", "dzs", "gy", "ly", "ny", "ty", "zs")
    __double_consonants = (
        "bb",
        "cc",
        "ccs",
        "dd",
        "ff",
        "gg",
        "ggy",
        "jj",
        "kk",
        "ll",
        "lly",
        "mm",
        "nn",
        "nny",
        "pp",
        "rr",
        "ss",
        "ssz",
        "tt",
        "tty",
        "vv",
        "zz",
        "zzs",
    )

    __step1_suffixes = ("al", "el")
    __step2_suffixes = (
        "k\xE9ppen",
        "onk\xE9nt",
        "enk\xE9nt",
        "ank\xE9nt",
        "k\xE9pp",
        "k\xE9nt",
        "ban",
        "ben",
        "nak",
        "nek",
        "val",
        "vel",
        "t\xF3l",
        "t\xF5l",
        "r\xF3l",
        "r\xF5l",
        "b\xF3l",
        "b\xF5l",
        "hoz",
        "hez",
        "h\xF6z",
        "n\xE1l",
        "n\xE9l",
        "\xE9rt",
        "kor",
        "ba",
        "be",
        "ra",
        "re",
        "ig",
        "at",
        "et",
        "ot",
        "\xF6t",
        "ul",
        "\xFCl",
        "v\xE1",
        "v\xE9",
        "en",
        "on",
        "an",
        "\xF6n",
        "n",
        "t",
    )
    __step3_suffixes = ("\xE1nk\xE9nt", "\xE1n", "\xE9n")
    __step4_suffixes = (
        "astul",
        "est\xFCl",
        "\xE1stul",
        "\xE9st\xFCl",
        "stul",
        "st\xFCl",
    )
    __step5_suffixes = ("\xE1", "\xE9")
    __step6_suffixes = (
        "ok\xE9",
        "\xF6k\xE9",
        "ak\xE9",
        "ek\xE9",
        "\xE1k\xE9",
        "\xE1\xE9i",
        "\xE9k\xE9",
        "\xE9\xE9i",
        "k\xE9",
        "\xE9i",
        "\xE9\xE9",
        "\xE9",
    )
    __step7_suffixes = (
        "\xE1juk",
        "\xE9j\xFCk",
        "\xFCnk",
        "unk",
        "juk",
        "j\xFCk",
        "\xE1nk",
        "\xE9nk",
        "nk",
        "uk",
        "\xFCk",
        "em",
        "om",
        "am",
        "od",
        "ed",
        "ad",
        "\xF6d",
        "ja",
        "je",
        "\xE1m",
        "\xE1d",
        "\xE9m",
        "\xE9d",
        "m",
        "d",
        "a",
        "e",
        "o",
        "\xE1",
        "\xE9",
    )
    __step8_suffixes = (
        "jaitok",
        "jeitek",
        "jaink",
        "jeink",
        "aitok",
        "eitek",
        "\xE1itok",
        "\xE9itek",
        "jaim",
        "jeim",
        "jaid",
        "jeid",
        "eink",
        "aink",
        "itek",
        "jeik",
        "jaik",
        "\xE1ink",
        "\xE9ink",
        "aim",
        "eim",
        "aid",
        "eid",
        "jai",
        "jei",
        "ink",
        "aik",
        "eik",
        "\xE1im",
        "\xE1id",
        "\xE1ik",
        "\xE9im",
        "\xE9id",
        "\xE9ik",
        "im",
        "id",
        "ai",
        "ei",
        "ik",
        "\xE1i",
        "\xE9i",
        "i",
    )
    __step9_suffixes = ("\xE1k", "\xE9k", "\xF6k", "ok", "ek", "ak", "k")

    def stem(self, word):
        """
        Stem an Hungarian word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        r1 = self.__r1_hungarian(word, self.__vowels, self.__digraphs)

        # STEP 1: Remove instrumental case
        if r1.endswith(self.__step1_suffixes):
            for double_cons in self.__double_consonants:
                if word[-2 - len(double_cons) : -2] == double_cons:
                    word = "".join((word[:-4], word[-3]))

                    if r1[-2 - len(double_cons) : -2] == double_cons:
                        r1 = "".join((r1[:-4], r1[-3]))
                    break

        # STEP 2: Remove frequent cases
        for suffix in self.__step2_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]

                    if r1.endswith("\xE1"):
                        word = "".join((word[:-1], "a"))
                        r1 = suffix_replace(r1, "\xE1", "a")

                    elif r1.endswith("\xE9"):
                        word = "".join((word[:-1], "e"))
                        r1 = suffix_replace(r1, "\xE9", "e")
                break

        # STEP 3: Remove special cases
        for suffix in self.__step3_suffixes:
            if r1.endswith(suffix):
                if suffix == "\xE9n":
                    word = suffix_replace(word, suffix, "e")
                    r1 = suffix_replace(r1, suffix, "e")
                else:
                    word = suffix_replace(word, suffix, "a")
                    r1 = suffix_replace(r1, suffix, "a")
                break

        # STEP 4: Remove other cases
        for suffix in self.__step4_suffixes:
            if r1.endswith(suffix):
                if suffix == "\xE1stul":
                    word = suffix_replace(word, suffix, "a")
                    r1 = suffix_replace(r1, suffix, "a")

                elif suffix == "\xE9st\xFCl":
                    word = suffix_replace(word, suffix, "e")
                    r1 = suffix_replace(r1, suffix, "e")
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                break

        # STEP 5: Remove factive case
        for suffix in self.__step5_suffixes:
            if r1.endswith(suffix):
                for double_cons in self.__double_consonants:
                    if word[-1 - len(double_cons) : -1] == double_cons:
                        word = "".join((word[:-3], word[-2]))

                        if r1[-1 - len(double_cons) : -1] == double_cons:
                            r1 = "".join((r1[:-3], r1[-2]))
                        break

        # STEP 6: Remove owned
        for suffix in self.__step6_suffixes:
            if r1.endswith(suffix):
                if suffix in ("\xE1k\xE9", "\xE1\xE9i"):
                    word = suffix_replace(word, suffix, "a")
                    r1 = suffix_replace(r1, suffix, "a")

                elif suffix in ("\xE9k\xE9", "\xE9\xE9i", "\xE9\xE9"):
                    word = suffix_replace(word, suffix, "e")
                    r1 = suffix_replace(r1, suffix, "e")
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                break

        # STEP 7: Remove singular owner suffixes
        for suffix in self.__step7_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix in ("\xE1nk", "\xE1juk", "\xE1m", "\xE1d", "\xE1"):
                        word = suffix_replace(word, suffix, "a")
                        r1 = suffix_replace(r1, suffix, "a")

                    elif suffix in ("\xE9nk", "\xE9j\xFCk", "\xE9m", "\xE9d", "\xE9"):
                        word = suffix_replace(word, suffix, "e")
                        r1 = suffix_replace(r1, suffix, "e")
                    else:
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                break

        # STEP 8: Remove plural owner suffixes
        for suffix in self.__step8_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix in (
                        "\xE1im",
                        "\xE1id",
                        "\xE1i",
                        "\xE1ink",
                        "\xE1itok",
                        "\xE1ik",
                    ):
                        word = suffix_replace(word, suffix, "a")
                        r1 = suffix_replace(r1, suffix, "a")

                    elif suffix in (
                        "\xE9im",
                        "\xE9id",
                        "\xE9i",
                        "\xE9ink",
                        "\xE9itek",
                        "\xE9ik",
                    ):
                        word = suffix_replace(word, suffix, "e")
                        r1 = suffix_replace(r1, suffix, "e")
                    else:
                        word = word[: -len(suffix)]
                        r1 = r1[: -len(suffix)]
                break

        # STEP 9: Remove plural suffixes
        for suffix in self.__step9_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix == "\xE1k":
                        word = suffix_replace(word, suffix, "a")
                    elif suffix == "\xE9k":
                        word = suffix_replace(word, suffix, "e")
                    else:
                        word = word[: -len(suffix)]
                break

        return word

    def __r1_hungarian(self, word, vowels, digraphs):
        """
        Return the region R1 that is used by the Hungarian stemmer.

        If the word begins with a vowel, R1 is defined as the region
        after the first consonant or digraph (= two letters stand for
        one phoneme) in the word. If the word begins with a consonant,
        it is defined as the region after the first vowel in the word.
        If the word does not contain both a vowel and consonant, R1
        is the null region at the end of the word.

        :param word: The Hungarian word whose region R1 is determined.
        :type word: str or unicode
        :param vowels: The Hungarian vowels that are used to determine
                       the region R1.
        :type vowels: unicode
        :param digraphs: The digraphs that are used to determine the
                         region R1.
        :type digraphs: tuple
        :return: the region R1 for the respective word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of the subclass
               HungarianStemmer. It is not to be invoked directly!

        """
        r1 = ""
        if word[0] in vowels:
            for digraph in digraphs:
                if digraph in word[1:]:
                    r1 = word[word.index(digraph[-1]) + 1 :]
                    return r1

            for i in range(1, len(word)):
                if word[i] not in vowels:
                    r1 = word[i + 1 :]
                    break
        else:
            for i in range(1, len(word)):
                if word[i] in vowels:
                    r1 = word[i + 1 :]
                    break

        return r1


class ItalianStemmer(_StandardStemmer):

    """
    The Italian Snowball stemmer.

    :cvar __vowels: The Italian vowels.
    :type __vowels: unicode
    :cvar __step0_suffixes: Suffixes to be deleted in step 0 of the algorithm.
    :type __step0_suffixes: tuple
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :note: A detailed description of the Italian
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/italian/stemmer.html

    """

    __vowels = "aeiou\xE0\xE8\xEC\xF2\xF9"
    __step0_suffixes = (
        "gliela",
        "gliele",
        "glieli",
        "glielo",
        "gliene",
        "sene",
        "mela",
        "mele",
        "meli",
        "melo",
        "mene",
        "tela",
        "tele",
        "teli",
        "telo",
        "tene",
        "cela",
        "cele",
        "celi",
        "celo",
        "cene",
        "vela",
        "vele",
        "veli",
        "velo",
        "vene",
        "gli",
        "ci",
        "la",
        "le",
        "li",
        "lo",
        "mi",
        "ne",
        "si",
        "ti",
        "vi",
    )
    __step1_suffixes = (
        "atrice",
        "atrici",
        "azione",
        "azioni",
        "uzione",
        "uzioni",
        "usione",
        "usioni",
        "amento",
        "amenti",
        "imento",
        "imenti",
        "amente",
        "abile",
        "abili",
        "ibile",
        "ibili",
        "mente",
        "atore",
        "atori",
        "logia",
        "logie",
        "anza",
        "anze",
        "iche",
        "ichi",
        "ismo",
        "ismi",
        "ista",
        "iste",
        "isti",
        "ist\xE0",
        "ist\xE8",
        "ist\xEC",
        "ante",
        "anti",
        "enza",
        "enze",
        "ico",
        "ici",
        "ica",
        "ice",
        "oso",
        "osi",
        "osa",
        "ose",
        "it\xE0",
        "ivo",
        "ivi",
        "iva",
        "ive",
    )
    __step2_suffixes = (
        "erebbero",
        "irebbero",
        "assero",
        "assimo",
        "eranno",
        "erebbe",
        "eremmo",
        "ereste",
        "eresti",
        "essero",
        "iranno",
        "irebbe",
        "iremmo",
        "ireste",
        "iresti",
        "iscano",
        "iscono",
        "issero",
        "arono",
        "avamo",
        "avano",
        "avate",
        "eremo",
        "erete",
        "erono",
        "evamo",
        "evano",
        "evate",
        "iremo",
        "irete",
        "irono",
        "ivamo",
        "ivano",
        "ivate",
        "ammo",
        "ando",
        "asse",
        "assi",
        "emmo",
        "enda",
        "ende",
        "endi",
        "endo",
        "erai",
        "erei",
        "Yamo",
        "iamo",
        "immo",
        "irai",
        "irei",
        "isca",
        "isce",
        "isci",
        "isco",
        "ano",
        "are",
        "ata",
        "ate",
        "ati",
        "ato",
        "ava",
        "avi",
        "avo",
        "er\xE0",
        "ere",
        "er\xF2",
        "ete",
        "eva",
        "evi",
        "evo",
        "ir\xE0",
        "ire",
        "ir\xF2",
        "ita",
        "ite",
        "iti",
        "ito",
        "iva",
        "ivi",
        "ivo",
        "ono",
        "uta",
        "ute",
        "uti",
        "uto",
        "ar",
        "ir",
    )

    def stem(self, word):
        """
        Stem an Italian word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        step1_success = False

        # All acute accents are replaced by grave accents.
        word = (
            word.replace("\xE1", "\xE0")
            .replace("\xE9", "\xE8")
            .replace("\xED", "\xEC")
            .replace("\xF3", "\xF2")
            .replace("\xFA", "\xF9")
        )

        # Every occurrence of 'u' after 'q'
        # is put into upper case.
        for i in range(1, len(word)):
            if word[i - 1] == "q" and word[i] == "u":
                word = "".join((word[:i], "U", word[i + 1 :]))

        # Every occurrence of 'u' and 'i'
        # between vowels is put into upper case.
        for i in range(1, len(word) - 1):
            if word[i - 1] in self.__vowels and word[i + 1] in self.__vowels:
                if word[i] == "u":
                    word = "".join((word[:i], "U", word[i + 1 :]))

                elif word[i] == "i":
                    word = "".join((word[:i], "I", word[i + 1 :]))

        r1, r2 = self._r1r2_standard(word, self.__vowels)
        rv = self._rv_standard(word, self.__vowels)

        # STEP 0: Attached pronoun
        for suffix in self.__step0_suffixes:
            if rv.endswith(suffix):
                if rv[-len(suffix) - 4 : -len(suffix)] in ("ando", "endo"):
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    rv = rv[: -len(suffix)]

                elif rv[-len(suffix) - 2 : -len(suffix)] in ("ar", "er", "ir"):
                    word = suffix_replace(word, suffix, "e")
                    r1 = suffix_replace(r1, suffix, "e")
                    r2 = suffix_replace(r2, suffix, "e")
                    rv = suffix_replace(rv, suffix, "e")
                break

        # STEP 1: Standard suffix removal
        for suffix in self.__step1_suffixes:
            if word.endswith(suffix):
                if suffix == "amente" and r1.endswith(suffix):
                    step1_success = True
                    word = word[:-6]
                    r2 = r2[:-6]
                    rv = rv[:-6]

                    if r2.endswith("iv"):
                        word = word[:-2]
                        r2 = r2[:-2]
                        rv = rv[:-2]

                        if r2.endswith("at"):
                            word = word[:-2]
                            rv = rv[:-2]

                    elif r2.endswith(("os", "ic")):
                        word = word[:-2]
                        rv = rv[:-2]

                    elif r2.endswith("abil"):
                        word = word[:-4]
                        rv = rv[:-4]

                elif suffix in ("amento", "amenti", "imento", "imenti") and rv.endswith(
                    suffix
                ):
                    step1_success = True
                    word = word[:-6]
                    rv = rv[:-6]

                elif r2.endswith(suffix):
                    step1_success = True
                    if suffix in ("azione", "azioni", "atore", "atori"):
                        word = word[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        rv = rv[: -len(suffix)]

                        if r2.endswith("ic"):
                            word = word[:-2]
                            rv = rv[:-2]

                    elif suffix in ("logia", "logie"):
                        word = word[:-2]
                        rv = word[:-2]

                    elif suffix in ("uzione", "uzioni", "usione", "usioni"):
                        word = word[:-5]
                        rv = rv[:-5]

                    elif suffix in ("enza", "enze"):
                        word = suffix_replace(word, suffix, "te")
                        rv = suffix_replace(rv, suffix, "te")

                    elif suffix == "it\xE0":
                        word = word[:-3]
                        r2 = r2[:-3]
                        rv = rv[:-3]

                        if r2.endswith(("ic", "iv")):
                            word = word[:-2]
                            rv = rv[:-2]

                        elif r2.endswith("abil"):
                            word = word[:-4]
                            rv = rv[:-4]

                    elif suffix in ("ivo", "ivi", "iva", "ive"):
                        word = word[:-3]
                        r2 = r2[:-3]
                        rv = rv[:-3]

                        if r2.endswith("at"):
                            word = word[:-2]
                            r2 = r2[:-2]
                            rv = rv[:-2]

                            if r2.endswith("ic"):
                                word = word[:-2]
                                rv = rv[:-2]
                    else:
                        word = word[: -len(suffix)]
                        rv = rv[: -len(suffix)]
                break

        # STEP 2: Verb suffixes
        if not step1_success:
            for suffix in self.__step2_suffixes:
                if rv.endswith(suffix):
                    word = word[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    break

        # STEP 3a
        if rv.endswith(("a", "e", "i", "o", "\xE0", "\xE8", "\xEC", "\xF2")):
            word = word[:-1]
            rv = rv[:-1]

            if rv.endswith("i"):
                word = word[:-1]
                rv = rv[:-1]

        # STEP 3b
        if rv.endswith(("ch", "gh")):
            word = word[:-1]

        word = word.replace("I", "i").replace("U", "u")

        return word


class NorwegianStemmer(_ScandinavianStemmer):

    """
    The Norwegian Snowball stemmer.

    :cvar __vowels: The Norwegian vowels.
    :type __vowels: unicode
    :cvar __s_ending: Letters that may directly appear before a word final 's'.
    :type __s_ending: unicode
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :note: A detailed description of the Norwegian
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/norwegian/stemmer.html

    """

    __vowels = "aeiouy\xE6\xE5\xF8"
    __s_ending = "bcdfghjlmnoprtvyz"
    __step1_suffixes = (
        "hetenes",
        "hetene",
        "hetens",
        "heter",
        "heten",
        "endes",
        "ande",
        "ende",
        "edes",
        "enes",
        "erte",
        "ede",
        "ane",
        "ene",
        "ens",
        "ers",
        "ets",
        "het",
        "ast",
        "ert",
        "en",
        "ar",
        "er",
        "as",
        "es",
        "et",
        "a",
        "e",
        "s",
    )

    __step2_suffixes = ("dt", "vt")

    __step3_suffixes = (
        "hetslov",
        "eleg",
        "elig",
        "elov",
        "slov",
        "leg",
        "eig",
        "lig",
        "els",
        "lov",
        "ig",
    )

    def stem(self, word):
        """
        Stem a Norwegian word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        r1 = self._r1_scandinavian(word, self.__vowels)

        # STEP 1
        for suffix in self.__step1_suffixes:
            if r1.endswith(suffix):
                if suffix in ("erte", "ert"):
                    word = suffix_replace(word, suffix, "er")
                    r1 = suffix_replace(r1, suffix, "er")

                elif suffix == "s":
                    if word[-2] in self.__s_ending or (
                        word[-2] == "k" and word[-3] not in self.__vowels
                    ):
                        word = word[:-1]
                        r1 = r1[:-1]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                break

        # STEP 2
        for suffix in self.__step2_suffixes:
            if r1.endswith(suffix):
                word = word[:-1]
                r1 = r1[:-1]
                break

        # STEP 3
        for suffix in self.__step3_suffixes:
            if r1.endswith(suffix):
                word = word[: -len(suffix)]
                break

        return word


class PortugueseStemmer(_StandardStemmer):

    """
    The Portuguese Snowball stemmer.

    :cvar __vowels: The Portuguese vowels.
    :type __vowels: unicode
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step4_suffixes: Suffixes to be deleted in step 4 of the algorithm.
    :type __step4_suffixes: tuple
    :note: A detailed description of the Portuguese
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/portuguese/stemmer.html

    """

    __vowels = "aeiou\xE1\xE9\xED\xF3\xFA\xE2\xEA\xF4"
    __step1_suffixes = (
        "amentos",
        "imentos",
        "uço~es",
        "amento",
        "imento",
        "adoras",
        "adores",
        "a\xE7o~es",
        "logias",
        "\xEAncias",
        "amente",
        "idades",
        "an\xE7as",
        "ismos",
        "istas",
        "adora",
        "a\xE7a~o",
        "antes",
        "\xE2ncia",
        "logia",
        "uça~o",
        "\xEAncia",
        "mente",
        "idade",
        "an\xE7a",
        "ezas",
        "icos",
        "icas",
        "ismo",
        "\xE1vel",
        "\xEDvel",
        "ista",
        "osos",
        "osas",
        "ador",
        "ante",
        "ivas",
        "ivos",
        "iras",
        "eza",
        "ico",
        "ica",
        "oso",
        "osa",
        "iva",
        "ivo",
        "ira",
    )
    __step2_suffixes = (
        "ar\xEDamos",
        "er\xEDamos",
        "ir\xEDamos",
        "\xE1ssemos",
        "\xEAssemos",
        "\xEDssemos",
        "ar\xEDeis",
        "er\xEDeis",
        "ir\xEDeis",
        "\xE1sseis",
        "\xE9sseis",
        "\xEDsseis",
        "\xE1ramos",
        "\xE9ramos",
        "\xEDramos",
        "\xE1vamos",
        "aremos",
        "eremos",
        "iremos",
        "ariam",
        "eriam",
        "iriam",
        "assem",
        "essem",
        "issem",
        "ara~o",
        "era~o",
        "ira~o",
        "arias",
        "erias",
        "irias",
        "ardes",
        "erdes",
        "irdes",
        "asses",
        "esses",
        "isses",
        "astes",
        "estes",
        "istes",
        "\xE1reis",
        "areis",
        "\xE9reis",
        "ereis",
        "\xEDreis",
        "ireis",
        "\xE1veis",
        "\xEDamos",
        "armos",
        "ermos",
        "irmos",
        "aria",
        "eria",
        "iria",
        "asse",
        "esse",
        "isse",
        "aste",
        "este",
        "iste",
        "arei",
        "erei",
        "irei",
        "aram",
        "eram",
        "iram",
        "avam",
        "arem",
        "erem",
        "irem",
        "ando",
        "endo",
        "indo",
        "adas",
        "idas",
        "ar\xE1s",
        "aras",
        "er\xE1s",
        "eras",
        "ir\xE1s",
        "avas",
        "ares",
        "eres",
        "ires",
        "\xEDeis",
        "ados",
        "idos",
        "\xE1mos",
        "amos",
        "emos",
        "imos",
        "iras",
        "ada",
        "ida",
        "ar\xE1",
        "ara",
        "er\xE1",
        "era",
        "ir\xE1",
        "ava",
        "iam",
        "ado",
        "ido",
        "ias",
        "ais",
        "eis",
        "ira",
        "ia",
        "ei",
        "am",
        "em",
        "ar",
        "er",
        "ir",
        "as",
        "es",
        "is",
        "eu",
        "iu",
        "ou",
    )
    __step4_suffixes = ("os", "a", "i", "o", "\xE1", "\xED", "\xF3")

    def stem(self, word):
        """
        Stem a Portuguese word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        step1_success = False
        step2_success = False

        word = (
            word.replace("\xE3", "a~")
            .replace("\xF5", "o~")
            .replace("q\xFC", "qu")
            .replace("g\xFC", "gu")
        )

        r1, r2 = self._r1r2_standard(word, self.__vowels)
        rv = self._rv_standard(word, self.__vowels)

        # STEP 1: Standard suffix removal
        for suffix in self.__step1_suffixes:
            if word.endswith(suffix):
                if suffix == "amente" and r1.endswith(suffix):
                    step1_success = True

                    word = word[:-6]
                    r2 = r2[:-6]
                    rv = rv[:-6]

                    if r2.endswith("iv"):
                        word = word[:-2]
                        r2 = r2[:-2]
                        rv = rv[:-2]

                        if r2.endswith("at"):
                            word = word[:-2]
                            rv = rv[:-2]

                    elif r2.endswith(("os", "ic", "ad")):
                        word = word[:-2]
                        rv = rv[:-2]

                elif (
                    suffix in ("ira", "iras")
                    and rv.endswith(suffix)
                    and word[-len(suffix) - 1 : -len(suffix)] == "e"
                ):
                    step1_success = True

                    word = suffix_replace(word, suffix, "ir")
                    rv = suffix_replace(rv, suffix, "ir")

                elif r2.endswith(suffix):
                    step1_success = True

                    if suffix in ("logia", "logias"):
                        word = suffix_replace(word, suffix, "log")
                        rv = suffix_replace(rv, suffix, "log")

                    elif suffix in ("uça~o", "uço~es"):
                        word = suffix_replace(word, suffix, "u")
                        rv = suffix_replace(rv, suffix, "u")

                    elif suffix in ("\xEAncia", "\xEAncias"):
                        word = suffix_replace(word, suffix, "ente")
                        rv = suffix_replace(rv, suffix, "ente")

                    elif suffix == "mente":
                        word = word[:-5]
                        r2 = r2[:-5]
                        rv = rv[:-5]

                        if r2.endswith(("ante", "avel", "ivel")):
                            word = word[:-4]
                            rv = rv[:-4]

                    elif suffix in ("idade", "idades"):
                        word = word[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        rv = rv[: -len(suffix)]

                        if r2.endswith(("ic", "iv")):
                            word = word[:-2]
                            rv = rv[:-2]

                        elif r2.endswith("abil"):
                            word = word[:-4]
                            rv = rv[:-4]

                    elif suffix in ("iva", "ivo", "ivas", "ivos"):
                        word = word[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        rv = rv[: -len(suffix)]

                        if r2.endswith("at"):
                            word = word[:-2]
                            rv = rv[:-2]
                    else:
                        word = word[: -len(suffix)]
                        rv = rv[: -len(suffix)]
                break

        # STEP 2: Verb suffixes
        if not step1_success:
            for suffix in self.__step2_suffixes:
                if rv.endswith(suffix):
                    step2_success = True

                    word = word[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    break

        # STEP 3
        if step1_success or step2_success:
            if rv.endswith("i") and word[-2] == "c":
                word = word[:-1]
                rv = rv[:-1]

        ### STEP 4: Residual suffix
        if not step1_success and not step2_success:
            for suffix in self.__step4_suffixes:
                if rv.endswith(suffix):
                    word = word[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    break

        # STEP 5
        if rv.endswith(("e", "\xE9", "\xEA")):
            word = word[:-1]
            rv = rv[:-1]

            if (word.endswith("gu") and rv.endswith("u")) or (
                word.endswith("ci") and rv.endswith("i")
            ):
                word = word[:-1]

        elif word.endswith("\xE7"):
            word = suffix_replace(word, "\xE7", "c")

        word = word.replace("a~", "\xE3").replace("o~", "\xF5")

        return word


class RomanianStemmer(_StandardStemmer):

    """
    The Romanian Snowball stemmer.

    :cvar __vowels: The Romanian vowels.
    :type __vowels: unicode
    :cvar __step0_suffixes: Suffixes to be deleted in step 0 of the algorithm.
    :type __step0_suffixes: tuple
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :note: A detailed description of the Romanian
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/romanian/stemmer.html

    """

    __vowels = "aeiou\u0103\xE2\xEE"
    __step0_suffixes = (
        "iilor",
        "ului",
        "elor",
        "iile",
        "ilor",
        "atei",
        "a\u0163ie",
        "a\u0163ia",
        "aua",
        "ele",
        "iua",
        "iei",
        "ile",
        "ul",
        "ea",
        "ii",
    )
    __step1_suffixes = (
        "abilitate",
        "abilitati",
        "abilit\u0103\u0163i",
        "ibilitate",
        "abilit\u0103i",
        "ivitate",
        "ivitati",
        "ivit\u0103\u0163i",
        "icitate",
        "icitati",
        "icit\u0103\u0163i",
        "icatori",
        "ivit\u0103i",
        "icit\u0103i",
        "icator",
        "a\u0163iune",
        "atoare",
        "\u0103toare",
        "i\u0163iune",
        "itoare",
        "iciva",
        "icive",
        "icivi",
        "iciv\u0103",
        "icala",
        "icale",
        "icali",
        "ical\u0103",
        "ativa",
        "ative",
        "ativi",
        "ativ\u0103",
        "atori",
        "\u0103tori",
        "itiva",
        "itive",
        "itivi",
        "itiv\u0103",
        "itori",
        "iciv",
        "ical",
        "ativ",
        "ator",
        "\u0103tor",
        "itiv",
        "itor",
    )
    __step2_suffixes = (
        "abila",
        "abile",
        "abili",
        "abil\u0103",
        "ibila",
        "ibile",
        "ibili",
        "ibil\u0103",
        "atori",
        "itate",
        "itati",
        "it\u0103\u0163i",
        "abil",
        "ibil",
        "oasa",
        "oas\u0103",
        "oase",
        "anta",
        "ante",
        "anti",
        "ant\u0103",
        "ator",
        "it\u0103i",
        "iune",
        "iuni",
        "isme",
        "ista",
        "iste",
        "isti",
        "ist\u0103",
        "i\u015Fti",
        "ata",
        "at\u0103",
        "ati",
        "ate",
        "uta",
        "ut\u0103",
        "uti",
        "ute",
        "ita",
        "it\u0103",
        "iti",
        "ite",
        "ica",
        "ice",
        "ici",
        "ic\u0103",
        "osi",
        "o\u015Fi",
        "ant",
        "iva",
        "ive",
        "ivi",
        "iv\u0103",
        "ism",
        "ist",
        "at",
        "ut",
        "it",
        "ic",
        "os",
        "iv",
    )
    __step3_suffixes = (
        "seser\u0103\u0163i",
        "aser\u0103\u0163i",
        "iser\u0103\u0163i",
        "\xE2ser\u0103\u0163i",
        "user\u0103\u0163i",
        "seser\u0103m",
        "aser\u0103m",
        "iser\u0103m",
        "\xE2ser\u0103m",
        "user\u0103m",
        "ser\u0103\u0163i",
        "sese\u015Fi",
        "seser\u0103",
        "easc\u0103",
        "ar\u0103\u0163i",
        "ur\u0103\u0163i",
        "ir\u0103\u0163i",
        "\xE2r\u0103\u0163i",
        "ase\u015Fi",
        "aser\u0103",
        "ise\u015Fi",
        "iser\u0103",
        "\xe2se\u015Fi",
        "\xE2ser\u0103",
        "use\u015Fi",
        "user\u0103",
        "ser\u0103m",
        "sesem",
        "indu",
        "\xE2ndu",
        "eaz\u0103",
        "e\u015Fti",
        "e\u015Fte",
        "\u0103\u015Fti",
        "\u0103\u015Fte",
        "ea\u0163i",
        "ia\u0163i",
        "ar\u0103m",
        "ur\u0103m",
        "ir\u0103m",
        "\xE2r\u0103m",
        "asem",
        "isem",
        "\xE2sem",
        "usem",
        "se\u015Fi",
        "ser\u0103",
        "sese",
        "are",
        "ere",
        "ire",
        "\xE2re",
        "ind",
        "\xE2nd",
        "eze",
        "ezi",
        "esc",
        "\u0103sc",
        "eam",
        "eai",
        "eau",
        "iam",
        "iai",
        "iau",
        "a\u015Fi",
        "ar\u0103",
        "u\u015Fi",
        "ur\u0103",
        "i\u015Fi",
        "ir\u0103",
        "\xE2\u015Fi",
        "\xe2r\u0103",
        "ase",
        "ise",
        "\xE2se",
        "use",
        "a\u0163i",
        "e\u0163i",
        "i\u0163i",
        "\xe2\u0163i",
        "sei",
        "ez",
        "am",
        "ai",
        "au",
        "ea",
        "ia",
        "ui",
        "\xE2i",
        "\u0103m",
        "em",
        "im",
        "\xE2m",
        "se",
    )

    def stem(self, word):
        """
        Stem a Romanian word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        step1_success = False
        step2_success = False

        for i in range(1, len(word) - 1):
            if word[i - 1] in self.__vowels and word[i + 1] in self.__vowels:
                if word[i] == "u":
                    word = "".join((word[:i], "U", word[i + 1 :]))

                elif word[i] == "i":
                    word = "".join((word[:i], "I", word[i + 1 :]))

        r1, r2 = self._r1r2_standard(word, self.__vowels)
        rv = self._rv_standard(word, self.__vowels)

        # STEP 0: Removal of plurals and other simplifications
        for suffix in self.__step0_suffixes:
            if word.endswith(suffix):
                if suffix in r1:
                    if suffix in ("ul", "ului"):
                        word = word[: -len(suffix)]

                        if suffix in rv:
                            rv = rv[: -len(suffix)]
                        else:
                            rv = ""

                    elif (
                        suffix == "aua"
                        or suffix == "atei"
                        or (suffix == "ile" and word[-5:-3] != "ab")
                    ):
                        word = word[:-2]

                    elif suffix in ("ea", "ele", "elor"):
                        word = suffix_replace(word, suffix, "e")

                        if suffix in rv:
                            rv = suffix_replace(rv, suffix, "e")
                        else:
                            rv = ""

                    elif suffix in ("ii", "iua", "iei", "iile", "iilor", "ilor"):
                        word = suffix_replace(word, suffix, "i")

                        if suffix in rv:
                            rv = suffix_replace(rv, suffix, "i")
                        else:
                            rv = ""

                    elif suffix in ("a\u0163ie", "a\u0163ia"):
                        word = word[:-1]
                break

        # STEP 1: Reduction of combining suffixes
        while True:

            replacement_done = False

            for suffix in self.__step1_suffixes:
                if word.endswith(suffix):
                    if suffix in r1:
                        step1_success = True
                        replacement_done = True

                        if suffix in (
                            "abilitate",
                            "abilitati",
                            "abilit\u0103i",
                            "abilit\u0103\u0163i",
                        ):
                            word = suffix_replace(word, suffix, "abil")

                        elif suffix == "ibilitate":
                            word = word[:-5]

                        elif suffix in (
                            "ivitate",
                            "ivitati",
                            "ivit\u0103i",
                            "ivit\u0103\u0163i",
                        ):
                            word = suffix_replace(word, suffix, "iv")

                        elif suffix in (
                            "icitate",
                            "icitati",
                            "icit\u0103i",
                            "icit\u0103\u0163i",
                            "icator",
                            "icatori",
                            "iciv",
                            "iciva",
                            "icive",
                            "icivi",
                            "iciv\u0103",
                            "ical",
                            "icala",
                            "icale",
                            "icali",
                            "ical\u0103",
                        ):
                            word = suffix_replace(word, suffix, "ic")

                        elif suffix in (
                            "ativ",
                            "ativa",
                            "ative",
                            "ativi",
                            "ativ\u0103",
                            "a\u0163iune",
                            "atoare",
                            "ator",
                            "atori",
                            "\u0103toare",
                            "\u0103tor",
                            "\u0103tori",
                        ):
                            word = suffix_replace(word, suffix, "at")

                            if suffix in r2:
                                r2 = suffix_replace(r2, suffix, "at")

                        elif suffix in (
                            "itiv",
                            "itiva",
                            "itive",
                            "itivi",
                            "itiv\u0103",
                            "i\u0163iune",
                            "itoare",
                            "itor",
                            "itori",
                        ):
                            word = suffix_replace(word, suffix, "it")

                            if suffix in r2:
                                r2 = suffix_replace(r2, suffix, "it")
                    else:
                        step1_success = False
                    break

            if not replacement_done:
                break

        # STEP 2: Removal of standard suffixes
        for suffix in self.__step2_suffixes:
            if word.endswith(suffix):
                if suffix in r2:
                    step2_success = True

                    if suffix in ("iune", "iuni"):
                        if word[-5] == "\u0163":
                            word = "".join((word[:-5], "t"))

                    elif suffix in (
                        "ism",
                        "isme",
                        "ist",
                        "ista",
                        "iste",
                        "isti",
                        "ist\u0103",
                        "i\u015Fti",
                    ):
                        word = suffix_replace(word, suffix, "ist")

                    else:
                        word = word[: -len(suffix)]
                break

        # STEP 3: Removal of verb suffixes
        if not step1_success and not step2_success:
            for suffix in self.__step3_suffixes:
                if word.endswith(suffix):
                    if suffix in rv:
                        if suffix in (
                            "seser\u0103\u0163i",
                            "seser\u0103m",
                            "ser\u0103\u0163i",
                            "sese\u015Fi",
                            "seser\u0103",
                            "ser\u0103m",
                            "sesem",
                            "se\u015Fi",
                            "ser\u0103",
                            "sese",
                            "a\u0163i",
                            "e\u0163i",
                            "i\u0163i",
                            "\xE2\u0163i",
                            "sei",
                            "\u0103m",
                            "em",
                            "im",
                            "\xE2m",
                            "se",
                        ):
                            word = word[: -len(suffix)]
                            rv = rv[: -len(suffix)]
                        else:
                            if (
                                not rv.startswith(suffix)
                                and rv[rv.index(suffix) - 1] not in "aeio\u0103\xE2\xEE"
                            ):
                                word = word[: -len(suffix)]
                        break

        # STEP 4: Removal of final vowel
        for suffix in ("ie", "a", "e", "i", "\u0103"):
            if word.endswith(suffix):
                if suffix in rv:
                    word = word[: -len(suffix)]
                break

        word = word.replace("I", "i").replace("U", "u")

        return word


class RussianStemmer(_LanguageSpecificStemmer):

    """
    The Russian Snowball stemmer.

    :cvar __perfective_gerund_suffixes: Suffixes to be deleted.
    :type __perfective_gerund_suffixes: tuple
    :cvar __adjectival_suffixes: Suffixes to be deleted.
    :type __adjectival_suffixes: tuple
    :cvar __reflexive_suffixes: Suffixes to be deleted.
    :type __reflexive_suffixes: tuple
    :cvar __verb_suffixes: Suffixes to be deleted.
    :type __verb_suffixes: tuple
    :cvar __noun_suffixes: Suffixes to be deleted.
    :type __noun_suffixes: tuple
    :cvar __superlative_suffixes: Suffixes to be deleted.
    :type __superlative_suffixes: tuple
    :cvar __derivational_suffixes: Suffixes to be deleted.
    :type __derivational_suffixes: tuple
    :note: A detailed description of the Russian
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/russian/stemmer.html

    """

    __perfective_gerund_suffixes = (
        "ivshis'",
        "yvshis'",
        "vshis'",
        "ivshi",
        "yvshi",
        "vshi",
        "iv",
        "yv",
        "v",
    )
    __adjectival_suffixes = (
        "ui^ushchi^ui^u",
        "ui^ushchi^ai^a",
        "ui^ushchimi",
        "ui^ushchymi",
        "ui^ushchego",
        "ui^ushchogo",
        "ui^ushchemu",
        "ui^ushchomu",
        "ui^ushchikh",
        "ui^ushchykh",
        "ui^ushchui^u",
        "ui^ushchaia",
        "ui^ushchoi^u",
        "ui^ushchei^u",
        "i^ushchi^ui^u",
        "i^ushchi^ai^a",
        "ui^ushchee",
        "ui^ushchie",
        "ui^ushchye",
        "ui^ushchoe",
        "ui^ushchei`",
        "ui^ushchii`",
        "ui^ushchyi`",
        "ui^ushchoi`",
        "ui^ushchem",
        "ui^ushchim",
        "ui^ushchym",
        "ui^ushchom",
        "i^ushchimi",
        "i^ushchymi",
        "i^ushchego",
        "i^ushchogo",
        "i^ushchemu",
        "i^ushchomu",
        "i^ushchikh",
        "i^ushchykh",
        "i^ushchui^u",
        "i^ushchai^a",
        "i^ushchoi^u",
        "i^ushchei^u",
        "i^ushchee",
        "i^ushchie",
        "i^ushchye",
        "i^ushchoe",
        "i^ushchei`",
        "i^ushchii`",
        "i^ushchyi`",
        "i^ushchoi`",
        "i^ushchem",
        "i^ushchim",
        "i^ushchym",
        "i^ushchom",
        "shchi^ui^u",
        "shchi^ai^a",
        "ivshi^ui^u",
        "ivshi^ai^a",
        "yvshi^ui^u",
        "yvshi^ai^a",
        "shchimi",
        "shchymi",
        "shchego",
        "shchogo",
        "shchemu",
        "shchomu",
        "shchikh",
        "shchykh",
        "shchui^u",
        "shchai^a",
        "shchoi^u",
        "shchei^u",
        "ivshimi",
        "ivshymi",
        "ivshego",
        "ivshogo",
        "ivshemu",
        "ivshomu",
        "ivshikh",
        "ivshykh",
        "ivshui^u",
        "ivshai^a",
        "ivshoi^u",
        "ivshei^u",
        "yvshimi",
        "yvshymi",
        "yvshego",
        "yvshogo",
        "yvshemu",
        "yvshomu",
        "yvshikh",
        "yvshykh",
        "yvshui^u",
        "yvshai^a",
        "yvshoi^u",
        "yvshei^u",
        "vshi^ui^u",
        "vshi^ai^a",
        "shchee",
        "shchie",
        "shchye",
        "shchoe",
        "shchei`",
        "shchii`",
        "shchyi`",
        "shchoi`",
        "shchem",
        "shchim",
        "shchym",
        "shchom",
        "ivshee",
        "ivshie",
        "ivshye",
        "ivshoe",
        "ivshei`",
        "ivshii`",
        "ivshyi`",
        "ivshoi`",
        "ivshem",
        "ivshim",
        "ivshym",
        "ivshom",
        "yvshee",
        "yvshie",
        "yvshye",
        "yvshoe",
        "yvshei`",
        "yvshii`",
        "yvshyi`",
        "yvshoi`",
        "yvshem",
        "yvshim",
        "yvshym",
        "yvshom",
        "vshimi",
        "vshymi",
        "vshego",
        "vshogo",
        "vshemu",
        "vshomu",
        "vshikh",
        "vshykh",
        "vshui^u",
        "vshai^a",
        "vshoi^u",
        "vshei^u",
        "emi^ui^u",
        "emi^ai^a",
        "nni^ui^u",
        "nni^ai^a",
        "vshee",
        "vshie",
        "vshye",
        "vshoe",
        "vshei`",
        "vshii`",
        "vshyi`",
        "vshoi`",
        "vshem",
        "vshim",
        "vshym",
        "vshom",
        "emimi",
        "emymi",
        "emego",
        "emogo",
        "ememu",
        "emomu",
        "emikh",
        "emykh",
        "emui^u",
        "emai^a",
        "emoi^u",
        "emei^u",
        "nnimi",
        "nnymi",
        "nnego",
        "nnogo",
        "nnemu",
        "nnomu",
        "nnikh",
        "nnykh",
        "nnui^u",
        "nnai^a",
        "nnoi^u",
        "nnei^u",
        "emee",
        "emie",
        "emye",
        "emoe",
        "emei`",
        "emii`",
        "emyi`",
        "emoi`",
        "emem",
        "emim",
        "emym",
        "emom",
        "nnee",
        "nnie",
        "nnye",
        "nnoe",
        "nnei`",
        "nnii`",
        "nnyi`",
        "nnoi`",
        "nnem",
        "nnim",
        "nnym",
        "nnom",
        "i^ui^u",
        "i^ai^a",
        "imi",
        "ymi",
        "ego",
        "ogo",
        "emu",
        "omu",
        "ikh",
        "ykh",
        "ui^u",
        "ai^a",
        "oi^u",
        "ei^u",
        "ee",
        "ie",
        "ye",
        "oe",
        "ei`",
        "ii`",
        "yi`",
        "oi`",
        "em",
        "im",
        "ym",
        "om",
    )
    __reflexive_suffixes = ("si^a", "s'")
    __verb_suffixes = (
        "esh'",
        "ei`te",
        "ui`te",
        "ui^ut",
        "ish'",
        "ete",
        "i`te",
        "i^ut",
        "nno",
        "ila",
        "yla",
        "ena",
        "ite",
        "ili",
        "yli",
        "ilo",
        "ylo",
        "eno",
        "i^at",
        "uet",
        "eny",
        "it'",
        "yt'",
        "ui^u",
        "la",
        "na",
        "li",
        "em",
        "lo",
        "no",
        "et",
        "ny",
        "t'",
        "ei`",
        "ui`",
        "il",
        "yl",
        "im",
        "ym",
        "en",
        "it",
        "yt",
        "i^u",
        "i`",
        "l",
        "n",
    )
    __noun_suffixes = (
        "ii^ami",
        "ii^akh",
        "i^ami",
        "ii^am",
        "i^akh",
        "ami",
        "iei`",
        "i^am",
        "iem",
        "akh",
        "ii^u",
        "'i^u",
        "ii^a",
        "'i^a",
        "ev",
        "ov",
        "ie",
        "'e",
        "ei",
        "ii",
        "ei`",
        "oi`",
        "ii`",
        "em",
        "am",
        "om",
        "i^u",
        "i^a",
        "a",
        "e",
        "i",
        "i`",
        "o",
        "u",
        "y",
        "'",
    )
    __superlative_suffixes = ("ei`she", "ei`sh")
    __derivational_suffixes = ("ost'", "ost")

    def stem(self, word):
        """
        Stem a Russian word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        if word in self.stopwords:
            return word

        chr_exceeded = False
        for i in range(len(word)):
            if ord(word[i]) > 255:
                chr_exceeded = True
                break

        if not chr_exceeded:
            return word

        word = self.__cyrillic_to_roman(word)

        step1_success = False
        adjectival_removed = False
        verb_removed = False
        undouble_success = False
        superlative_removed = False

        rv, r2 = self.__regions_russian(word)

        # Step 1
        for suffix in self.__perfective_gerund_suffixes:
            if rv.endswith(suffix):
                if suffix in ("v", "vshi", "vshis'"):
                    if (
                        rv[-len(suffix) - 3 : -len(suffix)] == "i^a"
                        or rv[-len(suffix) - 1 : -len(suffix)] == "a"
                    ):
                        word = word[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        rv = rv[: -len(suffix)]
                        step1_success = True
                        break
                else:
                    word = word[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    step1_success = True
                    break

        if not step1_success:
            for suffix in self.__reflexive_suffixes:
                if rv.endswith(suffix):
                    word = word[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    break

            for suffix in self.__adjectival_suffixes:
                if rv.endswith(suffix):
                    if suffix in (
                        "i^ushchi^ui^u",
                        "i^ushchi^ai^a",
                        "i^ushchui^u",
                        "i^ushchai^a",
                        "i^ushchoi^u",
                        "i^ushchei^u",
                        "i^ushchimi",
                        "i^ushchymi",
                        "i^ushchego",
                        "i^ushchogo",
                        "i^ushchemu",
                        "i^ushchomu",
                        "i^ushchikh",
                        "i^ushchykh",
                        "shchi^ui^u",
                        "shchi^ai^a",
                        "i^ushchee",
                        "i^ushchie",
                        "i^ushchye",
                        "i^ushchoe",
                        "i^ushchei`",
                        "i^ushchii`",
                        "i^ushchyi`",
                        "i^ushchoi`",
                        "i^ushchem",
                        "i^ushchim",
                        "i^ushchym",
                        "i^ushchom",
                        "vshi^ui^u",
                        "vshi^ai^a",
                        "shchui^u",
                        "shchai^a",
                        "shchoi^u",
                        "shchei^u",
                        "emi^ui^u",
                        "emi^ai^a",
                        "nni^ui^u",
                        "nni^ai^a",
                        "shchimi",
                        "shchymi",
                        "shchego",
                        "shchogo",
                        "shchemu",
                        "shchomu",
                        "shchikh",
                        "shchykh",
                        "vshui^u",
                        "vshai^a",
                        "vshoi^u",
                        "vshei^u",
                        "shchee",
                        "shchie",
                        "shchye",
                        "shchoe",
                        "shchei`",
                        "shchii`",
                        "shchyi`",
                        "shchoi`",
                        "shchem",
                        "shchim",
                        "shchym",
                        "shchom",
                        "vshimi",
                        "vshymi",
                        "vshego",
                        "vshogo",
                        "vshemu",
                        "vshomu",
                        "vshikh",
                        "vshykh",
                        "emui^u",
                        "emai^a",
                        "emoi^u",
                        "emei^u",
                        "nnui^u",
                        "nnai^a",
                        "nnoi^u",
                        "nnei^u",
                        "vshee",
                        "vshie",
                        "vshye",
                        "vshoe",
                        "vshei`",
                        "vshii`",
                        "vshyi`",
                        "vshoi`",
                        "vshem",
                        "vshim",
                        "vshym",
                        "vshom",
                        "emimi",
                        "emymi",
                        "emego",
                        "emogo",
                        "ememu",
                        "emomu",
                        "emikh",
                        "emykh",
                        "nnimi",
                        "nnymi",
                        "nnego",
                        "nnogo",
                        "nnemu",
                        "nnomu",
                        "nnikh",
                        "nnykh",
                        "emee",
                        "emie",
                        "emye",
                        "emoe",
                        "emei`",
                        "emii`",
                        "emyi`",
                        "emoi`",
                        "emem",
                        "emim",
                        "emym",
                        "emom",
                        "nnee",
                        "nnie",
                        "nnye",
                        "nnoe",
                        "nnei`",
                        "nnii`",
                        "nnyi`",
                        "nnoi`",
                        "nnem",
                        "nnim",
                        "nnym",
                        "nnom",
                    ):
                        if (
                            rv[-len(suffix) - 3 : -len(suffix)] == "i^a"
                            or rv[-len(suffix) - 1 : -len(suffix)] == "a"
                        ):
                            word = word[: -len(suffix)]
                            r2 = r2[: -len(suffix)]
                            rv = rv[: -len(suffix)]
                            adjectival_removed = True
                            break
                    else:
                        word = word[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        rv = rv[: -len(suffix)]
                        adjectival_removed = True
                        break

            if not adjectival_removed:
                for suffix in self.__verb_suffixes:
                    if rv.endswith(suffix):
                        if suffix in (
                            "la",
                            "na",
                            "ete",
                            "i`te",
                            "li",
                            "i`",
                            "l",
                            "em",
                            "n",
                            "lo",
                            "no",
                            "et",
                            "i^ut",
                            "ny",
                            "t'",
                            "esh'",
                            "nno",
                        ):
                            if (
                                rv[-len(suffix) - 3 : -len(suffix)] == "i^a"
                                or rv[-len(suffix) - 1 : -len(suffix)] == "a"
                            ):
                                word = word[: -len(suffix)]
                                r2 = r2[: -len(suffix)]
                                rv = rv[: -len(suffix)]
                                verb_removed = True
                                break
                        else:
                            word = word[: -len(suffix)]
                            r2 = r2[: -len(suffix)]
                            rv = rv[: -len(suffix)]
                            verb_removed = True
                            break

            if not adjectival_removed and not verb_removed:
                for suffix in self.__noun_suffixes:
                    if rv.endswith(suffix):
                        word = word[: -len(suffix)]
                        r2 = r2[: -len(suffix)]
                        rv = rv[: -len(suffix)]
                        break

        # Step 2
        if rv.endswith("i"):
            word = word[:-1]
            r2 = r2[:-1]

        # Step 3
        for suffix in self.__derivational_suffixes:
            if r2.endswith(suffix):
                word = word[: -len(suffix)]
                break

        # Step 4
        if word.endswith("nn"):
            word = word[:-1]
            undouble_success = True

        if not undouble_success:
            for suffix in self.__superlative_suffixes:
                if word.endswith(suffix):
                    word = word[: -len(suffix)]
                    superlative_removed = True
                    break
            if word.endswith("nn"):
                word = word[:-1]

        if not undouble_success and not superlative_removed:
            if word.endswith("'"):
                word = word[:-1]

        word = self.__roman_to_cyrillic(word)

        return word

    def __regions_russian(self, word):
        """
        Return the regions RV and R2 which are used by the Russian stemmer.

        In any word, RV is the region after the first vowel,
        or the end of the word if it contains no vowel.

        R2 is the region after the first non-vowel following
        a vowel in R1, or the end of the word if there is no such non-vowel.

        R1 is the region after the first non-vowel following a vowel,
        or the end of the word if there is no such non-vowel.

        :param word: The Russian word whose regions RV and R2 are determined.
        :type word: str or unicode
        :return: the regions RV and R2 for the respective Russian word.
        :rtype: tuple
        :note: This helper method is invoked by the stem method of the subclass
               RussianStemmer. It is not to be invoked directly!

        """
        r1 = ""
        r2 = ""
        rv = ""

        vowels = ("A", "U", "E", "a", "e", "i", "o", "u", "y")
        word = word.replace("i^a", "A").replace("i^u", "U").replace("e`", "E")

        for i in range(1, len(word)):
            if word[i] not in vowels and word[i - 1] in vowels:
                r1 = word[i + 1 :]
                break

        for i in range(1, len(r1)):
            if r1[i] not in vowels and r1[i - 1] in vowels:
                r2 = r1[i + 1 :]
                break

        for i in range(len(word)):
            if word[i] in vowels:
                rv = word[i + 1 :]
                break

        r2 = r2.replace("A", "i^a").replace("U", "i^u").replace("E", "e`")
        rv = rv.replace("A", "i^a").replace("U", "i^u").replace("E", "e`")

        return (rv, r2)

    def __cyrillic_to_roman(self, word):
        """
        Transliterate a Russian word into the Roman alphabet.

        A Russian word whose letters consist of the Cyrillic
        alphabet are transliterated into the Roman alphabet
        in order to ease the forthcoming stemming process.

        :param word: The word that is transliterated.
        :type word: unicode
        :return: the transliterated word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of the subclass
               RussianStemmer. It is not to be invoked directly!

        """
        word = (
            word.replace("\u0410", "a")
            .replace("\u0430", "a")
            .replace("\u0411", "b")
            .replace("\u0431", "b")
            .replace("\u0412", "v")
            .replace("\u0432", "v")
            .replace("\u0413", "g")
            .replace("\u0433", "g")
            .replace("\u0414", "d")
            .replace("\u0434", "d")
            .replace("\u0415", "e")
            .replace("\u0435", "e")
            .replace("\u0401", "e")
            .replace("\u0451", "e")
            .replace("\u0416", "zh")
            .replace("\u0436", "zh")
            .replace("\u0417", "z")
            .replace("\u0437", "z")
            .replace("\u0418", "i")
            .replace("\u0438", "i")
            .replace("\u0419", "i`")
            .replace("\u0439", "i`")
            .replace("\u041A", "k")
            .replace("\u043A", "k")
            .replace("\u041B", "l")
            .replace("\u043B", "l")
            .replace("\u041C", "m")
            .replace("\u043C", "m")
            .replace("\u041D", "n")
            .replace("\u043D", "n")
            .replace("\u041E", "o")
            .replace("\u043E", "o")
            .replace("\u041F", "p")
            .replace("\u043F", "p")
            .replace("\u0420", "r")
            .replace("\u0440", "r")
            .replace("\u0421", "s")
            .replace("\u0441", "s")
            .replace("\u0422", "t")
            .replace("\u0442", "t")
            .replace("\u0423", "u")
            .replace("\u0443", "u")
            .replace("\u0424", "f")
            .replace("\u0444", "f")
            .replace("\u0425", "kh")
            .replace("\u0445", "kh")
            .replace("\u0426", "t^s")
            .replace("\u0446", "t^s")
            .replace("\u0427", "ch")
            .replace("\u0447", "ch")
            .replace("\u0428", "sh")
            .replace("\u0448", "sh")
            .replace("\u0429", "shch")
            .replace("\u0449", "shch")
            .replace("\u042A", "''")
            .replace("\u044A", "''")
            .replace("\u042B", "y")
            .replace("\u044B", "y")
            .replace("\u042C", "'")
            .replace("\u044C", "'")
            .replace("\u042D", "e`")
            .replace("\u044D", "e`")
            .replace("\u042E", "i^u")
            .replace("\u044E", "i^u")
            .replace("\u042F", "i^a")
            .replace("\u044F", "i^a")
        )

        return word

    def __roman_to_cyrillic(self, word):
        """
        Transliterate a Russian word back into the Cyrillic alphabet.

        A Russian word formerly transliterated into the Roman alphabet
        in order to ease the stemming process, is transliterated back
        into the Cyrillic alphabet, its original form.

        :param word: The word that is transliterated.
        :type word: str or unicode
        :return: word, the transliterated word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of the subclass
               RussianStemmer. It is not to be invoked directly!

        """
        word = (
            word.replace("i^u", "\u044E")
            .replace("i^a", "\u044F")
            .replace("shch", "\u0449")
            .replace("kh", "\u0445")
            .replace("t^s", "\u0446")
            .replace("ch", "\u0447")
            .replace("e`", "\u044D")
            .replace("i`", "\u0439")
            .replace("sh", "\u0448")
            .replace("k", "\u043A")
            .replace("e", "\u0435")
            .replace("zh", "\u0436")
            .replace("a", "\u0430")
            .replace("b", "\u0431")
            .replace("v", "\u0432")
            .replace("g", "\u0433")
            .replace("d", "\u0434")
            .replace("e", "\u0435")
            .replace("z", "\u0437")
            .replace("i", "\u0438")
            .replace("l", "\u043B")
            .replace("m", "\u043C")
            .replace("n", "\u043D")
            .replace("o", "\u043E")
            .replace("p", "\u043F")
            .replace("r", "\u0440")
            .replace("s", "\u0441")
            .replace("t", "\u0442")
            .replace("u", "\u0443")
            .replace("f", "\u0444")
            .replace("''", "\u044A")
            .replace("y", "\u044B")
            .replace("'", "\u044C")
        )

        return word


class SpanishStemmer(_StandardStemmer):

    """
    The Spanish Snowball stemmer.

    :cvar __vowels: The Spanish vowels.
    :type __vowels: unicode
    :cvar __step0_suffixes: Suffixes to be deleted in step 0 of the algorithm.
    :type __step0_suffixes: tuple
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2a_suffixes: Suffixes to be deleted in step 2a of the algorithm.
    :type __step2a_suffixes: tuple
    :cvar __step2b_suffixes: Suffixes to be deleted in step 2b of the algorithm.
    :type __step2b_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :note: A detailed description of the Spanish
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/spanish/stemmer.html

    """

    __vowels = "aeiou\xE1\xE9\xED\xF3\xFA\xFC"
    __step0_suffixes = (
        "selas",
        "selos",
        "sela",
        "selo",
        "las",
        "les",
        "los",
        "nos",
        "me",
        "se",
        "la",
        "le",
        "lo",
    )
    __step1_suffixes = (
        "amientos",
        "imientos",
        "amiento",
        "imiento",
        "acion",
        "aciones",
        "uciones",
        "adoras",
        "adores",
        "ancias",
        "log\xEDas",
        "encias",
        "amente",
        "idades",
        "anzas",
        "ismos",
        "ables",
        "ibles",
        "istas",
        "adora",
        "aci\xF3n",
        "antes",
        "ancia",
        "log\xEDa",
        "uci\xf3n",
        "encia",
        "mente",
        "anza",
        "icos",
        "icas",
        "ismo",
        "able",
        "ible",
        "ista",
        "osos",
        "osas",
        "ador",
        "ante",
        "idad",
        "ivas",
        "ivos",
        "ico",
        "ica",
        "oso",
        "osa",
        "iva",
        "ivo",
    )
    __step2a_suffixes = (
        "yeron",
        "yendo",
        "yamos",
        "yais",
        "yan",
        "yen",
        "yas",
        "yes",
        "ya",
        "ye",
        "yo",
        "y\xF3",
    )
    __step2b_suffixes = (
        "ar\xEDamos",
        "er\xEDamos",
        "ir\xEDamos",
        "i\xE9ramos",
        "i\xE9semos",
        "ar\xEDais",
        "aremos",
        "er\xEDais",
        "eremos",
        "ir\xEDais",
        "iremos",
        "ierais",
        "ieseis",
        "asteis",
        "isteis",
        "\xE1bamos",
        "\xE1ramos",
        "\xE1semos",
        "ar\xEDan",
        "ar\xEDas",
        "ar\xE9is",
        "er\xEDan",
        "er\xEDas",
        "er\xE9is",
        "ir\xEDan",
        "ir\xEDas",
        "ir\xE9is",
        "ieran",
        "iesen",
        "ieron",
        "iendo",
        "ieras",
        "ieses",
        "abais",
        "arais",
        "aseis",
        "\xE9amos",
        "ar\xE1n",
        "ar\xE1s",
        "ar\xEDa",
        "er\xE1n",
        "er\xE1s",
        "er\xEDa",
        "ir\xE1n",
        "ir\xE1s",
        "ir\xEDa",
        "iera",
        "iese",
        "aste",
        "iste",
        "aban",
        "aran",
        "asen",
        "aron",
        "ando",
        "abas",
        "adas",
        "idas",
        "aras",
        "ases",
        "\xEDais",
        "ados",
        "idos",
        "amos",
        "imos",
        "emos",
        "ar\xE1",
        "ar\xE9",
        "er\xE1",
        "er\xE9",
        "ir\xE1",
        "ir\xE9",
        "aba",
        "ada",
        "ida",
        "ara",
        "ase",
        "\xEDan",
        "ado",
        "ido",
        "\xEDas",
        "\xE1is",
        "\xE9is",
        "\xEDa",
        "ad",
        "ed",
        "id",
        "an",
        "i\xF3",
        "ar",
        "er",
        "ir",
        "as",
        "\xEDs",
        "en",
        "es",
    )
    __step3_suffixes = ("os", "a", "e", "o", "\xE1", "\xE9", "\xED", "\xF3")

    def stem(self, word):
        """
        Stem a Spanish word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        step1_success = False

        r1, r2 = self._r1r2_standard(word, self.__vowels)
        rv = self._rv_standard(word, self.__vowels)

        # STEP 0: Attached pronoun
        for suffix in self.__step0_suffixes:
            if not (word.endswith(suffix) and rv.endswith(suffix)):
                continue

            if (
                rv[: -len(suffix)].endswith(
                    (
                        "ando",
                        "\xE1ndo",
                        "ar",
                        "\xE1r",
                        "er",
                        "\xE9r",
                        "iendo",
                        "i\xE9ndo",
                        "ir",
                        "\xEDr",
                    )
                )
            ) or (
                rv[: -len(suffix)].endswith("yendo")
                and word[: -len(suffix)].endswith("uyendo")
            ):

                word = self.__replace_accented(word[: -len(suffix)])
                r1 = self.__replace_accented(r1[: -len(suffix)])
                r2 = self.__replace_accented(r2[: -len(suffix)])
                rv = self.__replace_accented(rv[: -len(suffix)])
            break

        # STEP 1: Standard suffix removal
        for suffix in self.__step1_suffixes:
            if not word.endswith(suffix):
                continue

            if suffix == "amente" and r1.endswith(suffix):
                step1_success = True
                word = word[:-6]
                r2 = r2[:-6]
                rv = rv[:-6]

                if r2.endswith("iv"):
                    word = word[:-2]
                    r2 = r2[:-2]
                    rv = rv[:-2]

                    if r2.endswith("at"):
                        word = word[:-2]
                        rv = rv[:-2]

                elif r2.endswith(("os", "ic", "ad")):
                    word = word[:-2]
                    rv = rv[:-2]

            elif r2.endswith(suffix):
                step1_success = True
                if suffix in (
                    "adora",
                    "ador",
                    "aci\xF3n",
                    "adoras",
                    "adores",
                    "acion",
                    "aciones",
                    "ante",
                    "antes",
                    "ancia",
                    "ancias",
                ):
                    word = word[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    rv = rv[: -len(suffix)]

                    if r2.endswith("ic"):
                        word = word[:-2]
                        rv = rv[:-2]

                elif suffix in ("log\xEDa", "log\xEDas"):
                    word = suffix_replace(word, suffix, "log")
                    rv = suffix_replace(rv, suffix, "log")

                elif suffix in ("uci\xF3n", "uciones"):
                    word = suffix_replace(word, suffix, "u")
                    rv = suffix_replace(rv, suffix, "u")

                elif suffix in ("encia", "encias"):
                    word = suffix_replace(word, suffix, "ente")
                    rv = suffix_replace(rv, suffix, "ente")

                elif suffix == "mente":
                    word = word[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    rv = rv[: -len(suffix)]

                    if r2.endswith(("ante", "able", "ible")):
                        word = word[:-4]
                        rv = rv[:-4]

                elif suffix in ("idad", "idades"):
                    word = word[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    rv = rv[: -len(suffix)]

                    for pre_suff in ("abil", "ic", "iv"):
                        if r2.endswith(pre_suff):
                            word = word[: -len(pre_suff)]
                            rv = rv[: -len(pre_suff)]

                elif suffix in ("ivo", "iva", "ivos", "ivas"):
                    word = word[: -len(suffix)]
                    r2 = r2[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    if r2.endswith("at"):
                        word = word[:-2]
                        rv = rv[:-2]
                else:
                    word = word[: -len(suffix)]
                    rv = rv[: -len(suffix)]
            break

        # STEP 2a: Verb suffixes beginning 'y'
        if not step1_success:
            for suffix in self.__step2a_suffixes:
                if rv.endswith(suffix) and word[-len(suffix) - 1 : -len(suffix)] == "u":
                    word = word[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    break

            # STEP 2b: Other verb suffixes
            for suffix in self.__step2b_suffixes:
                if rv.endswith(suffix):
                    word = word[: -len(suffix)]
                    rv = rv[: -len(suffix)]
                    if suffix in ("en", "es", "\xE9is", "emos"):
                        if word.endswith("gu"):
                            word = word[:-1]

                        if rv.endswith("gu"):
                            rv = rv[:-1]
                    break

        # STEP 3: Residual suffix
        for suffix in self.__step3_suffixes:
            if rv.endswith(suffix):
                word = word[: -len(suffix)]
                if suffix in ("e", "\xE9"):
                    rv = rv[: -len(suffix)]

                    if word[-2:] == "gu" and rv.endswith("u"):
                        word = word[:-1]
                break

        word = self.__replace_accented(word)

        return word

    def __replace_accented(self, word):
        """
        Replaces all accented letters on a word with their non-accented
        counterparts.

        :param word: A spanish word, with or without accents
        :type word: str or unicode
        :return: a word with the accented letters (á, é, í, ó, ú) replaced with
                 their non-accented counterparts (a, e, i, o, u)
        :rtype: str or unicode
        """
        return (
            word.replace("\xE1", "a")
            .replace("\xE9", "e")
            .replace("\xED", "i")
            .replace("\xF3", "o")
            .replace("\xFA", "u")
        )


class SwedishStemmer(_ScandinavianStemmer):

    """
    The Swedish Snowball stemmer.

    :cvar __vowels: The Swedish vowels.
    :type __vowels: unicode
    :cvar __s_ending: Letters that may directly appear before a word final 's'.
    :type __s_ending: unicode
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :note: A detailed description of the Swedish
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/swedish/stemmer.html

    """

    __vowels = "aeiouy\xE4\xE5\xF6"
    __s_ending = "bcdfghjklmnoprtvy"
    __step1_suffixes = (
        "heterna",
        "hetens",
        "heter",
        "heten",
        "anden",
        "arnas",
        "ernas",
        "ornas",
        "andes",
        "andet",
        "arens",
        "arna",
        "erna",
        "orna",
        "ande",
        "arne",
        "aste",
        "aren",
        "ades",
        "erns",
        "ade",
        "are",
        "ern",
        "ens",
        "het",
        "ast",
        "ad",
        "en",
        "ar",
        "er",
        "or",
        "as",
        "es",
        "at",
        "a",
        "e",
        "s",
    )
    __step2_suffixes = ("dd", "gd", "nn", "dt", "gt", "kt", "tt")
    __step3_suffixes = ("fullt", "l\xF6st", "els", "lig", "ig")

    def stem(self, word):
        """
        Stem a Swedish word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()

        if word in self.stopwords:
            return word

        r1 = self._r1_scandinavian(word, self.__vowels)

        # STEP 1
        for suffix in self.__step1_suffixes:
            if r1.endswith(suffix):
                if suffix == "s":
                    if word[-2] in self.__s_ending:
                        word = word[:-1]
                        r1 = r1[:-1]
                else:
                    word = word[: -len(suffix)]
                    r1 = r1[: -len(suffix)]
                break

        # STEP 2
        for suffix in self.__step2_suffixes:
            if r1.endswith(suffix):
                word = word[:-1]
                r1 = r1[:-1]
                break

        # STEP 3
        for suffix in self.__step3_suffixes:
            if r1.endswith(suffix):
                if suffix in ("els", "lig", "ig"):
                    word = word[: -len(suffix)]
                elif suffix in ("fullt", "l\xF6st"):
                    word = word[:-1]
                break

        return word


def demo():
    """
    This function provides a demonstration of the Snowball stemmers.

    After invoking this function and specifying a language,
    it stems an excerpt of the Universal Declaration of Human Rights
    (which is a part of the NLTK corpus collection) and then prints
    out the original and the stemmed text.

    """

    from nltk.corpus import udhr

    udhr_corpus = {
        "arabic": "Arabic_Alarabia-Arabic",
        "danish": "Danish_Dansk-Latin1",
        "dutch": "Dutch_Nederlands-Latin1",
        "english": "English-Latin1",
        "finnish": "Finnish_Suomi-Latin1",
        "french": "French_Francais-Latin1",
        "german": "German_Deutsch-Latin1",
        "hungarian": "Hungarian_Magyar-UTF8",
        "italian": "Italian_Italiano-Latin1",
        "norwegian": "Norwegian-Latin1",
        "porter": "English-Latin1",
        "portuguese": "Portuguese_Portugues-Latin1",
        "romanian": "Romanian_Romana-Latin2",
        "russian": "Russian-UTF8",
        "spanish": "Spanish-Latin1",
        "swedish": "Swedish_Svenska-Latin1",
    }

    print("\n")
    print("******************************")
    print("Demo for the Snowball stemmers")
    print("******************************")

    while True:

        language = input(
            "Please enter the name of the language "
            + "to be demonstrated\n"
            + "/".join(SnowballStemmer.languages)
            + "\n"
            + "(enter 'exit' in order to leave): "
        )

        if language == "exit":
            break

        if language not in SnowballStemmer.languages:
            print(
                "\nOops, there is no stemmer for this language. "
                + "Please try again.\n"
            )
            continue

        stemmer = SnowballStemmer(language)
        excerpt = udhr.words(udhr_corpus[language])[:300]

        stemmed = " ".join(stemmer.stem(word) for word in excerpt)
        stemmed = re.sub(r"(.{,70})\s", r"\1\n", stemmed + " ").rstrip()
        excerpt = " ".join(excerpt)
        excerpt = re.sub(r"(.{,70})\s", r"\1\n", excerpt + " ").rstrip()

        print("\n")
        print("-" * 70)
        print("ORIGINAL".center(70))
        print(excerpt)
        print("\n\n")
        print("STEMMED RESULTS".center(70))
        print(stemmed)
        print("-" * 70)
        print("\n")
