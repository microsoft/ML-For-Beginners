# Natural Language Toolkit: Stemmers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Tomcavage <stomcava@law.upenn.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A word stemmer based on the Lancaster (Paice/Husk) stemming algorithm.
Paice, Chris D. "Another Stemmer." ACM SIGIR Forum 24.3 (1990): 56-61.
"""
import re

from nltk.stem.api import StemmerI


class LancasterStemmer(StemmerI):
    """
    Lancaster Stemmer

        >>> from nltk.stem.lancaster import LancasterStemmer
        >>> st = LancasterStemmer()
        >>> st.stem('maximum')     # Remove "-um" when word is intact
        'maxim'
        >>> st.stem('presumably')  # Don't remove "-um" when word is not intact
        'presum'
        >>> st.stem('multiply')    # No action taken if word ends with "-ply"
        'multiply'
        >>> st.stem('provision')   # Replace "-sion" with "-j" to trigger "j" set of rules
        'provid'
        >>> st.stem('owed')        # Word starting with vowel must contain at least 2 letters
        'ow'
        >>> st.stem('ear')         # ditto
        'ear'
        >>> st.stem('saying')      # Words starting with consonant must contain at least 3
        'say'
        >>> st.stem('crying')      #     letters and one of those letters must be a vowel
        'cry'
        >>> st.stem('string')      # ditto
        'string'
        >>> st.stem('meant')       # ditto
        'meant'
        >>> st.stem('cement')      # ditto
        'cem'
        >>> st_pre = LancasterStemmer(strip_prefix_flag=True)
        >>> st_pre.stem('kilometer') # Test Prefix
        'met'
        >>> st_custom = LancasterStemmer(rule_tuple=("ssen4>", "s1t."))
        >>> st_custom.stem("ness") # Change s to t
        'nest'
    """

    # The rule list is static since it doesn't change between instances
    default_rule_tuple = (
        "ai*2.",  # -ia > -   if intact
        "a*1.",  # -a > -    if intact
        "bb1.",  # -bb > -b
        "city3s.",  # -ytic > -ys
        "ci2>",  # -ic > -
        "cn1t>",  # -nc > -nt
        "dd1.",  # -dd > -d
        "dei3y>",  # -ied > -y
        "deec2ss.",  # -ceed >", -cess
        "dee1.",  # -eed > -ee
        "de2>",  # -ed > -
        "dooh4>",  # -hood > -
        "e1>",  # -e > -
        "feil1v.",  # -lief > -liev
        "fi2>",  # -if > -
        "gni3>",  # -ing > -
        "gai3y.",  # -iag > -y
        "ga2>",  # -ag > -
        "gg1.",  # -gg > -g
        "ht*2.",  # -th > -   if intact
        "hsiug5ct.",  # -guish > -ct
        "hsi3>",  # -ish > -
        "i*1.",  # -i > -    if intact
        "i1y>",  # -i > -y
        "ji1d.",  # -ij > -id   --  see nois4j> & vis3j>
        "juf1s.",  # -fuj > -fus
        "ju1d.",  # -uj > -ud
        "jo1d.",  # -oj > -od
        "jeh1r.",  # -hej > -her
        "jrev1t.",  # -verj > -vert
        "jsim2t.",  # -misj > -mit
        "jn1d.",  # -nj > -nd
        "j1s.",  # -j > -s
        "lbaifi6.",  # -ifiabl > -
        "lbai4y.",  # -iabl > -y
        "lba3>",  # -abl > -
        "lbi3.",  # -ibl > -
        "lib2l>",  # -bil > -bl
        "lc1.",  # -cl > c
        "lufi4y.",  # -iful > -y
        "luf3>",  # -ful > -
        "lu2.",  # -ul > -
        "lai3>",  # -ial > -
        "lau3>",  # -ual > -
        "la2>",  # -al > -
        "ll1.",  # -ll > -l
        "mui3.",  # -ium > -
        "mu*2.",  # -um > -   if intact
        "msi3>",  # -ism > -
        "mm1.",  # -mm > -m
        "nois4j>",  # -sion > -j
        "noix4ct.",  # -xion > -ct
        "noi3>",  # -ion > -
        "nai3>",  # -ian > -
        "na2>",  # -an > -
        "nee0.",  # protect  -een
        "ne2>",  # -en > -
        "nn1.",  # -nn > -n
        "pihs4>",  # -ship > -
        "pp1.",  # -pp > -p
        "re2>",  # -er > -
        "rae0.",  # protect  -ear
        "ra2.",  # -ar > -
        "ro2>",  # -or > -
        "ru2>",  # -ur > -
        "rr1.",  # -rr > -r
        "rt1>",  # -tr > -t
        "rei3y>",  # -ier > -y
        "sei3y>",  # -ies > -y
        "sis2.",  # -sis > -s
        "si2>",  # -is > -
        "ssen4>",  # -ness > -
        "ss0.",  # protect  -ss
        "suo3>",  # -ous > -
        "su*2.",  # -us > -   if intact
        "s*1>",  # -s > -    if intact
        "s0.",  # -s > -s
        "tacilp4y.",  # -plicat > -ply
        "ta2>",  # -at > -
        "tnem4>",  # -ment > -
        "tne3>",  # -ent > -
        "tna3>",  # -ant > -
        "tpir2b.",  # -ript > -rib
        "tpro2b.",  # -orpt > -orb
        "tcud1.",  # -duct > -duc
        "tpmus2.",  # -sumpt > -sum
        "tpec2iv.",  # -cept > -ceiv
        "tulo2v.",  # -olut > -olv
        "tsis0.",  # protect  -sist
        "tsi3>",  # -ist > -
        "tt1.",  # -tt > -t
        "uqi3.",  # -iqu > -
        "ugo1.",  # -ogu > -og
        "vis3j>",  # -siv > -j
        "vie0.",  # protect  -eiv
        "vi2>",  # -iv > -
        "ylb1>",  # -bly > -bl
        "yli3y>",  # -ily > -y
        "ylp0.",  # protect  -ply
        "yl2>",  # -ly > -
        "ygo1.",  # -ogy > -og
        "yhp1.",  # -phy > -ph
        "ymo1.",  # -omy > -om
        "ypo1.",  # -opy > -op
        "yti3>",  # -ity > -
        "yte3>",  # -ety > -
        "ytl2.",  # -lty > -l
        "yrtsi5.",  # -istry > -
        "yra3>",  # -ary > -
        "yro3>",  # -ory > -
        "yfi3.",  # -ify > -
        "ycn2t>",  # -ncy > -nt
        "yca3>",  # -acy > -
        "zi2>",  # -iz > -
        "zy1s.",  # -yz > -ys
    )

    def __init__(self, rule_tuple=None, strip_prefix_flag=False):
        """Create an instance of the Lancaster stemmer."""
        # Setup an empty rule dictionary - this will be filled in later
        self.rule_dictionary = {}
        # Check if a user wants to strip prefix
        self._strip_prefix = strip_prefix_flag
        # Check if a user wants to use his/her own rule tuples.
        self._rule_tuple = rule_tuple if rule_tuple else self.default_rule_tuple

    def parseRules(self, rule_tuple=None):
        """Validate the set of rules used in this stemmer.

        If this function is called as an individual method, without using stem
        method, rule_tuple argument will be compiled into self.rule_dictionary.
        If this function is called within stem, self._rule_tuple will be used.

        """
        # If there is no argument for the function, use class' own rule tuple.
        rule_tuple = rule_tuple if rule_tuple else self._rule_tuple
        valid_rule = re.compile(r"^[a-z]+\*?\d[a-z]*[>\.]?$")
        # Empty any old rules from the rule set before adding new ones
        self.rule_dictionary = {}

        for rule in rule_tuple:
            if not valid_rule.match(rule):
                raise ValueError(f"The rule {rule} is invalid")
            first_letter = rule[0:1]
            if first_letter in self.rule_dictionary:
                self.rule_dictionary[first_letter].append(rule)
            else:
                self.rule_dictionary[first_letter] = [rule]

    def stem(self, word):
        """Stem a word using the Lancaster stemmer."""
        # Lower-case the word, since all the rules are lower-cased
        word = word.lower()
        word = self.__stripPrefix(word) if self._strip_prefix else word

        # Save a copy of the original word
        intact_word = word

        # If rule dictionary is empty, parse rule tuple.
        if not self.rule_dictionary:
            self.parseRules()

        return self.__doStemming(word, intact_word)

    def __doStemming(self, word, intact_word):
        """Perform the actual word stemming"""

        valid_rule = re.compile(r"^([a-z]+)(\*?)(\d)([a-z]*)([>\.]?)$")

        proceed = True

        while proceed:

            # Find the position of the last letter of the word to be stemmed
            last_letter_position = self.__getLastLetter(word)

            # Only stem the word if it has a last letter and a rule matching that last letter
            if (
                last_letter_position < 0
                or word[last_letter_position] not in self.rule_dictionary
            ):
                proceed = False

            else:
                rule_was_applied = False

                # Go through each rule that matches the word's final letter
                for rule in self.rule_dictionary[word[last_letter_position]]:
                    rule_match = valid_rule.match(rule)
                    if rule_match:
                        (
                            ending_string,
                            intact_flag,
                            remove_total,
                            append_string,
                            cont_flag,
                        ) = rule_match.groups()

                        # Convert the number of chars to remove when stemming
                        # from a string to an integer
                        remove_total = int(remove_total)

                        # Proceed if word's ending matches rule's word ending
                        if word.endswith(ending_string[::-1]):
                            if intact_flag:
                                if word == intact_word and self.__isAcceptable(
                                    word, remove_total
                                ):
                                    word = self.__applyRule(
                                        word, remove_total, append_string
                                    )
                                    rule_was_applied = True
                                    if cont_flag == ".":
                                        proceed = False
                                    break
                            elif self.__isAcceptable(word, remove_total):
                                word = self.__applyRule(
                                    word, remove_total, append_string
                                )
                                rule_was_applied = True
                                if cont_flag == ".":
                                    proceed = False
                                break
                # If no rules apply, the word doesn't need any more stemming
                if rule_was_applied == False:
                    proceed = False
        return word

    def __getLastLetter(self, word):
        """Get the zero-based index of the last alphabetic character in this string"""
        last_letter = -1
        for position in range(len(word)):
            if word[position].isalpha():
                last_letter = position
            else:
                break
        return last_letter

    def __isAcceptable(self, word, remove_total):
        """Determine if the word is acceptable for stemming."""
        word_is_acceptable = False
        # If the word starts with a vowel, it must be at least 2
        # characters long to be stemmed
        if word[0] in "aeiouy":
            if len(word) - remove_total >= 2:
                word_is_acceptable = True
        # If the word starts with a consonant, it must be at least 3
        # characters long (including one vowel) to be stemmed
        elif len(word) - remove_total >= 3:
            if word[1] in "aeiouy":
                word_is_acceptable = True
            elif word[2] in "aeiouy":
                word_is_acceptable = True
        return word_is_acceptable

    def __applyRule(self, word, remove_total, append_string):
        """Apply the stemming rule to the word"""
        # Remove letters from the end of the word
        new_word_length = len(word) - remove_total
        word = word[0:new_word_length]

        # And add new letters to the end of the truncated word
        if append_string:
            word += append_string
        return word

    def __stripPrefix(self, word):
        """Remove prefix from a word.

        This function originally taken from Whoosh.

        """
        for prefix in (
            "kilo",
            "micro",
            "milli",
            "intra",
            "ultra",
            "mega",
            "nano",
            "pico",
            "pseudo",
        ):
            if word.startswith(prefix):
                return word[len(prefix) :]
        return word

    def __repr__(self):
        return "<LancasterStemmer>"
