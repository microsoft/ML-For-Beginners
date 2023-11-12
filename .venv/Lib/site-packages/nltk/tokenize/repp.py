# Natural Language Toolkit: Interface to the Repp Tokenizer
#
# Copyright (C) 2001-2015 NLTK Project
# Authors: Rebecca Dridan and Stephan Oepen
# Contributors: Liling Tan
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import os
import re
import subprocess
import sys
import tempfile

from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir
from nltk.tokenize.api import TokenizerI


class ReppTokenizer(TokenizerI):
    """
    A class for word tokenization using the REPP parser described in
    Rebecca Dridan and Stephan Oepen (2012) Tokenization: Returning to a
    Long Solved Problem - A Survey, Contrastive  Experiment, Recommendations,
    and Toolkit. In ACL. http://anthology.aclweb.org/P/P12/P12-2.pdf#page=406

    >>> sents = ['Tokenization is widely regarded as a solved problem due to the high accuracy that rulebased tokenizers achieve.' ,
    ... 'But rule-based tokenizers are hard to maintain and their rules language specific.' ,
    ... 'We evaluated our method on three languages and obtained error rates of 0.27% (English), 0.35% (Dutch) and 0.76% (Italian) for our best models.'
    ... ]
    >>> tokenizer = ReppTokenizer('/home/alvas/repp/') # doctest: +SKIP
    >>> for sent in sents:                             # doctest: +SKIP
    ...     tokenizer.tokenize(sent)                   # doctest: +SKIP
    ...
    (u'Tokenization', u'is', u'widely', u'regarded', u'as', u'a', u'solved', u'problem', u'due', u'to', u'the', u'high', u'accuracy', u'that', u'rulebased', u'tokenizers', u'achieve', u'.')
    (u'But', u'rule-based', u'tokenizers', u'are', u'hard', u'to', u'maintain', u'and', u'their', u'rules', u'language', u'specific', u'.')
    (u'We', u'evaluated', u'our', u'method', u'on', u'three', u'languages', u'and', u'obtained', u'error', u'rates', u'of', u'0.27', u'%', u'(', u'English', u')', u',', u'0.35', u'%', u'(', u'Dutch', u')', u'and', u'0.76', u'%', u'(', u'Italian', u')', u'for', u'our', u'best', u'models', u'.')

    >>> for sent in tokenizer.tokenize_sents(sents): # doctest: +SKIP
    ...     print(sent)                              # doctest: +SKIP
    ...
    (u'Tokenization', u'is', u'widely', u'regarded', u'as', u'a', u'solved', u'problem', u'due', u'to', u'the', u'high', u'accuracy', u'that', u'rulebased', u'tokenizers', u'achieve', u'.')
    (u'But', u'rule-based', u'tokenizers', u'are', u'hard', u'to', u'maintain', u'and', u'their', u'rules', u'language', u'specific', u'.')
    (u'We', u'evaluated', u'our', u'method', u'on', u'three', u'languages', u'and', u'obtained', u'error', u'rates', u'of', u'0.27', u'%', u'(', u'English', u')', u',', u'0.35', u'%', u'(', u'Dutch', u')', u'and', u'0.76', u'%', u'(', u'Italian', u')', u'for', u'our', u'best', u'models', u'.')
    >>> for sent in tokenizer.tokenize_sents(sents, keep_token_positions=True): # doctest: +SKIP
    ...     print(sent)                                                         # doctest: +SKIP
    ...
    [(u'Tokenization', 0, 12), (u'is', 13, 15), (u'widely', 16, 22), (u'regarded', 23, 31), (u'as', 32, 34), (u'a', 35, 36), (u'solved', 37, 43), (u'problem', 44, 51), (u'due', 52, 55), (u'to', 56, 58), (u'the', 59, 62), (u'high', 63, 67), (u'accuracy', 68, 76), (u'that', 77, 81), (u'rulebased', 82, 91), (u'tokenizers', 92, 102), (u'achieve', 103, 110), (u'.', 110, 111)]
    [(u'But', 0, 3), (u'rule-based', 4, 14), (u'tokenizers', 15, 25), (u'are', 26, 29), (u'hard', 30, 34), (u'to', 35, 37), (u'maintain', 38, 46), (u'and', 47, 50), (u'their', 51, 56), (u'rules', 57, 62), (u'language', 63, 71), (u'specific', 72, 80), (u'.', 80, 81)]
    [(u'We', 0, 2), (u'evaluated', 3, 12), (u'our', 13, 16), (u'method', 17, 23), (u'on', 24, 26), (u'three', 27, 32), (u'languages', 33, 42), (u'and', 43, 46), (u'obtained', 47, 55), (u'error', 56, 61), (u'rates', 62, 67), (u'of', 68, 70), (u'0.27', 71, 75), (u'%', 75, 76), (u'(', 77, 78), (u'English', 78, 85), (u')', 85, 86), (u',', 86, 87), (u'0.35', 88, 92), (u'%', 92, 93), (u'(', 94, 95), (u'Dutch', 95, 100), (u')', 100, 101), (u'and', 102, 105), (u'0.76', 106, 110), (u'%', 110, 111), (u'(', 112, 113), (u'Italian', 113, 120), (u')', 120, 121), (u'for', 122, 125), (u'our', 126, 129), (u'best', 130, 134), (u'models', 135, 141), (u'.', 141, 142)]
    """

    def __init__(self, repp_dir, encoding="utf8"):
        self.repp_dir = self.find_repptokenizer(repp_dir)
        # Set a directory to store the temporary files.
        self.working_dir = tempfile.gettempdir()
        # Set an encoding for the input strings.
        self.encoding = encoding

    def tokenize(self, sentence):
        """
        Use Repp to tokenize a single sentence.

        :param sentence: A single sentence string.
        :type sentence: str
        :return: A tuple of tokens.
        :rtype: tuple(str)
        """
        return next(self.tokenize_sents([sentence]))

    def tokenize_sents(self, sentences, keep_token_positions=False):
        """
        Tokenize multiple sentences using Repp.

        :param sentences: A list of sentence strings.
        :type sentences: list(str)
        :return: A list of tuples of tokens
        :rtype: iter(tuple(str))
        """
        with tempfile.NamedTemporaryFile(
            prefix="repp_input.", dir=self.working_dir, mode="w", delete=False
        ) as input_file:
            # Write sentences to temporary input file.
            for sent in sentences:
                input_file.write(str(sent) + "\n")
            input_file.close()
            # Generate command to run REPP.
            cmd = self.generate_repp_command(input_file.name)
            # Decode the stdout and strips the ending newline.
            repp_output = self._execute(cmd).decode(self.encoding).strip()
            for tokenized_sent in self.parse_repp_outputs(repp_output):
                if not keep_token_positions:
                    # Removes token position information.
                    tokenized_sent, starts, ends = zip(*tokenized_sent)
                yield tokenized_sent

    def generate_repp_command(self, inputfilename):
        """
        This module generates the REPP command to be used at the terminal.

        :param inputfilename: path to the input file
        :type inputfilename: str
        """
        cmd = [self.repp_dir + "/src/repp"]
        cmd += ["-c", self.repp_dir + "/erg/repp.set"]
        cmd += ["--format", "triple"]
        cmd += [inputfilename]
        return cmd

    @staticmethod
    def _execute(cmd):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        return stdout

    @staticmethod
    def parse_repp_outputs(repp_output):
        """
        This module parses the tri-tuple format that REPP outputs using the
        "--format triple" option and returns an generator with tuple of string
        tokens.

        :param repp_output:
        :type repp_output: type
        :return: an iterable of the tokenized sentences as tuples of strings
        :rtype: iter(tuple)
        """
        line_regex = re.compile(r"^\((\d+), (\d+), (.+)\)$", re.MULTILINE)
        for section in repp_output.split("\n\n"):
            words_with_positions = [
                (token, int(start), int(end))
                for start, end, token in line_regex.findall(section)
            ]
            words = tuple(t[2] for t in words_with_positions)
            yield words_with_positions

    def find_repptokenizer(self, repp_dirname):
        """
        A module to find REPP tokenizer binary and its *repp.set* config file.
        """
        if os.path.exists(repp_dirname):  # If a full path is given.
            _repp_dir = repp_dirname
        else:  # Try to find path to REPP directory in environment variables.
            _repp_dir = find_dir(repp_dirname, env_vars=("REPP_TOKENIZER",))
        # Checks for the REPP binary and erg/repp.set config file.
        assert os.path.exists(_repp_dir + "/src/repp")
        assert os.path.exists(_repp_dir + "/erg/repp.set")
        return _repp_dir
