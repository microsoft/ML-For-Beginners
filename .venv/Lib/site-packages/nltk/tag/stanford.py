# Natural Language Toolkit: Interface to the Stanford Part-of-speech and Named-Entity Taggers
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Nitin Madnani <nmadnani@ets.org>
#         Rami Al-Rfou' <ralrfou@cs.stonybrook.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A module for interfacing with the Stanford taggers.

Tagger models need to be downloaded from https://nlp.stanford.edu/software
and the STANFORD_MODELS environment variable set (a colon-separated
list of paths).

For more details see the documentation for StanfordPOSTagger and StanfordNERTagger.
"""

import os
import tempfile
import warnings
from abc import abstractmethod
from subprocess import PIPE

from nltk.internals import _java_options, config_java, find_file, find_jar, java
from nltk.tag.api import TaggerI

_stanford_url = "https://nlp.stanford.edu/software"


class StanfordTagger(TaggerI):
    """
    An interface to Stanford taggers. Subclasses must define:

    - ``_cmd`` property: A property that returns the command that will be
      executed.
    - ``_SEPARATOR``: Class constant that represents that character that
      is used to separate the tokens from their tags.
    - ``_JAR`` file: Class constant that represents the jar file name.
    """

    _SEPARATOR = ""
    _JAR = ""

    def __init__(
        self,
        model_filename,
        path_to_jar=None,
        encoding="utf8",
        verbose=False,
        java_options="-mx1000m",
    ):
        # Raise deprecation warning.
        warnings.warn(
            str(
                "\nThe StanfordTokenizer will "
                "be deprecated in version 3.2.6.\n"
                "Please use \033[91mnltk.parse.corenlp.CoreNLPParser\033[0m instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )

        if not self._JAR:
            warnings.warn(
                "The StanfordTagger class is not meant to be "
                "instantiated directly. Did you mean "
                "StanfordPOSTagger or StanfordNERTagger?"
            )
        self._stanford_jar = find_jar(
            self._JAR, path_to_jar, searchpath=(), url=_stanford_url, verbose=verbose
        )

        self._stanford_model = find_file(
            model_filename, env_vars=("STANFORD_MODELS",), verbose=verbose
        )

        self._encoding = encoding
        self.java_options = java_options

    @property
    @abstractmethod
    def _cmd(self):
        """
        A property that returns the command that will be executed.
        """

    def tag(self, tokens):
        # This function should return list of tuple rather than list of list
        return sum(self.tag_sents([tokens]), [])

    def tag_sents(self, sentences):
        encoding = self._encoding
        default_options = " ".join(_java_options)
        config_java(options=self.java_options, verbose=False)

        # Create a temporary input file
        _input_fh, self._input_file_path = tempfile.mkstemp(text=True)

        cmd = list(self._cmd)
        cmd.extend(["-encoding", encoding])

        # Write the actual sentences to the temporary input file
        _input_fh = os.fdopen(_input_fh, "wb")
        _input = "\n".join(" ".join(x) for x in sentences)
        if isinstance(_input, str) and encoding:
            _input = _input.encode(encoding)
        _input_fh.write(_input)
        _input_fh.close()

        # Run the tagger and get the output
        stanpos_output, _stderr = java(
            cmd, classpath=self._stanford_jar, stdout=PIPE, stderr=PIPE
        )
        stanpos_output = stanpos_output.decode(encoding)

        # Delete the temporary file
        os.unlink(self._input_file_path)

        # Return java configurations to their default values
        config_java(options=default_options, verbose=False)

        return self.parse_output(stanpos_output, sentences)

    def parse_output(self, text, sentences=None):
        # Output the tagged sentences
        tagged_sentences = []
        for tagged_sentence in text.strip().split("\n"):
            sentence = []
            for tagged_word in tagged_sentence.strip().split():
                word_tags = tagged_word.strip().split(self._SEPARATOR)
                sentence.append(
                    ("".join(word_tags[:-1]), word_tags[-1].replace("0", "").upper())
                )
            tagged_sentences.append(sentence)
        return tagged_sentences


class StanfordPOSTagger(StanfordTagger):
    """
    A class for pos tagging with Stanford Tagger. The input is the paths to:
     - a model trained on training data
     - (optionally) the path to the stanford tagger jar file. If not specified here,
       then this jar file must be specified in the CLASSPATH environment variable.
     - (optionally) the encoding of the training data (default: UTF-8)

    Example:

        >>> from nltk.tag import StanfordPOSTagger
        >>> st = StanfordPOSTagger('english-bidirectional-distsim.tagger') # doctest: +SKIP
        >>> st.tag('What is the airspeed of an unladen swallow ?'.split()) # doctest: +SKIP
        [('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'), ('of', 'IN'), ('an', 'DT'), ('unladen', 'JJ'), ('swallow', 'VB'), ('?', '.')]
    """

    _SEPARATOR = "_"
    _JAR = "stanford-postagger.jar"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _cmd(self):
        return [
            "edu.stanford.nlp.tagger.maxent.MaxentTagger",
            "-model",
            self._stanford_model,
            "-textFile",
            self._input_file_path,
            "-tokenize",
            "false",
            "-outputFormatOptions",
            "keepEmptySentences",
        ]


class StanfordNERTagger(StanfordTagger):
    """
    A class for Named-Entity Tagging with Stanford Tagger. The input is the paths to:

    - a model trained on training data
    - (optionally) the path to the stanford tagger jar file. If not specified here,
      then this jar file must be specified in the CLASSPATH environment variable.
    - (optionally) the encoding of the training data (default: UTF-8)

    Example:

        >>> from nltk.tag import StanfordNERTagger
        >>> st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz') # doctest: +SKIP
        >>> st.tag('Rami Eid is studying at Stony Brook University in NY'.split()) # doctest: +SKIP
        [('Rami', 'PERSON'), ('Eid', 'PERSON'), ('is', 'O'), ('studying', 'O'),
         ('at', 'O'), ('Stony', 'ORGANIZATION'), ('Brook', 'ORGANIZATION'),
         ('University', 'ORGANIZATION'), ('in', 'O'), ('NY', 'LOCATION')]
    """

    _SEPARATOR = "/"
    _JAR = "stanford-ner.jar"
    _FORMAT = "slashTags"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _cmd(self):
        # Adding -tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -tokenizerOptions tokenizeNLs=false for not using stanford Tokenizer
        return [
            "edu.stanford.nlp.ie.crf.CRFClassifier",
            "-loadClassifier",
            self._stanford_model,
            "-textFile",
            self._input_file_path,
            "-outputFormat",
            self._FORMAT,
            "-tokenizerFactory",
            "edu.stanford.nlp.process.WhitespaceTokenizer",
            "-tokenizerOptions",
            '"tokenizeNLs=false"',
        ]

    def parse_output(self, text, sentences):
        if self._FORMAT == "slashTags":
            # Joint together to a big list
            tagged_sentences = []
            for tagged_sentence in text.strip().split("\n"):
                for tagged_word in tagged_sentence.strip().split():
                    word_tags = tagged_word.strip().split(self._SEPARATOR)
                    tagged_sentences.append(("".join(word_tags[:-1]), word_tags[-1]))

            # Separate it according to the input
            result = []
            start = 0
            for sent in sentences:
                result.append(tagged_sentences[start : start + len(sent)])
                start += len(sent)
            return result

        raise NotImplementedError
