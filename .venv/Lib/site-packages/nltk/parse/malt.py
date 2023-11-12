# Natural Language Toolkit: Interface to MaltParser
#
# Author: Dan Garrette <dhgarrette@gmail.com>
# Contributor: Liling Tan, Mustufain, osamamukhtar11
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import inspect
import os
import subprocess
import sys
import tempfile

from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir, find_file, find_jars_within_path
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.parse.util import taggedsents_to_conll


def malt_regex_tagger():
    from nltk.tag import RegexpTagger

    _tagger = RegexpTagger(
        [
            (r"\.$", "."),
            (r"\,$", ","),
            (r"\?$", "?"),  # fullstop, comma, Qmark
            (r"\($", "("),
            (r"\)$", ")"),  # round brackets
            (r"\[$", "["),
            (r"\]$", "]"),  # square brackets
            (r"^-?[0-9]+(\.[0-9]+)?$", "CD"),  # cardinal numbers
            (r"(The|the|A|a|An|an)$", "DT"),  # articles
            (r"(He|he|She|she|It|it|I|me|Me|You|you)$", "PRP"),  # pronouns
            (r"(His|his|Her|her|Its|its)$", "PRP$"),  # possessive
            (r"(my|Your|your|Yours|yours)$", "PRP$"),  # possessive
            (r"(on|On|in|In|at|At|since|Since)$", "IN"),  # time prepopsitions
            (r"(for|For|ago|Ago|before|Before)$", "IN"),  # time prepopsitions
            (r"(till|Till|until|Until)$", "IN"),  # time prepopsitions
            (r"(by|By|beside|Beside)$", "IN"),  # space prepopsitions
            (r"(under|Under|below|Below)$", "IN"),  # space prepopsitions
            (r"(over|Over|above|Above)$", "IN"),  # space prepopsitions
            (r"(across|Across|through|Through)$", "IN"),  # space prepopsitions
            (r"(into|Into|towards|Towards)$", "IN"),  # space prepopsitions
            (r"(onto|Onto|from|From)$", "IN"),  # space prepopsitions
            (r".*able$", "JJ"),  # adjectives
            (r".*ness$", "NN"),  # nouns formed from adjectives
            (r".*ly$", "RB"),  # adverbs
            (r".*s$", "NNS"),  # plural nouns
            (r".*ing$", "VBG"),  # gerunds
            (r".*ed$", "VBD"),  # past tense verbs
            (r".*", "NN"),  # nouns (default)
        ]
    )
    return _tagger.tag


def find_maltparser(parser_dirname):
    """
    A module to find MaltParser .jar file and its dependencies.
    """
    if os.path.exists(parser_dirname):  # If a full path is given.
        _malt_dir = parser_dirname
    else:  # Try to find path to maltparser directory in environment variables.
        _malt_dir = find_dir(parser_dirname, env_vars=("MALT_PARSER",))
    # Checks that that the found directory contains all the necessary .jar
    malt_dependencies = ["", "", ""]
    _malt_jars = set(find_jars_within_path(_malt_dir))
    _jars = {os.path.split(jar)[1] for jar in _malt_jars}
    malt_dependencies = {"log4j.jar", "libsvm.jar", "liblinear-1.8.jar"}

    assert malt_dependencies.issubset(_jars)
    assert any(
        filter(lambda i: i.startswith("maltparser-") and i.endswith(".jar"), _jars)
    )
    return list(_malt_jars)


def find_malt_model(model_filename):
    """
    A module to find pre-trained MaltParser model.
    """
    if model_filename is None:
        return "malt_temp.mco"
    elif os.path.exists(model_filename):  # If a full path is given.
        return model_filename
    else:  # Try to find path to malt model in environment variables.
        return find_file(model_filename, env_vars=("MALT_MODEL",), verbose=False)


class MaltParser(ParserI):
    """
    A class for dependency parsing with MaltParser. The input is the paths to:
    - (optionally) a maltparser directory
    - (optionally) the path to a pre-trained MaltParser .mco model file
    - (optionally) the tagger to use for POS tagging before parsing
    - (optionally) additional Java arguments

    Example:
        >>> from nltk.parse import malt
        >>> # With MALT_PARSER and MALT_MODEL environment set.
        >>> mp = malt.MaltParser(model_filename='engmalt.linear-1.7.mco') # doctest: +SKIP
        >>> mp.parse_one('I shot an elephant in my pajamas .'.split()).tree() # doctest: +SKIP
        (shot I (elephant an) (in (pajamas my)) .)
        >>> # Without MALT_PARSER and MALT_MODEL environment.
        >>> mp = malt.MaltParser('/home/user/maltparser-1.9.2/', '/home/user/engmalt.linear-1.7.mco') # doctest: +SKIP
        >>> mp.parse_one('I shot an elephant in my pajamas .'.split()).tree() # doctest: +SKIP
        (shot I (elephant an) (in (pajamas my)) .)
    """

    def __init__(
        self,
        parser_dirname="",
        model_filename=None,
        tagger=None,
        additional_java_args=None,
    ):
        """
        An interface for parsing with the Malt Parser.

        :param parser_dirname: The path to the maltparser directory that
            contains the maltparser-1.x.jar
        :type parser_dirname: str
        :param model_filename: The name of the pre-trained model with .mco file
            extension. If provided, training will not be required.
            (see http://www.maltparser.org/mco/mco.html and
            see http://www.patful.com/chalk/node/185)
        :type model_filename: str
        :param tagger: The tagger used to POS tag the raw string before
            formatting to CONLL format. It should behave like `nltk.pos_tag`
        :type tagger: function
        :param additional_java_args: This is the additional Java arguments that
            one can use when calling Maltparser, usually this is the heapsize
            limits, e.g. `additional_java_args=['-Xmx1024m']`
            (see https://goo.gl/mpDBvQ)
        :type additional_java_args: list
        """

        # Find all the necessary jar files for MaltParser.
        self.malt_jars = find_maltparser(parser_dirname)
        # Initialize additional java arguments.
        self.additional_java_args = (
            additional_java_args if additional_java_args is not None else []
        )
        # Initialize model.
        self.model = find_malt_model(model_filename)
        self._trained = self.model != "malt_temp.mco"
        # Set the working_dir parameters i.e. `-w` from MaltParser's option.
        self.working_dir = tempfile.gettempdir()
        # Initialize POS tagger.
        self.tagger = tagger if tagger is not None else malt_regex_tagger()

    def parse_tagged_sents(self, sentences, verbose=False, top_relation_label="null"):
        """
        Use MaltParser to parse multiple POS tagged sentences. Takes multiple
        sentences where each sentence is a list of (word, tag) tuples.
        The sentences must have already been tokenized and tagged.

        :param sentences: Input sentences to parse
        :type sentence: list(list(tuple(str, str)))
        :return: iter(iter(``DependencyGraph``)) the dependency graph
            representation of each sentence
        """
        if not self._trained:
            raise Exception("Parser has not been trained. Call train() first.")

        with tempfile.NamedTemporaryFile(
            prefix="malt_input.conll.", dir=self.working_dir, mode="w", delete=False
        ) as input_file:
            with tempfile.NamedTemporaryFile(
                prefix="malt_output.conll.",
                dir=self.working_dir,
                mode="w",
                delete=False,
            ) as output_file:
                # Convert list of sentences to CONLL format.
                for line in taggedsents_to_conll(sentences):
                    input_file.write(str(line))
                input_file.close()

                # Generate command to run maltparser.
                cmd = self.generate_malt_command(
                    input_file.name, output_file.name, mode="parse"
                )

                # This is a maltparser quirk, it needs to be run
                # where the model file is. otherwise it goes into an awkward
                # missing .jars or strange -w working_dir problem.
                _current_path = os.getcwd()  # Remembers the current path.
                try:  # Change to modelfile path
                    os.chdir(os.path.split(self.model)[0])
                except:
                    pass
                ret = self._execute(cmd, verbose)  # Run command.
                os.chdir(_current_path)  # Change back to current path.

                if ret != 0:
                    raise Exception(
                        "MaltParser parsing (%s) failed with exit "
                        "code %d" % (" ".join(cmd), ret)
                    )

                # Must return iter(iter(Tree))
                with open(output_file.name) as infile:
                    for tree_str in infile.read().split("\n\n"):
                        yield (
                            iter(
                                [
                                    DependencyGraph(
                                        tree_str, top_relation_label=top_relation_label
                                    )
                                ]
                            )
                        )

        os.remove(input_file.name)
        os.remove(output_file.name)

    def parse_sents(self, sentences, verbose=False, top_relation_label="null"):
        """
        Use MaltParser to parse multiple sentences.
        Takes a list of sentences, where each sentence is a list of words.
        Each sentence will be automatically tagged with this
        MaltParser instance's tagger.

        :param sentences: Input sentences to parse
        :type sentence: list(list(str))
        :return: iter(DependencyGraph)
        """
        tagged_sentences = (self.tagger(sentence) for sentence in sentences)
        return self.parse_tagged_sents(
            tagged_sentences, verbose, top_relation_label=top_relation_label
        )

    def generate_malt_command(self, inputfilename, outputfilename=None, mode=None):
        """
        This function generates the maltparser command use at the terminal.

        :param inputfilename: path to the input file
        :type inputfilename: str
        :param outputfilename: path to the output file
        :type outputfilename: str
        """

        cmd = ["java"]
        cmd += self.additional_java_args  # Adds additional java arguments
        # Joins classpaths with ";" if on Windows and on Linux/Mac use ":"
        classpaths_separator = ";" if sys.platform.startswith("win") else ":"
        cmd += [
            "-cp",
            classpaths_separator.join(self.malt_jars),
        ]  # Adds classpaths for jars
        cmd += ["org.maltparser.Malt"]  # Adds the main function.

        # Adds the model file.
        if os.path.exists(self.model):  # when parsing
            cmd += ["-c", os.path.split(self.model)[-1]]
        else:  # when learning
            cmd += ["-c", self.model]

        cmd += ["-i", inputfilename]
        if mode == "parse":
            cmd += ["-o", outputfilename]
        cmd += ["-m", mode]  # mode use to generate parses.
        return cmd

    @staticmethod
    def _execute(cmd, verbose=False):
        output = None if verbose else subprocess.PIPE
        p = subprocess.Popen(cmd, stdout=output, stderr=output)
        return p.wait()

    def train(self, depgraphs, verbose=False):
        """
        Train MaltParser from a list of ``DependencyGraph`` objects

        :param depgraphs: list of ``DependencyGraph`` objects for training input data
        :type depgraphs: DependencyGraph
        """

        # Write the conll_str to malt_train.conll file in /tmp/
        with tempfile.NamedTemporaryFile(
            prefix="malt_train.conll.", dir=self.working_dir, mode="w", delete=False
        ) as input_file:
            input_str = "\n".join(dg.to_conll(10) for dg in depgraphs)
            input_file.write(str(input_str))
        # Trains the model with the malt_train.conll
        self.train_from_file(input_file.name, verbose=verbose)
        # Removes the malt_train.conll once training finishes.
        os.remove(input_file.name)

    def train_from_file(self, conll_file, verbose=False):
        """
        Train MaltParser from a file
        :param conll_file: str for the filename of the training input data
        :type conll_file: str
        """

        # If conll_file is a ZipFilePathPointer,
        # then we need to do some extra massaging
        if isinstance(conll_file, ZipFilePathPointer):
            with tempfile.NamedTemporaryFile(
                prefix="malt_train.conll.", dir=self.working_dir, mode="w", delete=False
            ) as input_file:
                with conll_file.open() as conll_input_file:
                    conll_str = conll_input_file.read()
                    input_file.write(str(conll_str))
                return self.train_from_file(input_file.name, verbose=verbose)

        # Generate command to run maltparser.
        cmd = self.generate_malt_command(conll_file, mode="learn")
        ret = self._execute(cmd, verbose)
        if ret != 0:
            raise Exception(
                "MaltParser training (%s) failed with exit "
                "code %d" % (" ".join(cmd), ret)
            )
        self._trained = True


if __name__ == "__main__":
    """
    A demonstration function to show how NLTK users can use the malt parser API.

    >>> from nltk import pos_tag
    >>> assert 'MALT_PARSER' in os.environ, str(
    ... "Please set MALT_PARSER in your global environment, e.g.:\n"
    ... "$ export MALT_PARSER='/home/user/maltparser-1.9.2/'")
    >>>
    >>> assert 'MALT_MODEL' in os.environ, str(
    ... "Please set MALT_MODEL in your global environment, e.g.:\n"
    ... "$ export MALT_MODEL='/home/user/engmalt.linear-1.7.mco'")
    >>>
    >>> _dg1_str = str("1    John    _    NNP   _    _    2    SUBJ    _    _\n"
    ...             "2    sees    _    VB    _    _    0    ROOT    _    _\n"
    ...             "3    a       _    DT    _    _    4    SPEC    _    _\n"
    ...             "4    dog     _    NN    _    _    2    OBJ     _    _\n"
    ...             "5    .     _    .    _    _    2    PUNCT     _    _\n")
    >>>
    >>>
    >>> _dg2_str  = str("1    John    _    NNP   _    _    2    SUBJ    _    _\n"
    ...             "2    walks   _    VB    _    _    0    ROOT    _    _\n"
    ...             "3    .     _    .    _    _    2    PUNCT     _    _\n")
    >>> dg1 = DependencyGraph(_dg1_str)
    >>> dg2 = DependencyGraph(_dg2_str)
    >>> # Initialize a MaltParser object
    >>> mp = MaltParser()
    >>>
    >>> # Trains a model.
    >>> mp.train([dg1,dg2], verbose=False)
    >>> sent1 = ['John','sees','Mary', '.']
    >>> sent2 = ['John', 'walks', 'a', 'dog', '.']
    >>>
    >>> # Parse a single sentence.
    >>> parsed_sent1 = mp.parse_one(sent1)
    >>> parsed_sent2 = mp.parse_one(sent2)
    >>> print(parsed_sent1.tree())
    (sees John Mary .)
    >>> print(parsed_sent2.tree())
    (walks John (dog a) .)
    >>>
    >>> # Parsing multiple sentences.
    >>> sentences = [sent1,sent2]
    >>> parsed_sents = mp.parse_sents(sentences)
    >>> print(next(next(parsed_sents)).tree())
    (sees John Mary .)
    >>> print(next(next(parsed_sents)).tree())
    (walks John (dog a) .)
    >>>
    >>> # Initialize a MaltParser object with an English pre-trained model.
    >>> parser_dirname = 'maltparser-1.9.2'
    >>> model_name = 'engmalt.linear-1.7.mco'
    >>> mp = MaltParser(parser_dirname=parser_dirname, model_filename=model_name, tagger=pos_tag)
    >>> sent1 = 'I shot an elephant in my pajamas .'.split()
    >>> sent2 = 'Time flies like banana .'.split()
    >>> # Parse a single sentence.
    >>> print(mp.parse_one(sent1).tree())
    (shot I (elephant an) (in (pajamas my)) .)
    # Parsing multiple sentences
    >>> sentences = [sent1,sent2]
    >>> parsed_sents = mp.parse_sents(sentences)
    >>> print(next(next(parsed_sents)).tree())
    (shot I (elephant an) (in (pajamas my)) .)
    >>> print(next(next(parsed_sents)).tree())
    (flies Time (like banana) .)
    """

    import doctest

    doctest.testmod()
