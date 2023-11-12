# Natural Language Toolkit: Discourse Processing
#
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
#         Dan Garrette <dhgarrette@gmail.com>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

r"""
Module for incrementally developing simple discourses, and checking for semantic ambiguity,
consistency and informativeness.

Many of the ideas are based on the CURT family of programs of Blackburn and Bos
(see http://homepages.inf.ed.ac.uk/jbos/comsem/book1.html).

Consistency checking is carried out  by using the ``mace`` module to call the Mace4 model builder.
Informativeness checking is carried out with a call to ``Prover.prove()`` from
the ``inference``  module.

``DiscourseTester`` is a constructor for discourses.
The basic data structure is a list of sentences, stored as ``self._sentences``. Each sentence in the list
is assigned a "sentence ID" (``sid``) of the form ``s``\ *i*. For example::

    s0: A boxer walks
    s1: Every boxer chases a girl

Each sentence can be ambiguous between a number of readings, each of which receives a
"reading ID" (``rid``) of the form ``s``\ *i* -``r``\ *j*. For example::

    s0 readings:

    s0-r1: some x.(boxer(x) & walk(x))
    s0-r0: some x.(boxerdog(x) & walk(x))

A "thread" is a list of readings, represented as a list of ``rid``\ s.
Each thread receives a "thread ID" (``tid``) of the form ``d``\ *i*.
For example::

    d0: ['s0-r0', 's1-r0']

The set of all threads for a discourse is the Cartesian product of all the readings of the sequences of sentences.
(This is not intended to scale beyond very short discourses!) The method ``readings(filter=True)`` will only show
those threads which are consistent (taking into account any background assumptions).
"""

import os
from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import add, and_

from nltk.data import show_cfg
from nltk.inference.mace import MaceCommand
from nltk.inference.prover9 import Prover9Command
from nltk.parse import load_parser
from nltk.parse.malt import MaltParser
from nltk.sem.drt import AnaphoraResolutionException, resolve_anaphora
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Expression
from nltk.tag import RegexpTagger


class ReadingCommand(metaclass=ABCMeta):
    @abstractmethod
    def parse_to_readings(self, sentence):
        """
        :param sentence: the sentence to read
        :type sentence: str
        """

    def process_thread(self, sentence_readings):
        """
        This method should be used to handle dependencies between readings such
        as resolving anaphora.

        :param sentence_readings: readings to process
        :type sentence_readings: list(Expression)
        :return: the list of readings after processing
        :rtype: list(Expression)
        """
        return sentence_readings

    @abstractmethod
    def combine_readings(self, readings):
        """
        :param readings: readings to combine
        :type readings: list(Expression)
        :return: one combined reading
        :rtype: Expression
        """

    @abstractmethod
    def to_fol(self, expression):
        """
        Convert this expression into a First-Order Logic expression.

        :param expression: an expression
        :type expression: Expression
        :return: a FOL version of the input expression
        :rtype: Expression
        """


class CfgReadingCommand(ReadingCommand):
    def __init__(self, gramfile=None):
        """
        :param gramfile: name of file where grammar can be loaded
        :type gramfile: str
        """
        self._gramfile = (
            gramfile if gramfile else "grammars/book_grammars/discourse.fcfg"
        )
        self._parser = load_parser(self._gramfile)

    def parse_to_readings(self, sentence):
        """:see: ReadingCommand.parse_to_readings()"""
        from nltk.sem import root_semrep

        tokens = sentence.split()
        trees = self._parser.parse(tokens)
        return [root_semrep(tree) for tree in trees]

    def combine_readings(self, readings):
        """:see: ReadingCommand.combine_readings()"""
        return reduce(and_, readings)

    def to_fol(self, expression):
        """:see: ReadingCommand.to_fol()"""
        return expression


class DrtGlueReadingCommand(ReadingCommand):
    def __init__(self, semtype_file=None, remove_duplicates=False, depparser=None):
        """
        :param semtype_file: name of file where grammar can be loaded
        :param remove_duplicates: should duplicates be removed?
        :param depparser: the dependency parser
        """
        if semtype_file is None:
            semtype_file = os.path.join(
                "grammars", "sample_grammars", "drt_glue.semtype"
            )
        self._glue = DrtGlue(
            semtype_file=semtype_file,
            remove_duplicates=remove_duplicates,
            depparser=depparser,
        )

    def parse_to_readings(self, sentence):
        """:see: ReadingCommand.parse_to_readings()"""
        return self._glue.parse_to_meaning(sentence)

    def process_thread(self, sentence_readings):
        """:see: ReadingCommand.process_thread()"""
        try:
            return [self.combine_readings(sentence_readings)]
        except AnaphoraResolutionException:
            return []

    def combine_readings(self, readings):
        """:see: ReadingCommand.combine_readings()"""
        thread_reading = reduce(add, readings)
        return resolve_anaphora(thread_reading.simplify())

    def to_fol(self, expression):
        """:see: ReadingCommand.to_fol()"""
        return expression.fol()


class DiscourseTester:
    """
    Check properties of an ongoing discourse.
    """

    def __init__(self, input, reading_command=None, background=None):
        """
        Initialize a ``DiscourseTester``.

        :param input: the discourse sentences
        :type input: list of str
        :param background: Formulas which express background assumptions
        :type background: list(Expression)
        """
        self._input = input
        self._sentences = {"s%s" % i: sent for i, sent in enumerate(input)}
        self._models = None
        self._readings = {}
        self._reading_command = (
            reading_command if reading_command else CfgReadingCommand()
        )
        self._threads = {}
        self._filtered_threads = {}
        if background is not None:
            from nltk.sem.logic import Expression

            for e in background:
                assert isinstance(e, Expression)
            self._background = background
        else:
            self._background = []

    ###############################
    # Sentences
    ###############################

    def sentences(self):
        """
        Display the list of sentences in the current discourse.
        """
        for id in sorted(self._sentences):
            print(f"{id}: {self._sentences[id]}")

    def add_sentence(self, sentence, informchk=False, consistchk=False):
        """
        Add a sentence to the current discourse.

        Updates ``self._input`` and ``self._sentences``.
        :param sentence: An input sentence
        :type sentence: str
        :param informchk: if ``True``, check that the result of adding the sentence is thread-informative. Updates ``self._readings``.
        :param consistchk: if ``True``, check that the result of adding the sentence is thread-consistent. Updates ``self._readings``.

        """
        # check whether the new sentence is informative (i.e. not entailed by the previous discourse)
        if informchk:
            self.readings(verbose=False)
            for tid in sorted(self._threads):
                assumptions = [reading for (rid, reading) in self.expand_threads(tid)]
                assumptions += self._background
                for sent_reading in self._get_readings(sentence):
                    tp = Prover9Command(goal=sent_reading, assumptions=assumptions)
                    if tp.prove():
                        print(
                            "Sentence '%s' under reading '%s':"
                            % (sentence, str(sent_reading))
                        )
                        print("Not informative relative to thread '%s'" % tid)

        self._input.append(sentence)
        self._sentences = {"s%s" % i: sent for i, sent in enumerate(self._input)}
        # check whether adding the new sentence to the discourse preserves consistency (i.e. a model can be found for the combined set of
        # of assumptions
        if consistchk:
            self.readings(verbose=False)
            self.models(show=False)

    def retract_sentence(self, sentence, verbose=True):
        """
        Remove a sentence from the current discourse.

        Updates ``self._input``, ``self._sentences`` and ``self._readings``.
        :param sentence: An input sentence
        :type sentence: str
        :param verbose: If ``True``,  report on the updated list of sentences.
        """
        try:
            self._input.remove(sentence)
        except ValueError:
            print(
                "Retraction failed. The sentence '%s' is not part of the current discourse:"
                % sentence
            )
            self.sentences()
            return None
        self._sentences = {"s%s" % i: sent for i, sent in enumerate(self._input)}
        self.readings(verbose=False)
        if verbose:
            print("Current sentences are ")
            self.sentences()

    def grammar(self):
        """
        Print out the grammar in use for parsing input sentences
        """
        show_cfg(self._reading_command._gramfile)

    ###############################
    # Readings and Threads
    ###############################

    def _get_readings(self, sentence):
        """
        Build a list of semantic readings for a sentence.

        :rtype: list(Expression)
        """
        return self._reading_command.parse_to_readings(sentence)

    def _construct_readings(self):
        """
        Use ``self._sentences`` to construct a value for ``self._readings``.
        """
        # re-initialize self._readings in case we have retracted a sentence
        self._readings = {}
        for sid in sorted(self._sentences):
            sentence = self._sentences[sid]
            readings = self._get_readings(sentence)
            self._readings[sid] = {
                f"{sid}-r{rid}": reading.simplify()
                for rid, reading in enumerate(sorted(readings, key=str))
            }

    def _construct_threads(self):
        """
        Use ``self._readings`` to construct a value for ``self._threads``
        and use the model builder to construct a value for ``self._filtered_threads``
        """
        thread_list = [[]]
        for sid in sorted(self._readings):
            thread_list = self.multiply(thread_list, sorted(self._readings[sid]))
        self._threads = {"d%s" % tid: thread for tid, thread in enumerate(thread_list)}
        # re-initialize the filtered threads
        self._filtered_threads = {}
        # keep the same ids, but only include threads which get models
        consistency_checked = self._check_consistency(self._threads)
        for (tid, thread) in self._threads.items():
            if (tid, True) in consistency_checked:
                self._filtered_threads[tid] = thread

    def _show_readings(self, sentence=None):
        """
        Print out the readings for  the discourse (or a single sentence).
        """
        if sentence is not None:
            print("The sentence '%s' has these readings:" % sentence)
            for r in [str(reading) for reading in (self._get_readings(sentence))]:
                print("    %s" % r)
        else:
            for sid in sorted(self._readings):
                print()
                print("%s readings:" % sid)
                print()  #'-' * 30
                for rid in sorted(self._readings[sid]):
                    lf = self._readings[sid][rid]
                    print(f"{rid}: {lf.normalize()}")

    def _show_threads(self, filter=False, show_thread_readings=False):
        """
        Print out the value of ``self._threads`` or ``self._filtered_hreads``
        """
        threads = self._filtered_threads if filter else self._threads
        for tid in sorted(threads):
            if show_thread_readings:
                readings = [
                    self._readings[rid.split("-")[0]][rid] for rid in self._threads[tid]
                ]
                try:
                    thread_reading = (
                        ": %s"
                        % self._reading_command.combine_readings(readings).normalize()
                    )
                except Exception as e:
                    thread_reading = ": INVALID: %s" % e.__class__.__name__
            else:
                thread_reading = ""

            print("%s:" % tid, self._threads[tid], thread_reading)

    def readings(
        self,
        sentence=None,
        threaded=False,
        verbose=True,
        filter=False,
        show_thread_readings=False,
    ):
        """
        Construct and show the readings of the discourse (or of a single sentence).

        :param sentence: test just this sentence
        :type sentence: str
        :param threaded: if ``True``, print out each thread ID and the corresponding thread.
        :param filter: if ``True``, only print out consistent thread IDs and threads.
        """
        self._construct_readings()
        self._construct_threads()

        # if we are filtering or showing thread readings, show threads
        if filter or show_thread_readings:
            threaded = True

        if verbose:
            if not threaded:
                self._show_readings(sentence=sentence)
            else:
                self._show_threads(
                    filter=filter, show_thread_readings=show_thread_readings
                )

    def expand_threads(self, thread_id, threads=None):
        """
        Given a thread ID, find the list of ``logic.Expression`` objects corresponding to the reading IDs in that thread.

        :param thread_id: thread ID
        :type thread_id: str
        :param threads: a mapping from thread IDs to lists of reading IDs
        :type threads: dict
        :return: A list of pairs ``(rid, reading)`` where reading is the ``logic.Expression`` associated with a reading ID
        :rtype: list of tuple
        """
        if threads is None:
            threads = self._threads
        return [
            (rid, self._readings[sid][rid])
            for rid in threads[thread_id]
            for sid in rid.split("-")[:1]
        ]

    ###############################
    # Models and Background
    ###############################

    def _check_consistency(self, threads, show=False, verbose=False):
        results = []
        for tid in sorted(threads):
            assumptions = [
                reading for (rid, reading) in self.expand_threads(tid, threads=threads)
            ]
            assumptions = list(
                map(
                    self._reading_command.to_fol,
                    self._reading_command.process_thread(assumptions),
                )
            )
            if assumptions:
                assumptions += self._background
                # if Mace4 finds a model, it always seems to find it quickly
                mb = MaceCommand(None, assumptions, max_models=20)
                modelfound = mb.build_model()
            else:
                modelfound = False
            results.append((tid, modelfound))
            if show:
                spacer(80)
                print("Model for Discourse Thread %s" % tid)
                spacer(80)
                if verbose:
                    for a in assumptions:
                        print(a)
                    spacer(80)
                if modelfound:
                    print(mb.model(format="cooked"))
                else:
                    print("No model found!\n")
        return results

    def models(self, thread_id=None, show=True, verbose=False):
        """
        Call Mace4 to build a model for each current discourse thread.

        :param thread_id: thread ID
        :type thread_id: str
        :param show: If ``True``, display the model that has been found.
        """
        self._construct_readings()
        self._construct_threads()
        threads = {thread_id: self._threads[thread_id]} if thread_id else self._threads

        for (tid, modelfound) in self._check_consistency(
            threads, show=show, verbose=verbose
        ):
            idlist = [rid for rid in threads[tid]]

            if not modelfound:
                print(f"Inconsistent discourse: {tid} {idlist}:")
                for rid, reading in self.expand_threads(tid):
                    print(f"    {rid}: {reading.normalize()}")
                print()
            else:
                print(f"Consistent discourse: {tid} {idlist}:")
                for rid, reading in self.expand_threads(tid):
                    print(f"    {rid}: {reading.normalize()}")
                print()

    def add_background(self, background, verbose=False):
        """
        Add a list of background assumptions for reasoning about the discourse.

        When called,  this method also updates the discourse model's set of readings and threads.
        :param background: Formulas which contain background information
        :type background: list(Expression)
        """
        from nltk.sem.logic import Expression

        for (count, e) in enumerate(background):
            assert isinstance(e, Expression)
            if verbose:
                print("Adding assumption %s to background" % count)
            self._background.append(e)

        # update the state
        self._construct_readings()
        self._construct_threads()

    def background(self):
        """
        Show the current background assumptions.
        """
        for e in self._background:
            print(str(e))

    ###############################
    # Misc
    ###############################

    @staticmethod
    def multiply(discourse, readings):
        """
        Multiply every thread in ``discourse`` by every reading in ``readings``.

        Given discourse = [['A'], ['B']], readings = ['a', 'b', 'c'] , returns
        [['A', 'a'], ['A', 'b'], ['A', 'c'], ['B', 'a'], ['B', 'b'], ['B', 'c']]

        :param discourse: the current list of readings
        :type discourse: list of lists
        :param readings: an additional list of readings
        :type readings: list(Expression)
        :rtype: A list of lists
        """
        result = []
        for sublist in discourse:
            for r in readings:
                new = []
                new += sublist
                new.append(r)
                result.append(new)
        return result


def load_fol(s):
    """
    Temporarily duplicated from ``nltk.sem.util``.
    Convert a  file of first order formulas into a list of ``Expression`` objects.

    :param s: the contents of the file
    :type s: str
    :return: a list of parsed formulas.
    :rtype: list(Expression)
    """
    statements = []
    for linenum, line in enumerate(s.splitlines()):
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        try:
            statements.append(Expression.fromstring(line))
        except Exception as e:
            raise ValueError(f"Unable to parse line {linenum}: {line}") from e
    return statements


###############################
# Demo
###############################
def discourse_demo(reading_command=None):
    """
    Illustrate the various methods of ``DiscourseTester``
    """
    dt = DiscourseTester(
        ["A boxer walks", "Every boxer chases a girl"], reading_command
    )
    dt.models()
    print()
    # dt.grammar()
    print()
    dt.sentences()
    print()
    dt.readings()
    print()
    dt.readings(threaded=True)
    print()
    dt.models("d1")
    dt.add_sentence("John is a boxer")
    print()
    dt.sentences()
    print()
    dt.readings(threaded=True)
    print()
    dt = DiscourseTester(
        ["A student dances", "Every student is a person"], reading_command
    )
    print()
    dt.add_sentence("No person dances", consistchk=True)
    print()
    dt.readings()
    print()
    dt.retract_sentence("No person dances", verbose=True)
    print()
    dt.models()
    print()
    dt.readings("A person dances")
    print()
    dt.add_sentence("A person dances", informchk=True)
    dt = DiscourseTester(
        ["Vincent is a boxer", "Fido is a boxer", "Vincent is married", "Fido barks"],
        reading_command,
    )
    dt.readings(filter=True)
    import nltk.data

    background_file = os.path.join("grammars", "book_grammars", "background.fol")
    background = nltk.data.load(background_file)

    print()
    dt.add_background(background, verbose=False)
    dt.background()
    print()
    dt.readings(filter=True)
    print()
    dt.models()


def drt_discourse_demo(reading_command=None):
    """
    Illustrate the various methods of ``DiscourseTester``
    """
    dt = DiscourseTester(["every dog chases a boy", "he runs"], reading_command)
    dt.models()
    print()
    dt.sentences()
    print()
    dt.readings()
    print()
    dt.readings(show_thread_readings=True)
    print()
    dt.readings(filter=True, show_thread_readings=True)


def spacer(num=30):
    print("-" * num)


def demo():
    discourse_demo()

    tagger = RegexpTagger(
        [
            ("^(chases|runs)$", "VB"),
            ("^(a)$", "ex_quant"),
            ("^(every)$", "univ_quant"),
            ("^(dog|boy)$", "NN"),
            ("^(he)$", "PRP"),
        ]
    )
    depparser = MaltParser(tagger=tagger)
    drt_discourse_demo(
        DrtGlueReadingCommand(remove_duplicates=False, depparser=depparser)
    )


if __name__ == "__main__":
    demo()
