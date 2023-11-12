# Natural Language Toolkit: Classifier Interface
#
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
#         Dan Garrette <dhgarrette@gmail.com>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Interfaces and base classes for theorem provers and model builders.

``Prover`` is a standard interface for a theorem prover which tries to prove a goal from a
list of assumptions.

``ModelBuilder`` is a standard interface for a model builder. Given just a set of assumptions.
the model builder tries to build a model for the assumptions. Given a set of assumptions and a
goal *G*, the model builder tries to find a counter-model, in the sense of a model that will satisfy
the assumptions plus the negation of *G*.
"""

import threading
import time
from abc import ABCMeta, abstractmethod


class Prover(metaclass=ABCMeta):
    """
    Interface for trying to prove a goal from assumptions.  Both the goal and
    the assumptions are constrained to be formulas of ``logic.Expression``.
    """

    def prove(self, goal=None, assumptions=None, verbose=False):
        """
        :return: Whether the proof was successful or not.
        :rtype: bool
        """
        return self._prove(goal, assumptions, verbose)[0]

    @abstractmethod
    def _prove(self, goal=None, assumptions=None, verbose=False):
        """
        :return: Whether the proof was successful or not, along with the proof
        :rtype: tuple: (bool, str)
        """


class ModelBuilder(metaclass=ABCMeta):
    """
    Interface for trying to build a model of set of formulas.
    Open formulas are assumed to be universally quantified.
    Both the goal and the assumptions are constrained to be formulas
    of ``logic.Expression``.
    """

    def build_model(self, goal=None, assumptions=None, verbose=False):
        """
        Perform the actual model building.
        :return: Whether a model was generated
        :rtype: bool
        """
        return self._build_model(goal, assumptions, verbose)[0]

    @abstractmethod
    def _build_model(self, goal=None, assumptions=None, verbose=False):
        """
        Perform the actual model building.
        :return: Whether a model was generated, and the model itself
        :rtype: tuple(bool, sem.Valuation)
        """


class TheoremToolCommand(metaclass=ABCMeta):
    """
    This class holds a goal and a list of assumptions to be used in proving
    or model building.
    """

    @abstractmethod
    def add_assumptions(self, new_assumptions):
        """
        Add new assumptions to the assumption list.

        :param new_assumptions: new assumptions
        :type new_assumptions: list(sem.Expression)
        """

    @abstractmethod
    def retract_assumptions(self, retracted, debug=False):
        """
        Retract assumptions from the assumption list.

        :param debug: If True, give warning when ``retracted`` is not present on
            assumptions list.
        :type debug: bool
        :param retracted: assumptions to be retracted
        :type retracted: list(sem.Expression)
        """

    @abstractmethod
    def assumptions(self):
        """
        List the current assumptions.

        :return: list of ``Expression``
        """

    @abstractmethod
    def goal(self):
        """
        Return the goal

        :return: ``Expression``
        """

    @abstractmethod
    def print_assumptions(self):
        """
        Print the list of the current assumptions.
        """


class ProverCommand(TheoremToolCommand):
    """
    This class holds a ``Prover``, a goal, and a list of assumptions.  When
    prove() is called, the ``Prover`` is executed with the goal and assumptions.
    """

    @abstractmethod
    def prove(self, verbose=False):
        """
        Perform the actual proof.
        """

    @abstractmethod
    def proof(self, simplify=True):
        """
        Return the proof string
        :param simplify: bool simplify the proof?
        :return: str
        """

    @abstractmethod
    def get_prover(self):
        """
        Return the prover object
        :return: ``Prover``
        """


class ModelBuilderCommand(TheoremToolCommand):
    """
    This class holds a ``ModelBuilder``, a goal, and a list of assumptions.
    When build_model() is called, the ``ModelBuilder`` is executed with the goal
    and assumptions.
    """

    @abstractmethod
    def build_model(self, verbose=False):
        """
        Perform the actual model building.
        :return: A model if one is generated; None otherwise.
        :rtype: sem.Valuation
        """

    @abstractmethod
    def model(self, format=None):
        """
        Return a string representation of the model

        :param simplify: bool simplify the proof?
        :return: str
        """

    @abstractmethod
    def get_model_builder(self):
        """
        Return the model builder object
        :return: ``ModelBuilder``
        """


class BaseTheoremToolCommand(TheoremToolCommand):
    """
    This class holds a goal and a list of assumptions to be used in proving
    or model building.
    """

    def __init__(self, goal=None, assumptions=None):
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        """
        self._goal = goal

        if not assumptions:
            self._assumptions = []
        else:
            self._assumptions = list(assumptions)

        self._result = None
        """A holder for the result, to prevent unnecessary re-proving"""

    def add_assumptions(self, new_assumptions):
        """
        Add new assumptions to the assumption list.

        :param new_assumptions: new assumptions
        :type new_assumptions: list(sem.Expression)
        """
        self._assumptions.extend(new_assumptions)
        self._result = None

    def retract_assumptions(self, retracted, debug=False):
        """
        Retract assumptions from the assumption list.

        :param debug: If True, give warning when ``retracted`` is not present on
            assumptions list.
        :type debug: bool
        :param retracted: assumptions to be retracted
        :type retracted: list(sem.Expression)
        """
        retracted = set(retracted)
        result_list = list(filter(lambda a: a not in retracted, self._assumptions))
        if debug and result_list == self._assumptions:
            print(Warning("Assumptions list has not been changed:"))
            self.print_assumptions()

        self._assumptions = result_list

        self._result = None

    def assumptions(self):
        """
        List the current assumptions.

        :return: list of ``Expression``
        """
        return self._assumptions

    def goal(self):
        """
        Return the goal

        :return: ``Expression``
        """
        return self._goal

    def print_assumptions(self):
        """
        Print the list of the current assumptions.
        """
        for a in self.assumptions():
            print(a)


class BaseProverCommand(BaseTheoremToolCommand, ProverCommand):
    """
    This class holds a ``Prover``, a goal, and a list of assumptions.  When
    prove() is called, the ``Prover`` is executed with the goal and assumptions.
    """

    def __init__(self, prover, goal=None, assumptions=None):
        """
        :param prover: The theorem tool to execute with the assumptions
        :type prover: Prover
        :see: ``BaseTheoremToolCommand``
        """
        self._prover = prover
        """The theorem tool to execute with the assumptions"""

        BaseTheoremToolCommand.__init__(self, goal, assumptions)

        self._proof = None

    def prove(self, verbose=False):
        """
        Perform the actual proof.  Store the result to prevent unnecessary
        re-proving.
        """
        if self._result is None:
            self._result, self._proof = self._prover._prove(
                self.goal(), self.assumptions(), verbose
            )
        return self._result

    def proof(self, simplify=True):
        """
        Return the proof string
        :param simplify: bool simplify the proof?
        :return: str
        """
        if self._result is None:
            raise LookupError("You have to call prove() first to get a proof!")
        else:
            return self.decorate_proof(self._proof, simplify)

    def decorate_proof(self, proof_string, simplify=True):
        """
        Modify and return the proof string
        :param proof_string: str the proof to decorate
        :param simplify: bool simplify the proof?
        :return: str
        """
        return proof_string

    def get_prover(self):
        return self._prover


class BaseModelBuilderCommand(BaseTheoremToolCommand, ModelBuilderCommand):
    """
    This class holds a ``ModelBuilder``, a goal, and a list of assumptions.  When
    build_model() is called, the ``ModelBuilder`` is executed with the goal and
    assumptions.
    """

    def __init__(self, modelbuilder, goal=None, assumptions=None):
        """
        :param modelbuilder: The theorem tool to execute with the assumptions
        :type modelbuilder: ModelBuilder
        :see: ``BaseTheoremToolCommand``
        """
        self._modelbuilder = modelbuilder
        """The theorem tool to execute with the assumptions"""

        BaseTheoremToolCommand.__init__(self, goal, assumptions)

        self._model = None

    def build_model(self, verbose=False):
        """
        Attempt to build a model.  Store the result to prevent unnecessary
        re-building.
        """
        if self._result is None:
            self._result, self._model = self._modelbuilder._build_model(
                self.goal(), self.assumptions(), verbose
            )
        return self._result

    def model(self, format=None):
        """
        Return a string representation of the model

        :param simplify: bool simplify the proof?
        :return: str
        """
        if self._result is None:
            raise LookupError("You have to call build_model() first to " "get a model!")
        else:
            return self._decorate_model(self._model, format)

    def _decorate_model(self, valuation_str, format=None):
        """
        :param valuation_str: str with the model builder's output
        :param format: str indicating the format for displaying
        :return: str
        """
        return valuation_str

    def get_model_builder(self):
        return self._modelbuilder


class TheoremToolCommandDecorator(TheoremToolCommand):
    """
    A base decorator for the ``ProverCommandDecorator`` and
    ``ModelBuilderCommandDecorator`` classes from which decorators can extend.
    """

    def __init__(self, command):
        """
        :param command: ``TheoremToolCommand`` to decorate
        """
        self._command = command

        # The decorator has its own versions of 'result' different from the
        # underlying command
        self._result = None

    def assumptions(self):
        return self._command.assumptions()

    def goal(self):
        return self._command.goal()

    def add_assumptions(self, new_assumptions):
        self._command.add_assumptions(new_assumptions)
        self._result = None

    def retract_assumptions(self, retracted, debug=False):
        self._command.retract_assumptions(retracted, debug)
        self._result = None

    def print_assumptions(self):
        self._command.print_assumptions()


class ProverCommandDecorator(TheoremToolCommandDecorator, ProverCommand):
    """
    A base decorator for the ``ProverCommand`` class from which other
    prover command decorators can extend.
    """

    def __init__(self, proverCommand):
        """
        :param proverCommand: ``ProverCommand`` to decorate
        """
        TheoremToolCommandDecorator.__init__(self, proverCommand)

        # The decorator has its own versions of 'result' and 'proof'
        # because they may be different from the underlying command
        self._proof = None

    def prove(self, verbose=False):
        if self._result is None:
            prover = self.get_prover()
            self._result, self._proof = prover._prove(
                self.goal(), self.assumptions(), verbose
            )
        return self._result

    def proof(self, simplify=True):
        """
        Return the proof string
        :param simplify: bool simplify the proof?
        :return: str
        """
        if self._result is None:
            raise LookupError("You have to call prove() first to get a proof!")
        else:
            return self.decorate_proof(self._proof, simplify)

    def decorate_proof(self, proof_string, simplify=True):
        """
        Modify and return the proof string
        :param proof_string: str the proof to decorate
        :param simplify: bool simplify the proof?
        :return: str
        """
        return self._command.decorate_proof(proof_string, simplify)

    def get_prover(self):
        return self._command.get_prover()


class ModelBuilderCommandDecorator(TheoremToolCommandDecorator, ModelBuilderCommand):
    """
    A base decorator for the ``ModelBuilderCommand`` class from which other
    prover command decorators can extend.
    """

    def __init__(self, modelBuilderCommand):
        """
        :param modelBuilderCommand: ``ModelBuilderCommand`` to decorate
        """
        TheoremToolCommandDecorator.__init__(self, modelBuilderCommand)

        # The decorator has its own versions of 'result' and 'valuation'
        # because they may be different from the underlying command
        self._model = None

    def build_model(self, verbose=False):
        """
        Attempt to build a model.  Store the result to prevent unnecessary
        re-building.
        """
        if self._result is None:
            modelbuilder = self.get_model_builder()
            self._result, self._model = modelbuilder._build_model(
                self.goal(), self.assumptions(), verbose
            )
        return self._result

    def model(self, format=None):
        """
        Return a string representation of the model

        :param simplify: bool simplify the proof?
        :return: str
        """
        if self._result is None:
            raise LookupError("You have to call build_model() first to " "get a model!")
        else:
            return self._decorate_model(self._model, format)

    def _decorate_model(self, valuation_str, format=None):
        """
        Modify and return the proof string
        :param valuation_str: str with the model builder's output
        :param format: str indicating the format for displaying
        :return: str
        """
        return self._command._decorate_model(valuation_str, format)

    def get_model_builder(self):
        return self._command.get_prover()


class ParallelProverBuilder(Prover, ModelBuilder):
    """
    This class stores both a prover and a model builder and when either
    prove() or build_model() is called, then both theorem tools are run in
    parallel.  Whichever finishes first, the prover or the model builder, is the
    result that will be used.
    """

    def __init__(self, prover, modelbuilder):
        self._prover = prover
        self._modelbuilder = modelbuilder

    def _prove(self, goal=None, assumptions=None, verbose=False):
        return self._run(goal, assumptions, verbose), ""

    def _build_model(self, goal=None, assumptions=None, verbose=False):
        return not self._run(goal, assumptions, verbose), ""

    def _run(self, goal, assumptions, verbose):
        # Set up two thread, Prover and ModelBuilder to run in parallel
        tp_thread = TheoremToolThread(
            lambda: self._prover.prove(goal, assumptions, verbose), verbose, "TP"
        )
        mb_thread = TheoremToolThread(
            lambda: self._modelbuilder.build_model(goal, assumptions, verbose),
            verbose,
            "MB",
        )

        tp_thread.start()
        mb_thread.start()

        while tp_thread.is_alive() and mb_thread.is_alive():
            # wait until either the prover or the model builder is done
            pass

        if tp_thread.result is not None:
            return tp_thread.result
        elif mb_thread.result is not None:
            return not mb_thread.result
        else:
            return None


class ParallelProverBuilderCommand(BaseProverCommand, BaseModelBuilderCommand):
    """
    This command stores both a prover and a model builder and when either
    prove() or build_model() is called, then both theorem tools are run in
    parallel.  Whichever finishes first, the prover or the model builder, is the
    result that will be used.

    Because the theorem prover result is the opposite of the model builder
    result, we will treat self._result as meaning "proof found/no model found".
    """

    def __init__(self, prover, modelbuilder, goal=None, assumptions=None):
        BaseProverCommand.__init__(self, prover, goal, assumptions)
        BaseModelBuilderCommand.__init__(self, modelbuilder, goal, assumptions)

    def prove(self, verbose=False):
        return self._run(verbose)

    def build_model(self, verbose=False):
        return not self._run(verbose)

    def _run(self, verbose):
        # Set up two thread, Prover and ModelBuilder to run in parallel
        tp_thread = TheoremToolThread(
            lambda: BaseProverCommand.prove(self, verbose), verbose, "TP"
        )
        mb_thread = TheoremToolThread(
            lambda: BaseModelBuilderCommand.build_model(self, verbose), verbose, "MB"
        )

        tp_thread.start()
        mb_thread.start()

        while tp_thread.is_alive() and mb_thread.is_alive():
            # wait until either the prover or the model builder is done
            pass

        if tp_thread.result is not None:
            self._result = tp_thread.result
        elif mb_thread.result is not None:
            self._result = not mb_thread.result
        return self._result


class TheoremToolThread(threading.Thread):
    def __init__(self, command, verbose, name=None):
        threading.Thread.__init__(self)
        self._command = command
        self._result = None
        self._verbose = verbose
        self._name = name

    def run(self):
        try:
            self._result = self._command()
            if self._verbose:
                print(
                    "Thread %s finished with result %s at %s"
                    % (self._name, self._result, time.localtime(time.time()))
                )
        except Exception as e:
            print(e)
            print("Thread %s completed abnormally" % (self._name))

    @property
    def result(self):
        return self._result
