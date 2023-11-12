# Natural Language Toolkit: Combinatory Categorial Grammar
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Tanin Na Nakorn (@tanin)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
Helper functions for CCG semantics computation
"""

from nltk.sem.logic import *


def compute_type_raised_semantics(semantics):
    core = semantics
    parent = None
    while isinstance(core, LambdaExpression):
        parent = core
        core = core.term

    var = Variable("F")
    while var in core.free():
        var = unique_variable(pattern=var)
    core = ApplicationExpression(FunctionVariableExpression(var), core)

    if parent is not None:
        parent.term = core
    else:
        semantics = core

    return LambdaExpression(var, semantics)


def compute_function_semantics(function, argument):
    return ApplicationExpression(function, argument).simplify()


def compute_composition_semantics(function, argument):
    assert isinstance(argument, LambdaExpression), (
        "`" + str(argument) + "` must be a lambda expression"
    )
    return LambdaExpression(
        argument.variable, ApplicationExpression(function, argument.term).simplify()
    )


def compute_substitution_semantics(function, argument):
    assert isinstance(function, LambdaExpression) and isinstance(
        function.term, LambdaExpression
    ), ("`" + str(function) + "` must be a lambda expression with 2 arguments")
    assert isinstance(argument, LambdaExpression), (
        "`" + str(argument) + "` must be a lambda expression"
    )

    new_argument = ApplicationExpression(
        argument, VariableExpression(function.variable)
    ).simplify()
    new_term = ApplicationExpression(function.term, new_argument).simplify()

    return LambdaExpression(function.variable, new_term)
