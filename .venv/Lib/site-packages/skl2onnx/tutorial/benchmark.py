# SPDX-License-Identifier: Apache-2.0

"""
Tools to help benchmarking.
"""
from timeit import Timer
import numpy


def measure_time(stmt, context, repeat=10, number=50, div_by_number=False):
    """
    Measures a statement and returns the results as a dictionary.

    :param stmt: string
    :param context: variable to know in a dictionary
    :param repeat: average over *repeat* experiment
    :param number: number of executions in one row
    :param div_by_number: divide by the number of executions
    :return: dictionary

    .. runpython::
        :showcode:

        from skl2onnx.tutorial import measure_time
        from math import cos

        res = measure_time("cos(x)", context=dict(cos=cos, x=5.))
        print(res)

    See `Timer.repeat <https://docs.python.org/3/library/
    timeit.html?timeit.Timer.repeat>`_
    for a better understanding of parameter *repeat* and *number*.
    The function returns a duration corresponding to
    *number* times the execution of the main statement.
    """
    tim = Timer(stmt, globals=context)
    res = numpy.array(tim.repeat(repeat=repeat, number=number))
    if div_by_number:
        res /= number
    mean = numpy.mean(res)
    dev = numpy.mean(res**2)
    dev = (dev - mean**2) ** 0.5
    mes = dict(
        average=mean,
        deviation=dev,
        min_exec=numpy.min(res),
        max_exec=numpy.max(res),
        repeat=repeat,
        number=number,
    )
    return mes
