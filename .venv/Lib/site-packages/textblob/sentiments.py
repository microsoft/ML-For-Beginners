# -*- coding: utf-8 -*-
'''Default sentiment analyzers are English for backwards compatibility, so
you can still do

>>> from textblob.sentiments import PatternAnalyzer

which is equivalent to

>>> from textblob.en.sentiments import PatternAnalyzer
'''
from __future__ import absolute_import
from textblob.base import BaseSentimentAnalyzer
from textblob.en.sentiments import (DISCRETE, CONTINUOUS,
                                PatternAnalyzer, NaiveBayesAnalyzer)

__all__ = [
    'BaseSentimentAnalyzer',
    'DISCRETE',
    'CONTINUOUS',
    'PatternAnalyzer',
    'NaiveBayesAnalyzer',
]
