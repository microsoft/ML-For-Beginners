# -*- coding: utf-8 -*-
"""Default noun phrase extractors are for English to maintain backwards
compatibility, so you can still do

>>> from textblob.np_extractors import ConllExtractor

which is equivalent to

>>> from textblob.en.np_extractors import ConllExtractor
"""
from __future__ import absolute_import
from textblob.base import BaseNPExtractor
from textblob.en.np_extractors import ConllExtractor, FastNPExtractor

__all__ = [
    'BaseNPExtractor',
    'ConllExtractor',
    'FastNPExtractor',
]
