# -*- coding: utf-8 -*-
'''Default parsers to English for backwards compatibility so you can still do

>>> from textblob.parsers import PatternParser

which is equivalent to

>>> from textblob.en.parsers import PatternParser
'''
from __future__ import absolute_import
from textblob.base import BaseParser
from textblob.en.parsers import PatternParser

__all__ = [
    'BaseParser',
    'PatternParser',
]
