'''Default taggers to the English taggers for backwards incompatibility, so you
can still do

>>> from textblob.taggers import NLTKTagger

which is equivalent to

>>> from textblob.en.taggers import NLTKTagger
'''
from __future__ import absolute_import
from textblob.base import BaseTagger
from textblob.en.taggers import PatternTagger, NLTKTagger

__all__ = [
    'BaseTagger',
    'PatternTagger',
    'NLTKTagger',
]
