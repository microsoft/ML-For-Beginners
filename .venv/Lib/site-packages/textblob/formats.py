# -*- coding: utf-8 -*-
"""File formats for training and testing data.

Includes a registry of valid file formats. New file formats can be added to the
registry like so: ::

    from textblob import formats

    class PipeDelimitedFormat(formats.DelimitedFormat):
        delimiter = '|'

    formats.register('psv', PipeDelimitedFormat)

Once a format has been registered, classifiers will be able to read data files with
that format. ::

    from textblob.classifiers import NaiveBayesAnalyzer

    with open('training_data.psv', 'r') as fp:
        cl = NaiveBayesAnalyzer(fp, format='psv')
"""
from __future__ import absolute_import
import json
from collections import OrderedDict

from textblob.compat import PY2, csv
from textblob.utils import is_filelike

DEFAULT_ENCODING = 'utf-8'

class BaseFormat(object):
    """Interface for format classes. Individual formats can decide on the
    composition and meaning of ``**kwargs``.

    :param File fp: A file-like object.

    .. versionchanged:: 0.9.0
        Constructor receives a file pointer rather than a file path.
    """
    def __init__(self, fp, **kwargs):
        pass

    def to_iterable(self):
        """Return an iterable object from the data."""
        raise NotImplementedError('Must implement a "to_iterable" method.')

    @classmethod
    def detect(cls, stream):
        """Detect the file format given a filename.
        Return True if a stream is this file format.

        .. versionchanged:: 0.9.0
            Changed from a static method to a class method.
        """
        raise NotImplementedError('Must implement a "detect" class method.')

class DelimitedFormat(BaseFormat):
    """A general character-delimited format."""

    delimiter = ","

    def __init__(self, fp, **kwargs):
        BaseFormat.__init__(self, fp, **kwargs)
        if PY2:
            reader = csv.reader(fp, delimiter=self.delimiter,
                                encoding=DEFAULT_ENCODING)
        else:
            reader = csv.reader(fp, delimiter=self.delimiter)
        self.data = [row for row in reader]

    def to_iterable(self):
        """Return an iterable object from the data."""
        return self.data

    @classmethod
    def detect(cls, stream):
        """Return True if stream is valid."""
        try:
            csv.Sniffer().sniff(stream, delimiters=cls.delimiter)
            return True
        except (csv.Error, TypeError):
            return False


class CSV(DelimitedFormat):
    """CSV format. Assumes each row is of the form ``text,label``.
    ::

        Today is a good day,pos
        I hate this car.,pos
    """
    delimiter = ","


class TSV(DelimitedFormat):
    """TSV format. Assumes each row is of the form ``text\tlabel``.
    """
    delimiter = "\t"


class JSON(BaseFormat):
    """JSON format.

    Assumes that JSON is formatted as an array of objects with ``text`` and
    ``label`` properties.
    ::

        [
            {"text": "Today is a good day.", "label": "pos"},
            {"text": "I hate this car.", "label": "neg"}
        ]
    """
    def __init__(self, fp, **kwargs):
        BaseFormat.__init__(self, fp, **kwargs)
        self.dict = json.load(fp)

    def to_iterable(self):
        """Return an iterable object from the JSON data."""
        return [(d['text'], d['label']) for d in self.dict]

    @classmethod
    def detect(cls, stream):
        """Return True if stream is valid JSON."""
        try:
            json.loads(stream)
            return True
        except ValueError:
            return False


_registry = OrderedDict([
    ('csv', CSV),
    ('json', JSON),
    ('tsv', TSV),
])

def detect(fp, max_read=1024):
    """Attempt to detect a file's format, trying each of the supported
    formats. Return the format class that was detected. If no format is
    detected, return ``None``.
    """
    if not is_filelike(fp):
        return None
    for Format in _registry.values():
        if Format.detect(fp.read(max_read)):
            fp.seek(0)
            return Format
        fp.seek(0)
    return None

def get_registry():
    """Return a dictionary of registered formats."""
    return _registry

def register(name, format_class):
    """Register a new format.

    :param str name: The name that will be used to refer to the format, e.g. 'csv'
    :param type format_class: The format class to register.
    """
    get_registry()[name] = format_class
