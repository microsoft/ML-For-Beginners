# Natural Language Toolkit: Corpus Reader Utility Functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

######################################################################
# { Lazy Corpus Loader
######################################################################

import gc
import re

import nltk

TRY_ZIPFILE_FIRST = False


class LazyCorpusLoader:
    """
    To see the API documentation for this lazily loaded corpus, first
    run corpus.ensure_loaded(), and then run help(this_corpus).

    LazyCorpusLoader is a proxy object which is used to stand in for a
    corpus object before the corpus is loaded.  This allows NLTK to
    create an object for each corpus, but defer the costs associated
    with loading those corpora until the first time that they're
    actually accessed.

    The first time this object is accessed in any way, it will load
    the corresponding corpus, and transform itself into that corpus
    (by modifying its own ``__class__`` and ``__dict__`` attributes).

    If the corpus can not be found, then accessing this object will
    raise an exception, displaying installation instructions for the
    NLTK data package.  Once they've properly installed the data
    package (or modified ``nltk.data.path`` to point to its location),
    they can then use the corpus object without restarting python.

    :param name: The name of the corpus
    :type name: str
    :param reader_cls: The specific CorpusReader class, e.g. PlaintextCorpusReader, WordListCorpusReader
    :type reader: nltk.corpus.reader.api.CorpusReader
    :param nltk_data_subdir: The subdirectory where the corpus is stored.
    :type nltk_data_subdir: str
    :param `*args`: Any other non-keywords arguments that `reader_cls` might need.
    :param `**kwargs`: Any other keywords arguments that `reader_cls` might need.
    """

    def __init__(self, name, reader_cls, *args, **kwargs):
        from nltk.corpus.reader.api import CorpusReader

        assert issubclass(reader_cls, CorpusReader)
        self.__name = self.__name__ = name
        self.__reader_cls = reader_cls
        # If nltk_data_subdir is set explicitly
        if "nltk_data_subdir" in kwargs:
            # Use the specified subdirectory path
            self.subdir = kwargs["nltk_data_subdir"]
            # Pops the `nltk_data_subdir` argument, we don't need it anymore.
            kwargs.pop("nltk_data_subdir", None)
        else:  # Otherwise use 'nltk_data/corpora'
            self.subdir = "corpora"
        self.__args = args
        self.__kwargs = kwargs

    def __load(self):
        # Find the corpus root directory.
        zip_name = re.sub(r"(([^/]+)(/.*)?)", r"\2.zip/\1/", self.__name)
        if TRY_ZIPFILE_FIRST:
            try:
                root = nltk.data.find(f"{self.subdir}/{zip_name}")
            except LookupError as e:
                try:
                    root = nltk.data.find(f"{self.subdir}/{self.__name}")
                except LookupError:
                    raise e
        else:
            try:
                root = nltk.data.find(f"{self.subdir}/{self.__name}")
            except LookupError as e:
                try:
                    root = nltk.data.find(f"{self.subdir}/{zip_name}")
                except LookupError:
                    raise e

        # Load the corpus.
        corpus = self.__reader_cls(root, *self.__args, **self.__kwargs)

        # This is where the magic happens!  Transform ourselves into
        # the corpus by modifying our own __dict__ and __class__ to
        # match that of the corpus.

        args, kwargs = self.__args, self.__kwargs
        name, reader_cls = self.__name, self.__reader_cls

        self.__dict__ = corpus.__dict__
        self.__class__ = corpus.__class__

        # _unload support: assign __dict__ and __class__ back, then do GC.
        # after reassigning __dict__ there shouldn't be any references to
        # corpus data so the memory should be deallocated after gc.collect()
        def _unload(self):
            lazy_reader = LazyCorpusLoader(name, reader_cls, *args, **kwargs)
            self.__dict__ = lazy_reader.__dict__
            self.__class__ = lazy_reader.__class__
            gc.collect()

        self._unload = _make_bound_method(_unload, self)

    def __getattr__(self, attr):

        # Fix for inspect.isclass under Python 2.6
        # (see https://bugs.python.org/issue1225107).
        # Without this fix tests may take extra 1.5GB RAM
        # because all corpora gets loaded during test collection.
        if attr == "__bases__":
            raise AttributeError("LazyCorpusLoader object has no attribute '__bases__'")

        self.__load()
        # This looks circular, but its not, since __load() changes our
        # __class__ to something new:
        return getattr(self, attr)

    def __repr__(self):
        return "<{} in {!r} (not loaded yet)>".format(
            self.__reader_cls.__name__,
            ".../corpora/" + self.__name,
        )

    def _unload(self):
        # If an exception occurs during corpus loading then
        # '_unload' method may be unattached, so __getattr__ can be called;
        # we shouldn't trigger corpus loading again in this case.
        pass


def _make_bound_method(func, self):
    """
    Magic for creating bound methods (used for _unload).
    """

    class Foo:
        def meth(self):
            pass

    f = Foo()
    bound_method = type(f.meth)

    try:
        return bound_method(func, self, self.__class__)
    except TypeError:  # python3
        return bound_method(func, self)
