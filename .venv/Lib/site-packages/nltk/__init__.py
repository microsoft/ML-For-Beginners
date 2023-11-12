# Natural Language Toolkit (NLTK)
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Steven Bird <stevenbird1@gmail.com>
#          Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
The Natural Language Toolkit (NLTK) is an open source Python library
for Natural Language Processing.  A free online book is available.
(If you use the library for academic research, please cite the book.)

Steven Bird, Ewan Klein, and Edward Loper (2009).
Natural Language Processing with Python.  O'Reilly Media Inc.
https://www.nltk.org/book/

isort:skip_file
"""

import os

# //////////////////////////////////////////////////////
# Metadata
# //////////////////////////////////////////////////////

# Version.  For each new release, the version number should be updated
# in the file VERSION.
try:
    # If a VERSION file exists, use it!
    version_file = os.path.join(os.path.dirname(__file__), "VERSION")
    with open(version_file) as infile:
        __version__ = infile.read().strip()
except NameError:
    __version__ = "unknown (running code interactively?)"
except OSError as ex:
    __version__ = "unknown (%s)" % ex

if __doc__ is not None:  # fix for the ``python -OO``
    __doc__ += "\n@version: " + __version__


# Copyright notice
__copyright__ = """\
Copyright (C) 2001-2023 NLTK Project.

Distributed and Licensed under the Apache License, Version 2.0,
which is included by reference.
"""

__license__ = "Apache License, Version 2.0"
# Description of the toolkit, keywords, and the project's primary URL.
__longdescr__ = """\
The Natural Language Toolkit (NLTK) is a Python package for
natural language processing.  NLTK requires Python 3.7, 3.8, 3.9, 3.10 or 3.11."""
__keywords__ = [
    "NLP",
    "CL",
    "natural language processing",
    "computational linguistics",
    "parsing",
    "tagging",
    "tokenizing",
    "syntax",
    "linguistics",
    "language",
    "natural language",
    "text analytics",
]
__url__ = "https://www.nltk.org/"

# Maintainer, contributors, etc.
__maintainer__ = "NLTK Team"
__maintainer_email__ = "nltk.team@gmail.com"
__author__ = __maintainer__
__author_email__ = __maintainer_email__

# "Trove" classifiers for Python Package Index.
__classifiers__ = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Text Processing",
    "Topic :: Text Processing :: Filters",
    "Topic :: Text Processing :: General",
    "Topic :: Text Processing :: Indexing",
    "Topic :: Text Processing :: Linguistic",
]

from nltk.internals import config_java

# support numpy from pypy
try:
    import numpypy
except ImportError:
    pass

# Override missing methods on environments where it cannot be used like GAE.
import subprocess

if not hasattr(subprocess, "PIPE"):

    def _fake_PIPE(*args, **kwargs):
        raise NotImplementedError("subprocess.PIPE is not supported.")

    subprocess.PIPE = _fake_PIPE
if not hasattr(subprocess, "Popen"):

    def _fake_Popen(*args, **kwargs):
        raise NotImplementedError("subprocess.Popen is not supported.")

    subprocess.Popen = _fake_Popen

###########################################################
# TOP-LEVEL MODULES
###########################################################

# Import top-level functionality into top-level namespace

from nltk.collocations import *
from nltk.decorators import decorator, memoize
from nltk.featstruct import *
from nltk.grammar import *
from nltk.probability import *
from nltk.text import *
from nltk.util import *
from nltk.jsontags import *

###########################################################
# PACKAGES
###########################################################

from nltk.chunk import *
from nltk.classify import *
from nltk.inference import *
from nltk.metrics import *
from nltk.parse import *
from nltk.tag import *
from nltk.tokenize import *
from nltk.translate import *
from nltk.tree import *
from nltk.sem import *
from nltk.stem import *

# Packages which can be lazily imported
# (a) we don't import *
# (b) they're slow to import or have run-time dependencies
#     that can safely fail at run time

from nltk import lazyimport

app = lazyimport.LazyModule("app", locals(), globals())
chat = lazyimport.LazyModule("chat", locals(), globals())
corpus = lazyimport.LazyModule("corpus", locals(), globals())
draw = lazyimport.LazyModule("draw", locals(), globals())
toolbox = lazyimport.LazyModule("toolbox", locals(), globals())

# Optional loading

try:
    import numpy
except ImportError:
    pass
else:
    from nltk import cluster

from nltk.downloader import download, download_shell

try:
    import tkinter
except ImportError:
    pass
else:
    try:
        from nltk.downloader import download_gui
    except RuntimeError as e:
        import warnings

        warnings.warn(
            "Corpus downloader GUI not loaded "
            "(RuntimeError during import: %s)" % str(e)
        )

# explicitly import all top-level modules (ensuring
# they override the same names inadvertently imported
# from a subpackage)

from nltk import ccg, chunk, classify, collocations
from nltk import data, featstruct, grammar, help, inference, metrics
from nltk import misc, parse, probability, sem, stem, wsd
from nltk import tag, tbl, text, tokenize, translate, tree, util


# FIXME:  override any accidentally imported demo, see https://github.com/nltk/nltk/issues/2116
def demo():
    print("To run the demo code for a module, type nltk.module.demo()")
