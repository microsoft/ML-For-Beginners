# Natural Language Toolkit: SVM-based classifier
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Leon Derczynski <leon@dcs.shef.ac.uk>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
nltk.classify.svm was deprecated. For classification based
on support vector machines SVMs use nltk.classify.scikitlearn
(or `scikit-learn <https://scikit-learn.org>`_ directly).
"""


class SvmClassifier:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(__doc__)
