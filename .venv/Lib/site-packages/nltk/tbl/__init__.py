# Natural Language Toolkit: Transformation-based learning
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Marcus Uneson <marcus.uneson@gmail.com>
#   based on previous (nltk2) version by
#   Christopher Maloof, Edward Loper, Steven Bird
# URL: <https://www.nltk.org/>
# For license information, see  LICENSE.TXT

"""
Transformation Based Learning

A general purpose package for Transformation Based Learning,
currently used by nltk.tag.BrillTagger.

isort:skip_file
"""

from nltk.tbl.template import Template

# API: Template(...), Template.expand(...)

from nltk.tbl.feature import Feature

# API: Feature(...), Feature.expand(...)

from nltk.tbl.rule import Rule

# API: Rule.format(...), Rule.templatetid

from nltk.tbl.erroranalysis import error_list
