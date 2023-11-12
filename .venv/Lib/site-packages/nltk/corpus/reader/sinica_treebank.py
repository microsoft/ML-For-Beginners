# Natural Language Toolkit: Sinica Treebank Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Sinica Treebank Corpus Sample

http://rocling.iis.sinica.edu.tw/CKIP/engversion/treebank.htm

10,000 parsed sentences, drawn from the Academia Sinica Balanced
Corpus of Modern Chinese.  Parse tree notation is based on
Information-based Case Grammar.  Tagset documentation is available
at https://www.sinica.edu.tw/SinicaCorpus/modern_e_wordtype.html

Language and Knowledge Processing Group, Institute of Information
Science, Academia Sinica

The data is distributed with the Natural Language Toolkit under the terms of
the Creative Commons Attribution-NonCommercial-ShareAlike License
[https://creativecommons.org/licenses/by-nc-sa/2.5/].

References:

Feng-Yi Chen, Pi-Fang Tsai, Keh-Jiann Chen, and Chu-Ren Huang (1999)
The Construction of Sinica Treebank. Computational Linguistics and
Chinese Language Processing, 4, pp 87-104.

Huang Chu-Ren, Keh-Jiann Chen, Feng-Yi Chen, Keh-Jiann Chen, Zhao-Ming
Gao, and Kuang-Yu Chen. 2000. Sinica Treebank: Design Criteria,
Annotation Guidelines, and On-line Interface. Proceedings of 2nd
Chinese Language Processing Workshop, Association for Computational
Linguistics.

Chen Keh-Jiann and Yu-Ming Hsieh (2004) Chinese Treebanks and Grammar
Extraction, Proceedings of IJCNLP-04, pp560-565.
"""

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import sinica_parse

IDENTIFIER = re.compile(r"^#\S+\s")
APPENDIX = re.compile(r"(?<=\))#.*$")
TAGWORD = re.compile(r":([^:()|]+):([^:()|]+)")
WORD = re.compile(r":[^:()|]+:([^:()|]+)")


class SinicaTreebankCorpusReader(SyntaxCorpusReader):
    """
    Reader for the sinica treebank.
    """

    def _read_block(self, stream):
        sent = stream.readline()
        sent = IDENTIFIER.sub("", sent)
        sent = APPENDIX.sub("", sent)
        return [sent]

    def _parse(self, sent):
        return sinica_parse(sent)

    def _tag(self, sent, tagset=None):
        tagged_sent = [(w, t) for (t, w) in TAGWORD.findall(sent)]
        if tagset and tagset != self._tagset:
            tagged_sent = [
                (w, map_tag(self._tagset, tagset, t)) for (w, t) in tagged_sent
            ]
        return tagged_sent

    def _word(self, sent):
        return WORD.findall(sent)
