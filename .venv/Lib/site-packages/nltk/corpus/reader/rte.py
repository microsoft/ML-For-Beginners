# Natural Language Toolkit: RTE Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author:  Ewan Klein <ewan@inf.ed.ac.uk>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Corpus reader for the Recognizing Textual Entailment (RTE) Challenge Corpora.

The files were taken from the RTE1, RTE2 and RTE3 datasets and the files
were regularized.

Filenames are of the form rte*_dev.xml and rte*_test.xml. The latter are the
gold standard annotated files.

Each entailment corpus is a list of 'text'/'hypothesis' pairs. The following
example is taken from RTE3::

 <pair id="1" entailment="YES" task="IE" length="short" >

    <t>The sale was made to pay Yukos' US$ 27.5 billion tax bill,
    Yuganskneftegaz was originally sold for US$ 9.4 billion to a little known
    company Baikalfinansgroup which was later bought by the Russian
    state-owned oil company Rosneft .</t>

   <h>Baikalfinansgroup was sold to Rosneft.</h>
 </pair>

In order to provide globally unique IDs for each pair, a new attribute
``challenge`` has been added to the root element ``entailment-corpus`` of each
file, taking values 1, 2 or 3. The GID is formatted 'm-n', where 'm' is the
challenge number and 'n' is the pair ID.
"""
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *


def norm(value_string):
    """
    Normalize the string value in an RTE pair's ``value`` or ``entailment``
    attribute as an integer (1, 0).

    :param value_string: the label used to classify a text/hypothesis pair
    :type value_string: str
    :rtype: int
    """

    valdict = {"TRUE": 1, "FALSE": 0, "YES": 1, "NO": 0}
    return valdict[value_string.upper()]


class RTEPair:
    """
    Container for RTE text-hypothesis pairs.

    The entailment relation is signalled by the ``value`` attribute in RTE1, and by
    ``entailment`` in RTE2 and RTE3. These both get mapped on to the ``entailment``
    attribute of this class.
    """

    def __init__(
        self,
        pair,
        challenge=None,
        id=None,
        text=None,
        hyp=None,
        value=None,
        task=None,
        length=None,
    ):
        """
        :param challenge: version of the RTE challenge (i.e., RTE1, RTE2 or RTE3)
        :param id: identifier for the pair
        :param text: the text component of the pair
        :param hyp: the hypothesis component of the pair
        :param value: classification label for the pair
        :param task: attribute for the particular NLP task that the data was drawn from
        :param length: attribute for the length of the text of the pair
        """
        self.challenge = challenge
        self.id = pair.attrib["id"]
        self.gid = f"{self.challenge}-{self.id}"
        self.text = pair[0].text
        self.hyp = pair[1].text

        if "value" in pair.attrib:
            self.value = norm(pair.attrib["value"])
        elif "entailment" in pair.attrib:
            self.value = norm(pair.attrib["entailment"])
        else:
            self.value = value
        if "task" in pair.attrib:
            self.task = pair.attrib["task"]
        else:
            self.task = task
        if "length" in pair.attrib:
            self.length = pair.attrib["length"]
        else:
            self.length = length

    def __repr__(self):
        if self.challenge:
            return f"<RTEPair: gid={self.challenge}-{self.id}>"
        else:
            return "<RTEPair: id=%s>" % self.id


class RTECorpusReader(XMLCorpusReader):
    """
    Corpus reader for corpora in RTE challenges.

    This is just a wrapper around the XMLCorpusReader. See module docstring above for the expected
    structure of input documents.
    """

    def _read_etree(self, doc):
        """
        Map the XML input into an RTEPair.

        This uses the ``getiterator()`` method from the ElementTree package to
        find all the ``<pair>`` elements.

        :param doc: a parsed XML document
        :rtype: list(RTEPair)
        """
        try:
            challenge = doc.attrib["challenge"]
        except KeyError:
            challenge = None
        pairiter = doc.iter("pair")
        return [RTEPair(pair, challenge=challenge) for pair in pairiter]

    def pairs(self, fileids):
        """
        Build a list of RTEPairs from a RTE corpus.

        :param fileids: a list of RTE corpus fileids
        :type: list
        :rtype: list(RTEPair)
        """
        if isinstance(fileids, str):
            fileids = [fileids]
        return concat([self._read_etree(self.xml(fileid)) for fileid in fileids])
