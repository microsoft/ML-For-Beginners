# Natural Language Toolkit: SemCor Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Nathan Schneider <nschneid@cs.cmu.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Corpus reader for the SemCor Corpus.
"""

__docformat__ = "epytext en"

from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
from nltk.tree import Tree


class SemcorCorpusReader(XMLCorpusReader):
    """
    Corpus reader for the SemCor Corpus.
    For access to the complete XML data structure, use the ``xml()``
    method.  For access to simple word lists and tagged word lists, use
    ``words()``, ``sents()``, ``tagged_words()``, and ``tagged_sents()``.
    """

    def __init__(self, root, fileids, wordnet, lazy=True):
        XMLCorpusReader.__init__(self, root, fileids)
        self._lazy = lazy
        self._wordnet = wordnet

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        """
        return self._items(fileids, "word", False, False, False)

    def chunks(self, fileids=None):
        """
        :return: the given file(s) as a list of chunks,
            each of which is a list of words and punctuation symbols
            that form a unit.
        :rtype: list(list(str))
        """
        return self._items(fileids, "chunk", False, False, False)

    def tagged_chunks(self, fileids=None, tag=("pos" or "sem" or "both")):
        """
        :return: the given file(s) as a list of tagged chunks, represented
            in tree form.
        :rtype: list(Tree)

        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'`
            to indicate the kind of tags to include.  Semantic tags consist of
            WordNet lemma IDs, plus an `'NE'` node if the chunk is a named entity
            without a specific entry in WordNet.  (Named entities of type 'other'
            have no lemma.  Other chunks not in WordNet have no semantic tag.
            Punctuation tokens have `None` for their part of speech tag.)
        """
        return self._items(fileids, "chunk", False, tag != "sem", tag != "pos")

    def sents(self, fileids=None):
        """
        :return: the given file(s) as a list of sentences, each encoded
            as a list of word strings.
        :rtype: list(list(str))
        """
        return self._items(fileids, "word", True, False, False)

    def chunk_sents(self, fileids=None):
        """
        :return: the given file(s) as a list of sentences, each encoded
            as a list of chunks.
        :rtype: list(list(list(str)))
        """
        return self._items(fileids, "chunk", True, False, False)

    def tagged_sents(self, fileids=None, tag=("pos" or "sem" or "both")):
        """
        :return: the given file(s) as a list of sentences. Each sentence
            is represented as a list of tagged chunks (in tree form).
        :rtype: list(list(Tree))

        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'`
            to indicate the kind of tags to include.  Semantic tags consist of
            WordNet lemma IDs, plus an `'NE'` node if the chunk is a named entity
            without a specific entry in WordNet.  (Named entities of type 'other'
            have no lemma.  Other chunks not in WordNet have no semantic tag.
            Punctuation tokens have `None` for their part of speech tag.)
        """
        return self._items(fileids, "chunk", True, tag != "sem", tag != "pos")

    def _items(self, fileids, unit, bracket_sent, pos_tag, sem_tag):
        if unit == "word" and not bracket_sent:
            # the result of the SemcorWordView may be a multiword unit, so the
            # LazyConcatenation will make sure the sentence is flattened
            _ = lambda *args: LazyConcatenation(
                (SemcorWordView if self._lazy else self._words)(*args)
            )
        else:
            _ = SemcorWordView if self._lazy else self._words
        return concat(
            [
                _(fileid, unit, bracket_sent, pos_tag, sem_tag, self._wordnet)
                for fileid in self.abspaths(fileids)
            ]
        )

    def _words(self, fileid, unit, bracket_sent, pos_tag, sem_tag):
        """
        Helper used to implement the view methods -- returns a list of
        tokens, (segmented) words, chunks, or sentences. The tokens
        and chunks may optionally be tagged (with POS and sense
        information).

        :param fileid: The name of the underlying file.
        :param unit: One of `'token'`, `'word'`, or `'chunk'`.
        :param bracket_sent: If true, include sentence bracketing.
        :param pos_tag: Whether to include part-of-speech tags.
        :param sem_tag: Whether to include semantic tags, namely WordNet lemma
            and OOV named entity status.
        """
        assert unit in ("token", "word", "chunk")
        result = []

        xmldoc = ElementTree.parse(fileid).getroot()
        for xmlsent in xmldoc.findall(".//s"):
            sent = []
            for xmlword in _all_xmlwords_in(xmlsent):
                itm = SemcorCorpusReader._word(
                    xmlword, unit, pos_tag, sem_tag, self._wordnet
                )
                if unit == "word":
                    sent.extend(itm)
                else:
                    sent.append(itm)

            if bracket_sent:
                result.append(SemcorSentence(xmlsent.attrib["snum"], sent))
            else:
                result.extend(sent)

        assert None not in result
        return result

    @staticmethod
    def _word(xmlword, unit, pos_tag, sem_tag, wordnet):
        tkn = xmlword.text
        if not tkn:
            tkn = ""  # fixes issue 337?

        lemma = xmlword.get("lemma", tkn)  # lemma or NE class
        lexsn = xmlword.get("lexsn")  # lex_sense (locator for the lemma's sense)
        if lexsn is not None:
            sense_key = lemma + "%" + lexsn
            wnpos = ("n", "v", "a", "r", "s")[
                int(lexsn.split(":")[0]) - 1
            ]  # see http://wordnet.princeton.edu/man/senseidx.5WN.html
        else:
            sense_key = wnpos = None
        redef = xmlword.get(
            "rdf", tkn
        )  # redefinition--this indicates the lookup string
        # does not exactly match the enclosed string, e.g. due to typographical adjustments
        # or discontinuity of a multiword expression. If a redefinition has occurred,
        # the "rdf" attribute holds its inflected form and "lemma" holds its lemma.
        # For NEs, "rdf", "lemma", and "pn" all hold the same value (the NE class).
        sensenum = xmlword.get("wnsn")  # WordNet sense number
        isOOVEntity = "pn" in xmlword.keys()  # a "personal name" (NE) not in WordNet
        pos = xmlword.get(
            "pos"
        )  # part of speech for the whole chunk (None for punctuation)

        if unit == "token":
            if not pos_tag and not sem_tag:
                itm = tkn
            else:
                itm = (
                    (tkn,)
                    + ((pos,) if pos_tag else ())
                    + ((lemma, wnpos, sensenum, isOOVEntity) if sem_tag else ())
                )
            return itm
        else:
            ww = tkn.split("_")  # TODO: case where punctuation intervenes in MWE
            if unit == "word":
                return ww
            else:
                if sensenum is not None:
                    try:
                        sense = wordnet.lemma_from_key(sense_key)  # Lemma object
                    except Exception:
                        # cannot retrieve the wordnet.Lemma object. possible reasons:
                        #  (a) the wordnet corpus is not downloaded;
                        #  (b) a nonexistent sense is annotated: e.g., such.s.00 triggers:
                        #  nltk.corpus.reader.wordnet.WordNetError: No synset found for key u'such%5:00:01:specified:00'
                        # solution: just use the lemma name as a string
                        try:
                            sense = "%s.%s.%02d" % (
                                lemma,
                                wnpos,
                                int(sensenum),
                            )  # e.g.: reach.v.02
                        except ValueError:
                            sense = (
                                lemma + "." + wnpos + "." + sensenum
                            )  # e.g. the sense number may be "2;1"

                bottom = [Tree(pos, ww)] if pos_tag else ww

                if sem_tag and isOOVEntity:
                    if sensenum is not None:
                        return Tree(sense, [Tree("NE", bottom)])
                    else:  # 'other' NE
                        return Tree("NE", bottom)
                elif sem_tag and sensenum is not None:
                    return Tree(sense, bottom)
                elif pos_tag:
                    return bottom[0]
                else:
                    return bottom  # chunk as a list


def _all_xmlwords_in(elt, result=None):
    if result is None:
        result = []
    for child in elt:
        if child.tag in ("wf", "punc"):
            result.append(child)
        else:
            _all_xmlwords_in(child, result)
    return result


class SemcorSentence(list):
    """
    A list of words, augmented by an attribute ``num`` used to record
    the sentence identifier (the ``n`` attribute from the XML).
    """

    def __init__(self, num, items):
        self.num = num
        list.__init__(self, items)


class SemcorWordView(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with the BNC corpus.
    """

    def __init__(self, fileid, unit, bracket_sent, pos_tag, sem_tag, wordnet):
        """
        :param fileid: The name of the underlying file.
        :param unit: One of `'token'`, `'word'`, or `'chunk'`.
        :param bracket_sent: If true, include sentence bracketing.
        :param pos_tag: Whether to include part-of-speech tags.
        :param sem_tag: Whether to include semantic tags, namely WordNet lemma
            and OOV named entity status.
        """
        if bracket_sent:
            tagspec = ".*/s"
        else:
            tagspec = ".*/s/(punc|wf)"

        self._unit = unit
        self._sent = bracket_sent
        self._pos_tag = pos_tag
        self._sem_tag = sem_tag
        self._wordnet = wordnet

        XMLCorpusView.__init__(self, fileid, tagspec)

    def handle_elt(self, elt, context):
        if self._sent:
            return self.handle_sent(elt)
        else:
            return self.handle_word(elt)

    def handle_word(self, elt):
        return SemcorCorpusReader._word(
            elt, self._unit, self._pos_tag, self._sem_tag, self._wordnet
        )

    def handle_sent(self, elt):
        sent = []
        for child in elt:
            if child.tag in ("wf", "punc"):
                itm = self.handle_word(child)
                if self._unit == "word":
                    sent.extend(itm)
                else:
                    sent.append(itm)
            else:
                raise ValueError("Unexpected element %s" % child.tag)
        return SemcorSentence(elt.attrib["snum"], sent)
