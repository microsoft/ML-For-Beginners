# Natural Language Toolkit: York-Toronto-Helsinki Parsed Corpus of Old English Prose (YCOE)
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Selina Dennis <selina@tranzfusion.net>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Corpus reader for the York-Toronto-Helsinki Parsed Corpus of Old
English Prose (YCOE), a 1.5 million word syntactically-annotated
corpus of Old English prose texts. The corpus is distributed by the
Oxford Text Archive: http://www.ota.ahds.ac.uk/ It is not included
with NLTK.

The YCOE corpus is divided into 100 files, each representing
an Old English prose text. Tags used within each text complies
to the YCOE standard: https://www-users.york.ac.uk/~lang22/YCOE/YcoeHome.htm
"""

import os
import re

from nltk.corpus.reader.api import *
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.corpus.reader.util import *
from nltk.tokenize import RegexpTokenizer


class YCOECorpusReader(CorpusReader):
    """
    Corpus reader for the York-Toronto-Helsinki Parsed Corpus of Old
    English Prose (YCOE), a 1.5 million word syntactically-annotated
    corpus of Old English prose texts.
    """

    def __init__(self, root, encoding="utf8"):
        CorpusReader.__init__(self, root, [], encoding)

        self._psd_reader = YCOEParseCorpusReader(
            self.root.join("psd"), ".*", ".psd", encoding=encoding
        )
        self._pos_reader = YCOETaggedCorpusReader(self.root.join("pos"), ".*", ".pos")

        # Make sure we have a consistent set of items:
        documents = {f[:-4] for f in self._psd_reader.fileids()}
        if {f[:-4] for f in self._pos_reader.fileids()} != documents:
            raise ValueError('Items in "psd" and "pos" ' "subdirectories do not match.")

        fileids = sorted(
            ["%s.psd" % doc for doc in documents]
            + ["%s.pos" % doc for doc in documents]
        )
        CorpusReader.__init__(self, root, fileids, encoding)
        self._documents = sorted(documents)

    def documents(self, fileids=None):
        """
        Return a list of document identifiers for all documents in
        this corpus, or for the documents with the given file(s) if
        specified.
        """
        if fileids is None:
            return self._documents
        if isinstance(fileids, str):
            fileids = [fileids]
        for f in fileids:
            if f not in self._fileids:
                raise KeyError("File id %s not found" % fileids)
        # Strip off the '.pos' and '.psd' extensions.
        return sorted({f[:-4] for f in fileids})

    def fileids(self, documents=None):
        """
        Return a list of file identifiers for the files that make up
        this corpus, or that store the given document(s) if specified.
        """
        if documents is None:
            return self._fileids
        elif isinstance(documents, str):
            documents = [documents]
        return sorted(
            set(
                ["%s.pos" % doc for doc in documents]
                + ["%s.psd" % doc for doc in documents]
            )
        )

    def _getfileids(self, documents, subcorpus):
        """
        Helper that selects the appropriate fileids for a given set of
        documents from a given subcorpus (pos or psd).
        """
        if documents is None:
            documents = self._documents
        else:
            if isinstance(documents, str):
                documents = [documents]
            for document in documents:
                if document not in self._documents:
                    if document[-4:] in (".pos", ".psd"):
                        raise ValueError(
                            "Expected a document identifier, not a file "
                            "identifier.  (Use corpus.documents() to get "
                            "a list of document identifiers."
                        )
                    else:
                        raise ValueError("Document identifier %s not found" % document)
        return [f"{d}.{subcorpus}" for d in documents]

    # Delegate to one of our two sub-readers:
    def words(self, documents=None):
        return self._pos_reader.words(self._getfileids(documents, "pos"))

    def sents(self, documents=None):
        return self._pos_reader.sents(self._getfileids(documents, "pos"))

    def paras(self, documents=None):
        return self._pos_reader.paras(self._getfileids(documents, "pos"))

    def tagged_words(self, documents=None):
        return self._pos_reader.tagged_words(self._getfileids(documents, "pos"))

    def tagged_sents(self, documents=None):
        return self._pos_reader.tagged_sents(self._getfileids(documents, "pos"))

    def tagged_paras(self, documents=None):
        return self._pos_reader.tagged_paras(self._getfileids(documents, "pos"))

    def parsed_sents(self, documents=None):
        return self._psd_reader.parsed_sents(self._getfileids(documents, "psd"))


class YCOEParseCorpusReader(BracketParseCorpusReader):
    """Specialized version of the standard bracket parse corpus reader
    that strips out (CODE ...) and (ID ...) nodes."""

    def _parse(self, t):
        t = re.sub(r"(?u)\((CODE|ID)[^\)]*\)", "", t)
        if re.match(r"\s*\(\s*\)\s*$", t):
            return None
        return BracketParseCorpusReader._parse(self, t)


class YCOETaggedCorpusReader(TaggedCorpusReader):
    def __init__(self, root, items, encoding="utf8"):
        gaps_re = r"(?u)(?<=/\.)\s+|\s*\S*_CODE\s*|\s*\S*_ID\s*"
        sent_tokenizer = RegexpTokenizer(gaps_re, gaps=True)
        TaggedCorpusReader.__init__(
            self, root, items, sep="_", sent_tokenizer=sent_tokenizer
        )


#: A list of all documents and their titles in ycoe.
documents = {
    "coadrian.o34": "Adrian and Ritheus",
    "coaelhom.o3": "Ælfric, Supplemental Homilies",
    "coaelive.o3": "Ælfric's Lives of Saints",
    "coalcuin": "Alcuin De virtutibus et vitiis",
    "coalex.o23": "Alexander's Letter to Aristotle",
    "coapollo.o3": "Apollonius of Tyre",
    "coaugust": "Augustine",
    "cobede.o2": "Bede's History of the English Church",
    "cobenrul.o3": "Benedictine Rule",
    "coblick.o23": "Blickling Homilies",
    "coboeth.o2": "Boethius' Consolation of Philosophy",
    "cobyrhtf.o3": "Byrhtferth's Manual",
    "cocanedgD": "Canons of Edgar (D)",
    "cocanedgX": "Canons of Edgar (X)",
    "cocathom1.o3": "Ælfric's Catholic Homilies I",
    "cocathom2.o3": "Ælfric's Catholic Homilies II",
    "cochad.o24": "Saint Chad",
    "cochdrul": "Chrodegang of Metz, Rule",
    "cochristoph": "Saint Christopher",
    "cochronA.o23": "Anglo-Saxon Chronicle A",
    "cochronC": "Anglo-Saxon Chronicle C",
    "cochronD": "Anglo-Saxon Chronicle D",
    "cochronE.o34": "Anglo-Saxon Chronicle E",
    "cocura.o2": "Cura Pastoralis",
    "cocuraC": "Cura Pastoralis (Cotton)",
    "codicts.o34": "Dicts of Cato",
    "codocu1.o1": "Documents 1 (O1)",
    "codocu2.o12": "Documents 2 (O1/O2)",
    "codocu2.o2": "Documents 2 (O2)",
    "codocu3.o23": "Documents 3 (O2/O3)",
    "codocu3.o3": "Documents 3 (O3)",
    "codocu4.o24": "Documents 4 (O2/O4)",
    "coeluc1": "Honorius of Autun, Elucidarium 1",
    "coeluc2": "Honorius of Autun, Elucidarium 1",
    "coepigen.o3": "Ælfric's Epilogue to Genesis",
    "coeuphr": "Saint Euphrosyne",
    "coeust": "Saint Eustace and his companions",
    "coexodusP": "Exodus (P)",
    "cogenesiC": "Genesis (C)",
    "cogregdC.o24": "Gregory's Dialogues (C)",
    "cogregdH.o23": "Gregory's Dialogues (H)",
    "coherbar": "Pseudo-Apuleius, Herbarium",
    "coinspolD.o34": "Wulfstan's Institute of Polity (D)",
    "coinspolX": "Wulfstan's Institute of Polity (X)",
    "cojames": "Saint James",
    "colacnu.o23": "Lacnunga",
    "colaece.o2": "Leechdoms",
    "colaw1cn.o3": "Laws, Cnut I",
    "colaw2cn.o3": "Laws, Cnut II",
    "colaw5atr.o3": "Laws, Æthelred V",
    "colaw6atr.o3": "Laws, Æthelred VI",
    "colawaf.o2": "Laws, Alfred",
    "colawafint.o2": "Alfred's Introduction to Laws",
    "colawger.o34": "Laws, Gerefa",
    "colawine.ox2": "Laws, Ine",
    "colawnorthu.o3": "Northumbra Preosta Lagu",
    "colawwllad.o4": "Laws, William I, Lad",
    "coleofri.o4": "Leofric",
    "colsigef.o3": "Ælfric's Letter to Sigefyrth",
    "colsigewB": "Ælfric's Letter to Sigeweard (B)",
    "colsigewZ.o34": "Ælfric's Letter to Sigeweard (Z)",
    "colwgeat": "Ælfric's Letter to Wulfgeat",
    "colwsigeT": "Ælfric's Letter to Wulfsige (T)",
    "colwsigeXa.o34": "Ælfric's Letter to Wulfsige (Xa)",
    "colwstan1.o3": "Ælfric's Letter to Wulfstan I",
    "colwstan2.o3": "Ælfric's Letter to Wulfstan II",
    "comargaC.o34": "Saint Margaret (C)",
    "comargaT": "Saint Margaret (T)",
    "comart1": "Martyrology, I",
    "comart2": "Martyrology, II",
    "comart3.o23": "Martyrology, III",
    "comarvel.o23": "Marvels of the East",
    "comary": "Mary of Egypt",
    "coneot": "Saint Neot",
    "conicodA": "Gospel of Nicodemus (A)",
    "conicodC": "Gospel of Nicodemus (C)",
    "conicodD": "Gospel of Nicodemus (D)",
    "conicodE": "Gospel of Nicodemus (E)",
    "coorosiu.o2": "Orosius",
    "cootest.o3": "Heptateuch",
    "coprefcath1.o3": "Ælfric's Preface to Catholic Homilies I",
    "coprefcath2.o3": "Ælfric's Preface to Catholic Homilies II",
    "coprefcura.o2": "Preface to the Cura Pastoralis",
    "coprefgen.o3": "Ælfric's Preface to Genesis",
    "copreflives.o3": "Ælfric's Preface to Lives of Saints",
    "coprefsolilo": "Preface to Augustine's Soliloquies",
    "coquadru.o23": "Pseudo-Apuleius, Medicina de quadrupedibus",
    "corood": "History of the Holy Rood-Tree",
    "cosevensl": "Seven Sleepers",
    "cosolilo": "St. Augustine's Soliloquies",
    "cosolsat1.o4": "Solomon and Saturn I",
    "cosolsat2": "Solomon and Saturn II",
    "cotempo.o3": "Ælfric's De Temporibus Anni",
    "coverhom": "Vercelli Homilies",
    "coverhomE": "Vercelli Homilies (E)",
    "coverhomL": "Vercelli Homilies (L)",
    "covinceB": "Saint Vincent (Bodley 343)",
    "covinsal": "Vindicta Salvatoris",
    "cowsgosp.o3": "West-Saxon Gospels",
    "cowulf.o34": "Wulfstan's Homilies",
}
