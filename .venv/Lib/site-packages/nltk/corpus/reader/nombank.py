# Natural Language Toolkit: NomBank Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Paul Bedaride <paul.bedaride@gmail.com>
#          Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from functools import total_ordering
from xml.etree import ElementTree

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree


class NombankCorpusReader(CorpusReader):
    """
    Corpus reader for the nombank corpus, which augments the Penn
    Treebank with information about the predicate argument structure
    of every noun instance.  The corpus consists of two parts: the
    predicate-argument annotations themselves, and a set of "frameset
    files" which define the argument labels used by the annotations,
    on a per-noun basis.  Each "frameset file" contains one or more
    predicates, such as ``'turn'`` or ``'turn_on'``, each of which is
    divided into coarse-grained word senses called "rolesets".  For
    each "roleset", the frameset file provides descriptions of the
    argument roles, along with examples.
    """

    def __init__(
        self,
        root,
        nomfile,
        framefiles="",
        nounsfile=None,
        parse_fileid_xform=None,
        parse_corpus=None,
        encoding="utf8",
    ):
        """
        :param root: The root directory for this corpus.
        :param nomfile: The name of the file containing the predicate-
            argument annotations (relative to ``root``).
        :param framefiles: A list or regexp specifying the frameset
            fileids for this corpus.
        :param parse_fileid_xform: A transform that should be applied
            to the fileids in this corpus.  This should be a function
            of one argument (a fileid) that returns a string (the new
            fileid).
        :param parse_corpus: The corpus containing the parse trees
            corresponding to this corpus.  These parse trees are
            necessary to resolve the tree pointers used by nombank.
        """

        # If framefiles is specified as a regexp, expand it.
        if isinstance(framefiles, str):
            self._fileids = find_corpus_fileids(root, framefiles)
        self._fileids = list(framefiles)
        # Initialize the corpus reader.
        CorpusReader.__init__(self, root, framefiles, encoding)

        # Record our nom file & nouns file.
        self._nomfile = nomfile
        self._nounsfile = nounsfile
        self._parse_fileid_xform = parse_fileid_xform
        self._parse_corpus = parse_corpus

    def instances(self, baseform=None):
        """
        :return: a corpus view that acts as a list of
            ``NombankInstance`` objects, one for each noun in the corpus.
        """
        kwargs = {}
        if baseform is not None:
            kwargs["instance_filter"] = lambda inst: inst.baseform == baseform
        return StreamBackedCorpusView(
            self.abspath(self._nomfile),
            lambda stream: self._read_instance_block(stream, **kwargs),
            encoding=self.encoding(self._nomfile),
        )

    def lines(self):
        """
        :return: a corpus view that acts as a list of strings, one for
            each line in the predicate-argument annotation file.
        """
        return StreamBackedCorpusView(
            self.abspath(self._nomfile),
            read_line_block,
            encoding=self.encoding(self._nomfile),
        )

    def roleset(self, roleset_id):
        """
        :return: the xml description for the given roleset.
        """
        baseform = roleset_id.split(".")[0]
        baseform = baseform.replace("perc-sign", "%")
        baseform = baseform.replace("oneslashonezero", "1/10").replace(
            "1/10", "1-slash-10"
        )
        framefile = "frames/%s.xml" % baseform
        if framefile not in self.fileids():
            raise ValueError("Frameset file for %s not found" % roleset_id)

        # n.b.: The encoding for XML fileids is specified by the file
        # itself; so we ignore self._encoding here.
        with self.abspath(framefile).open() as fp:
            etree = ElementTree.parse(fp).getroot()
        for roleset in etree.findall("predicate/roleset"):
            if roleset.attrib["id"] == roleset_id:
                return roleset
        raise ValueError(f"Roleset {roleset_id} not found in {framefile}")

    def rolesets(self, baseform=None):
        """
        :return: list of xml descriptions for rolesets.
        """
        if baseform is not None:
            framefile = "frames/%s.xml" % baseform
            if framefile not in self.fileids():
                raise ValueError("Frameset file for %s not found" % baseform)
            framefiles = [framefile]
        else:
            framefiles = self.fileids()

        rsets = []
        for framefile in framefiles:
            # n.b.: The encoding for XML fileids is specified by the file
            # itself; so we ignore self._encoding here.
            with self.abspath(framefile).open() as fp:
                etree = ElementTree.parse(fp).getroot()
            rsets.append(etree.findall("predicate/roleset"))
        return LazyConcatenation(rsets)

    def nouns(self):
        """
        :return: a corpus view that acts as a list of all noun lemmas
            in this corpus (from the nombank.1.0.words file).
        """
        return StreamBackedCorpusView(
            self.abspath(self._nounsfile),
            read_line_block,
            encoding=self.encoding(self._nounsfile),
        )

    def _read_instance_block(self, stream, instance_filter=lambda inst: True):
        block = []

        # Read 100 at a time.
        for i in range(100):
            line = stream.readline().strip()
            if line:
                inst = NombankInstance.parse(
                    line, self._parse_fileid_xform, self._parse_corpus
                )
                if instance_filter(inst):
                    block.append(inst)

        return block


######################################################################
# { Nombank Instance & related datatypes
######################################################################


class NombankInstance:
    def __init__(
        self,
        fileid,
        sentnum,
        wordnum,
        baseform,
        sensenumber,
        predicate,
        predid,
        arguments,
        parse_corpus=None,
    ):

        self.fileid = fileid
        """The name of the file containing the parse tree for this
        instance's sentence."""

        self.sentnum = sentnum
        """The sentence number of this sentence within ``fileid``.
        Indexing starts from zero."""

        self.wordnum = wordnum
        """The word number of this instance's predicate within its
        containing sentence.  Word numbers are indexed starting from
        zero, and include traces and other empty parse elements."""

        self.baseform = baseform
        """The baseform of the predicate."""

        self.sensenumber = sensenumber
        """The sense number of the predicate."""

        self.predicate = predicate
        """A ``NombankTreePointer`` indicating the position of this
        instance's predicate within its containing sentence."""

        self.predid = predid
        """Identifier of the predicate."""

        self.arguments = tuple(arguments)
        """A list of tuples (argloc, argid), specifying the location
        and identifier for each of the predicate's argument in the
        containing sentence.  Argument identifiers are strings such as
        ``'ARG0'`` or ``'ARGM-TMP'``.  This list does *not* contain
        the predicate."""

        self.parse_corpus = parse_corpus
        """A corpus reader for the parse trees corresponding to the
        instances in this nombank corpus."""

    @property
    def roleset(self):
        """The name of the roleset used by this instance's predicate.
        Use ``nombank.roleset() <NombankCorpusReader.roleset>`` to
        look up information about the roleset."""
        r = self.baseform.replace("%", "perc-sign")
        r = r.replace("1/10", "1-slash-10").replace("1-slash-10", "oneslashonezero")
        return f"{r}.{self.sensenumber}"

    def __repr__(self):
        return "<NombankInstance: {}, sent {}, word {}>".format(
            self.fileid,
            self.sentnum,
            self.wordnum,
        )

    def __str__(self):
        s = "{} {} {} {} {}".format(
            self.fileid,
            self.sentnum,
            self.wordnum,
            self.baseform,
            self.sensenumber,
        )
        items = self.arguments + ((self.predicate, "rel"),)
        for (argloc, argid) in sorted(items):
            s += f" {argloc}-{argid}"
        return s

    def _get_tree(self):
        if self.parse_corpus is None:
            return None
        if self.fileid not in self.parse_corpus.fileids():
            return None
        return self.parse_corpus.parsed_sents(self.fileid)[self.sentnum]

    tree = property(
        _get_tree,
        doc="""
        The parse tree corresponding to this instance, or None if
        the corresponding tree is not available.""",
    )

    @staticmethod
    def parse(s, parse_fileid_xform=None, parse_corpus=None):
        pieces = s.split()
        if len(pieces) < 6:
            raise ValueError("Badly formatted nombank line: %r" % s)

        # Divide the line into its basic pieces.
        (fileid, sentnum, wordnum, baseform, sensenumber) = pieces[:5]

        args = pieces[5:]
        rel = [args.pop(i) for i, p in enumerate(args) if "-rel" in p]
        if len(rel) != 1:
            raise ValueError("Badly formatted nombank line: %r" % s)

        # Apply the fileid selector, if any.
        if parse_fileid_xform is not None:
            fileid = parse_fileid_xform(fileid)

        # Convert sentence & word numbers to ints.
        sentnum = int(sentnum)
        wordnum = int(wordnum)

        # Parse the predicate location.

        predloc, predid = rel[0].split("-", 1)
        predicate = NombankTreePointer.parse(predloc)

        # Parse the arguments.
        arguments = []
        for arg in args:
            argloc, argid = arg.split("-", 1)
            arguments.append((NombankTreePointer.parse(argloc), argid))

        # Put it all together.
        return NombankInstance(
            fileid,
            sentnum,
            wordnum,
            baseform,
            sensenumber,
            predicate,
            predid,
            arguments,
            parse_corpus,
        )


class NombankPointer:
    """
    A pointer used by nombank to identify one or more constituents in
    a parse tree.  ``NombankPointer`` is an abstract base class with
    three concrete subclasses:

    - ``NombankTreePointer`` is used to point to single constituents.
    - ``NombankSplitTreePointer`` is used to point to 'split'
      constituents, which consist of a sequence of two or more
      ``NombankTreePointer`` pointers.
    - ``NombankChainTreePointer`` is used to point to entire trace
      chains in a tree.  It consists of a sequence of pieces, which
      can be ``NombankTreePointer`` or ``NombankSplitTreePointer`` pointers.
    """

    def __init__(self):
        if self.__class__ == NombankPointer:
            raise NotImplementedError()


class NombankChainTreePointer(NombankPointer):
    def __init__(self, pieces):
        self.pieces = pieces
        """A list of the pieces that make up this chain.  Elements may
           be either ``NombankSplitTreePointer`` or
           ``NombankTreePointer`` pointers."""

    def __str__(self):
        return "*".join("%s" % p for p in self.pieces)

    def __repr__(self):
        return "<NombankChainTreePointer: %s>" % self

    def select(self, tree):
        if tree is None:
            raise ValueError("Parse tree not available")
        return Tree("*CHAIN*", [p.select(tree) for p in self.pieces])


class NombankSplitTreePointer(NombankPointer):
    def __init__(self, pieces):
        self.pieces = pieces
        """A list of the pieces that make up this chain.  Elements are
           all ``NombankTreePointer`` pointers."""

    def __str__(self):
        return ",".join("%s" % p for p in self.pieces)

    def __repr__(self):
        return "<NombankSplitTreePointer: %s>" % self

    def select(self, tree):
        if tree is None:
            raise ValueError("Parse tree not available")
        return Tree("*SPLIT*", [p.select(tree) for p in self.pieces])


@total_ordering
class NombankTreePointer(NombankPointer):
    """
    wordnum:height*wordnum:height*...
    wordnum:height,

    """

    def __init__(self, wordnum, height):
        self.wordnum = wordnum
        self.height = height

    @staticmethod
    def parse(s):
        # Deal with chains (xx*yy*zz)
        pieces = s.split("*")
        if len(pieces) > 1:
            return NombankChainTreePointer(
                [NombankTreePointer.parse(elt) for elt in pieces]
            )

        # Deal with split args (xx,yy,zz)
        pieces = s.split(",")
        if len(pieces) > 1:
            return NombankSplitTreePointer(
                [NombankTreePointer.parse(elt) for elt in pieces]
            )

        # Deal with normal pointers.
        pieces = s.split(":")
        if len(pieces) != 2:
            raise ValueError("bad nombank pointer %r" % s)
        return NombankTreePointer(int(pieces[0]), int(pieces[1]))

    def __str__(self):
        return f"{self.wordnum}:{self.height}"

    def __repr__(self):
        return "NombankTreePointer(%d, %d)" % (self.wordnum, self.height)

    def __eq__(self, other):
        while isinstance(other, (NombankChainTreePointer, NombankSplitTreePointer)):
            other = other.pieces[0]

        if not isinstance(other, NombankTreePointer):
            return self is other

        return self.wordnum == other.wordnum and self.height == other.height

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        while isinstance(other, (NombankChainTreePointer, NombankSplitTreePointer)):
            other = other.pieces[0]

        if not isinstance(other, NombankTreePointer):
            return id(self) < id(other)

        return (self.wordnum, -self.height) < (other.wordnum, -other.height)

    def select(self, tree):
        if tree is None:
            raise ValueError("Parse tree not available")
        return tree[self.treepos(tree)]

    def treepos(self, tree):
        """
        Convert this pointer to a standard 'tree position' pointer,
        given that it points to the given tree.
        """
        if tree is None:
            raise ValueError("Parse tree not available")
        stack = [tree]
        treepos = []

        wordnum = 0
        while True:
            # tree node:
            if isinstance(stack[-1], Tree):
                # Select the next child.
                if len(treepos) < len(stack):
                    treepos.append(0)
                else:
                    treepos[-1] += 1
                # Update the stack.
                if treepos[-1] < len(stack[-1]):
                    stack.append(stack[-1][treepos[-1]])
                else:
                    # End of node's child list: pop up a level.
                    stack.pop()
                    treepos.pop()
            # word node:
            else:
                if wordnum == self.wordnum:
                    return tuple(treepos[: len(treepos) - self.height - 1])
                else:
                    wordnum += 1
                    stack.pop()
