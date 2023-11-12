# Natural Language Toolkit: CONLL Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Read CoNLL-style chunk fileids.
"""

import textwrap

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap


class ConllCorpusReader(CorpusReader):
    """
    A corpus reader for CoNLL-style files.  These files consist of a
    series of sentences, separated by blank lines.  Each sentence is
    encoded using a table (or "grid") of values, where each line
    corresponds to a single word, and each column corresponds to an
    annotation type.  The set of columns used by CoNLL-style files can
    vary from corpus to corpus; the ``ConllCorpusReader`` constructor
    therefore takes an argument, ``columntypes``, which is used to
    specify the columns that are used by a given corpus. By default
    columns are split by consecutive whitespaces, with the
    ``separator`` argument you can set a string to split by (e.g.
    ``\'\t\'``).


    @todo: Add support for reading from corpora where different
        parallel files contain different columns.
    @todo: Possibly add caching of the grid corpus view?  This would
        allow the same grid view to be used by different data access
        methods (eg words() and parsed_sents() could both share the
        same grid corpus view object).
    @todo: Better support for -DOCSTART-.  Currently, we just ignore
        it, but it could be used to define methods that retrieve a
        document at a time (eg parsed_documents()).
    """

    # /////////////////////////////////////////////////////////////////
    # Column Types
    # /////////////////////////////////////////////////////////////////

    WORDS = "words"  #: column type for words
    POS = "pos"  #: column type for part-of-speech tags
    TREE = "tree"  #: column type for parse trees
    CHUNK = "chunk"  #: column type for chunk structures
    NE = "ne"  #: column type for named entities
    SRL = "srl"  #: column type for semantic role labels
    IGNORE = "ignore"  #: column type for column that should be ignored

    #: A list of all column types supported by the conll corpus reader.
    COLUMN_TYPES = (WORDS, POS, TREE, CHUNK, NE, SRL, IGNORE)

    # /////////////////////////////////////////////////////////////////
    # Constructor
    # /////////////////////////////////////////////////////////////////

    def __init__(
        self,
        root,
        fileids,
        columntypes,
        chunk_types=None,
        root_label="S",
        pos_in_tree=False,
        srl_includes_roleset=True,
        encoding="utf8",
        tree_class=Tree,
        tagset=None,
        separator=None,
    ):
        for columntype in columntypes:
            if columntype not in self.COLUMN_TYPES:
                raise ValueError("Bad column type %r" % columntype)
        if isinstance(chunk_types, str):
            chunk_types = [chunk_types]
        self._chunk_types = chunk_types
        self._colmap = {c: i for (i, c) in enumerate(columntypes)}
        self._pos_in_tree = pos_in_tree
        self._root_label = root_label  # for chunks
        self._srl_includes_roleset = srl_includes_roleset
        self._tree_class = tree_class
        CorpusReader.__init__(self, root, fileids, encoding)
        self._tagset = tagset
        self.sep = separator

    # /////////////////////////////////////////////////////////////////
    # Data Access Methods
    # /////////////////////////////////////////////////////////////////

    def words(self, fileids=None):
        self._require(self.WORDS)
        return LazyConcatenation(LazyMap(self._get_words, self._grids(fileids)))

    def sents(self, fileids=None):
        self._require(self.WORDS)
        return LazyMap(self._get_words, self._grids(fileids))

    def tagged_words(self, fileids=None, tagset=None):
        self._require(self.WORDS, self.POS)

        def get_tagged_words(grid):
            return self._get_tagged_words(grid, tagset)

        return LazyConcatenation(LazyMap(get_tagged_words, self._grids(fileids)))

    def tagged_sents(self, fileids=None, tagset=None):
        self._require(self.WORDS, self.POS)

        def get_tagged_words(grid):
            return self._get_tagged_words(grid, tagset)

        return LazyMap(get_tagged_words, self._grids(fileids))

    def chunked_words(self, fileids=None, chunk_types=None, tagset=None):
        self._require(self.WORDS, self.POS, self.CHUNK)
        if chunk_types is None:
            chunk_types = self._chunk_types

        def get_chunked_words(grid):  # capture chunk_types as local var
            return self._get_chunked_words(grid, chunk_types, tagset)

        return LazyConcatenation(LazyMap(get_chunked_words, self._grids(fileids)))

    def chunked_sents(self, fileids=None, chunk_types=None, tagset=None):
        self._require(self.WORDS, self.POS, self.CHUNK)
        if chunk_types is None:
            chunk_types = self._chunk_types

        def get_chunked_words(grid):  # capture chunk_types as local var
            return self._get_chunked_words(grid, chunk_types, tagset)

        return LazyMap(get_chunked_words, self._grids(fileids))

    def parsed_sents(self, fileids=None, pos_in_tree=None, tagset=None):
        self._require(self.WORDS, self.POS, self.TREE)
        if pos_in_tree is None:
            pos_in_tree = self._pos_in_tree

        def get_parsed_sent(grid):  # capture pos_in_tree as local var
            return self._get_parsed_sent(grid, pos_in_tree, tagset)

        return LazyMap(get_parsed_sent, self._grids(fileids))

    def srl_spans(self, fileids=None):
        self._require(self.SRL)
        return LazyMap(self._get_srl_spans, self._grids(fileids))

    def srl_instances(self, fileids=None, pos_in_tree=None, flatten=True):
        self._require(self.WORDS, self.POS, self.TREE, self.SRL)
        if pos_in_tree is None:
            pos_in_tree = self._pos_in_tree

        def get_srl_instances(grid):  # capture pos_in_tree as local var
            return self._get_srl_instances(grid, pos_in_tree)

        result = LazyMap(get_srl_instances, self._grids(fileids))
        if flatten:
            result = LazyConcatenation(result)
        return result

    def iob_words(self, fileids=None, tagset=None):
        """
        :return: a list of word/tag/IOB tuples
        :rtype: list(tuple)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.WORDS, self.POS, self.CHUNK)

        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset)

        return LazyConcatenation(LazyMap(get_iob_words, self._grids(fileids)))

    def iob_sents(self, fileids=None, tagset=None):
        """
        :return: a list of lists of word/tag/IOB tuples
        :rtype: list(list)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.WORDS, self.POS, self.CHUNK)

        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset)

        return LazyMap(get_iob_words, self._grids(fileids))

    # /////////////////////////////////////////////////////////////////
    # Grid Reading
    # /////////////////////////////////////////////////////////////////

    def _grids(self, fileids=None):
        # n.b.: we could cache the object returned here (keyed on
        # fileids), which would let us reuse the same corpus view for
        # different things (eg srl and parse trees).
        return concat(
            [
                StreamBackedCorpusView(fileid, self._read_grid_block, encoding=enc)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def _read_grid_block(self, stream):
        grids = []
        for block in read_blankline_block(stream):
            block = block.strip()
            if not block:
                continue

            grid = [line.split(self.sep) for line in block.split("\n")]

            # If there's a docstart row, then discard. ([xx] eventually it
            # would be good to actually use it)
            if grid[0][self._colmap.get("words", 0)] == "-DOCSTART-":
                del grid[0]

            # Check that the grid is consistent.
            for row in grid:
                if len(row) != len(grid[0]):
                    raise ValueError("Inconsistent number of columns:\n%s" % block)
            grids.append(grid)
        return grids

    # /////////////////////////////////////////////////////////////////
    # Transforms
    # /////////////////////////////////////////////////////////////////
    # given a grid, transform it into some representation (e.g.,
    # a list of words or a parse tree).

    def _get_words(self, grid):
        return self._get_column(grid, self._colmap["words"])

    def _get_tagged_words(self, grid, tagset=None):
        pos_tags = self._get_column(grid, self._colmap["pos"])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(zip(self._get_column(grid, self._colmap["words"]), pos_tags))

    def _get_iob_words(self, grid, tagset=None):
        pos_tags = self._get_column(grid, self._colmap["pos"])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(
            zip(
                self._get_column(grid, self._colmap["words"]),
                pos_tags,
                self._get_column(grid, self._colmap["chunk"]),
            )
        )

    def _get_chunked_words(self, grid, chunk_types, tagset=None):
        # n.b.: this method is very similar to conllstr2tree.
        words = self._get_column(grid, self._colmap["words"])
        pos_tags = self._get_column(grid, self._colmap["pos"])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        chunk_tags = self._get_column(grid, self._colmap["chunk"])

        stack = [Tree(self._root_label, [])]

        for (word, pos_tag, chunk_tag) in zip(words, pos_tags, chunk_tags):
            if chunk_tag == "O":
                state, chunk_type = "O", ""
            else:
                (state, chunk_type) = chunk_tag.split("-")
            # If it's a chunk we don't care about, treat it as O.
            if chunk_types is not None and chunk_type not in chunk_types:
                state = "O"
            # Treat a mismatching I like a B.
            if state == "I" and chunk_type != stack[-1].label():
                state = "B"
            # For B or I: close any open chunks
            if state in "BO" and len(stack) == 2:
                stack.pop()
            # For B: start a new chunk.
            if state == "B":
                new_chunk = Tree(chunk_type, [])
                stack[-1].append(new_chunk)
                stack.append(new_chunk)
            # Add the word token.
            stack[-1].append((word, pos_tag))

        return stack[0]

    def _get_parsed_sent(self, grid, pos_in_tree, tagset=None):
        words = self._get_column(grid, self._colmap["words"])
        pos_tags = self._get_column(grid, self._colmap["pos"])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        parse_tags = self._get_column(grid, self._colmap["tree"])

        treestr = ""
        for (word, pos_tag, parse_tag) in zip(words, pos_tags, parse_tags):
            if word == "(":
                word = "-LRB-"
            if word == ")":
                word = "-RRB-"
            if pos_tag == "(":
                pos_tag = "-LRB-"
            if pos_tag == ")":
                pos_tag = "-RRB-"
            (left, right) = parse_tag.split("*")
            right = right.count(")") * ")"  # only keep ')'.
            treestr += f"{left} ({pos_tag} {word}) {right}"
        try:
            tree = self._tree_class.fromstring(treestr)
        except (ValueError, IndexError):
            tree = self._tree_class.fromstring(f"({self._root_label} {treestr})")

        if not pos_in_tree:
            for subtree in tree.subtrees():
                for i, child in enumerate(subtree):
                    if (
                        isinstance(child, Tree)
                        and len(child) == 1
                        and isinstance(child[0], str)
                    ):
                        subtree[i] = (child[0], child.label())

        return tree

    def _get_srl_spans(self, grid):
        """
        list of list of (start, end), tag) tuples
        """
        if self._srl_includes_roleset:
            predicates = self._get_column(grid, self._colmap["srl"] + 1)
            start_col = self._colmap["srl"] + 2
        else:
            predicates = self._get_column(grid, self._colmap["srl"])
            start_col = self._colmap["srl"] + 1

        # Count how many predicates there are.  This tells us how many
        # columns to expect for SRL data.
        num_preds = len([p for p in predicates if p != "-"])

        spanlists = []
        for i in range(num_preds):
            col = self._get_column(grid, start_col + i)
            spanlist = []
            stack = []
            for wordnum, srl_tag in enumerate(col):
                (left, right) = srl_tag.split("*")
                for tag in left.split("("):
                    if tag:
                        stack.append((tag, wordnum))
                for i in range(right.count(")")):
                    (tag, start) = stack.pop()
                    spanlist.append(((start, wordnum + 1), tag))
            spanlists.append(spanlist)

        return spanlists

    def _get_srl_instances(self, grid, pos_in_tree):
        tree = self._get_parsed_sent(grid, pos_in_tree)
        spanlists = self._get_srl_spans(grid)
        if self._srl_includes_roleset:
            predicates = self._get_column(grid, self._colmap["srl"] + 1)
            rolesets = self._get_column(grid, self._colmap["srl"])
        else:
            predicates = self._get_column(grid, self._colmap["srl"])
            rolesets = [None] * len(predicates)

        instances = ConllSRLInstanceList(tree)
        for wordnum, predicate in enumerate(predicates):
            if predicate == "-":
                continue
            # Decide which spanlist to use.  Don't assume that they're
            # sorted in the same order as the predicates (even though
            # they usually are).
            for spanlist in spanlists:
                for (start, end), tag in spanlist:
                    if wordnum in range(start, end) and tag in ("V", "C-V"):
                        break
                else:
                    continue
                break
            else:
                raise ValueError("No srl column found for %r" % predicate)
            instances.append(
                ConllSRLInstance(tree, wordnum, predicate, rolesets[wordnum], spanlist)
            )

        return instances

    # /////////////////////////////////////////////////////////////////
    # Helper Methods
    # /////////////////////////////////////////////////////////////////

    def _require(self, *columntypes):
        for columntype in columntypes:
            if columntype not in self._colmap:
                raise ValueError(
                    "This corpus does not contain a %s " "column." % columntype
                )

    @staticmethod
    def _get_column(grid, column_index):
        return [grid[i][column_index] for i in range(len(grid))]


class ConllSRLInstance:
    """
    An SRL instance from a CoNLL corpus, which identifies and
    providing labels for the arguments of a single verb.
    """

    # [xx] add inst.core_arguments, inst.argm_arguments?

    def __init__(self, tree, verb_head, verb_stem, roleset, tagged_spans):
        self.verb = []
        """A list of the word indices of the words that compose the
           verb whose arguments are identified by this instance.
           This will contain multiple word indices when multi-word
           verbs are used (e.g. 'turn on')."""

        self.verb_head = verb_head
        """The word index of the head word of the verb whose arguments
           are identified by this instance.  E.g., for a sentence that
           uses the verb 'turn on,' ``verb_head`` will be the word index
           of the word 'turn'."""

        self.verb_stem = verb_stem

        self.roleset = roleset

        self.arguments = []
        """A list of ``(argspan, argid)`` tuples, specifying the location
           and type for each of the arguments identified by this
           instance.  ``argspan`` is a tuple ``start, end``, indicating
           that the argument consists of the ``words[start:end]``."""

        self.tagged_spans = tagged_spans
        """A list of ``(span, id)`` tuples, specifying the location and
           type for each of the arguments, as well as the verb pieces,
           that make up this instance."""

        self.tree = tree
        """The parse tree for the sentence containing this instance."""

        self.words = tree.leaves()
        """A list of the words in the sentence containing this
           instance."""

        # Fill in the self.verb and self.arguments values.
        for (start, end), tag in tagged_spans:
            if tag in ("V", "C-V"):
                self.verb += list(range(start, end))
            else:
                self.arguments.append(((start, end), tag))

    def __repr__(self):
        # Originally, its:
        ##plural = 's' if len(self.arguments) != 1 else ''
        plural = "s" if len(self.arguments) != 1 else ""
        return "<ConllSRLInstance for %r with %d argument%s>" % (
            (self.verb_stem, len(self.arguments), plural)
        )

    def pprint(self):
        verbstr = " ".join(self.words[i][0] for i in self.verb)
        hdr = f"SRL for {verbstr!r} (stem={self.verb_stem!r}):\n"
        s = ""
        for i, word in enumerate(self.words):
            if isinstance(word, tuple):
                word = word[0]
            for (start, end), argid in self.arguments:
                if i == start:
                    s += "[%s " % argid
                if i == end:
                    s += "] "
            if i in self.verb:
                word = "<<%s>>" % word
            s += word + " "
        return hdr + textwrap.fill(
            s.replace(" ]", "]"), initial_indent="    ", subsequent_indent="    "
        )


class ConllSRLInstanceList(list):
    """
    Set of instances for a single sentence
    """

    def __init__(self, tree, instances=()):
        self.tree = tree
        list.__init__(self, instances)

    def __str__(self):
        return self.pprint()

    def pprint(self, include_tree=False):
        # Sanity check: trees should be the same
        for inst in self:
            if inst.tree != self.tree:
                raise ValueError("Tree mismatch!")

        # If desired, add trees:
        if include_tree:
            words = self.tree.leaves()
            pos = [None] * len(words)
            synt = ["*"] * len(words)
            self._tree2conll(self.tree, 0, words, pos, synt)

        s = ""
        for i in range(len(words)):
            # optional tree columns
            if include_tree:
                s += "%-20s " % words[i]
                s += "%-8s " % pos[i]
                s += "%15s*%-8s " % tuple(synt[i].split("*"))

            # verb head column
            for inst in self:
                if i == inst.verb_head:
                    s += "%-20s " % inst.verb_stem
                    break
            else:
                s += "%-20s " % "-"
            # Remaining columns: self
            for inst in self:
                argstr = "*"
                for (start, end), argid in inst.tagged_spans:
                    if i == start:
                        argstr = f"({argid}{argstr}"
                    if i == (end - 1):
                        argstr += ")"
                s += "%-12s " % argstr
            s += "\n"
        return s

    def _tree2conll(self, tree, wordnum, words, pos, synt):
        assert isinstance(tree, Tree)
        if len(tree) == 1 and isinstance(tree[0], str):
            pos[wordnum] = tree.label()
            assert words[wordnum] == tree[0]
            return wordnum + 1
        elif len(tree) == 1 and isinstance(tree[0], tuple):
            assert len(tree[0]) == 2
            pos[wordnum], pos[wordnum] = tree[0]
            return wordnum + 1
        else:
            synt[wordnum] = f"({tree.label()}{synt[wordnum]}"
            for child in tree:
                wordnum = self._tree2conll(child, wordnum, words, pos, synt)
            synt[wordnum - 1] += ")"
            return wordnum


class ConllChunkCorpusReader(ConllCorpusReader):
    """
    A ConllCorpusReader whose data file contains three columns: words,
    pos, and chunk.
    """

    def __init__(
        self, root, fileids, chunk_types, encoding="utf8", tagset=None, separator=None
    ):
        ConllCorpusReader.__init__(
            self,
            root,
            fileids,
            ("words", "pos", "chunk"),
            chunk_types=chunk_types,
            encoding=encoding,
            tagset=tagset,
            separator=separator,
        )
