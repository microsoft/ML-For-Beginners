# Natural Language Toolkit: Corpus Reader Utilities
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree

from nltk.data import (
    FileSystemPathPointer,
    PathPointer,
    SeekableUnicodeStreamReader,
    ZipFilePathPointer,
)
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence

######################################################################
# { Corpus View
######################################################################


class StreamBackedCorpusView(AbstractLazySequence):
    """
    A 'view' of a corpus file, which acts like a sequence of tokens:
    it can be accessed by index, iterated over, etc.  However, the
    tokens are only constructed as-needed -- the entire corpus is
    never stored in memory at once.

    The constructor to ``StreamBackedCorpusView`` takes two arguments:
    a corpus fileid (specified as a string or as a ``PathPointer``);
    and a block reader.  A "block reader" is a function that reads
    zero or more tokens from a stream, and returns them as a list.  A
    very simple example of a block reader is:

        >>> def simple_block_reader(stream):
        ...     return stream.readline().split()

    This simple block reader reads a single line at a time, and
    returns a single token (consisting of a string) for each
    whitespace-separated substring on the line.

    When deciding how to define the block reader for a given
    corpus, careful consideration should be given to the size of
    blocks handled by the block reader.  Smaller block sizes will
    increase the memory requirements of the corpus view's internal
    data structures (by 2 integers per block).  On the other hand,
    larger block sizes may decrease performance for random access to
    the corpus.  (But note that larger block sizes will *not*
    decrease performance for iteration.)

    Internally, ``CorpusView`` maintains a partial mapping from token
    index to file position, with one entry per block.  When a token
    with a given index *i* is requested, the ``CorpusView`` constructs
    it as follows:

      1. First, it searches the toknum/filepos mapping for the token
         index closest to (but less than or equal to) *i*.

      2. Then, starting at the file position corresponding to that
         index, it reads one block at a time using the block reader
         until it reaches the requested token.

    The toknum/filepos mapping is created lazily: it is initially
    empty, but every time a new block is read, the block's
    initial token is added to the mapping.  (Thus, the toknum/filepos
    map has one entry per block.)

    In order to increase efficiency for random access patterns that
    have high degrees of locality, the corpus view may cache one or
    more blocks.

    :note: Each ``CorpusView`` object internally maintains an open file
        object for its underlying corpus file.  This file should be
        automatically closed when the ``CorpusView`` is garbage collected,
        but if you wish to close it manually, use the ``close()``
        method.  If you access a ``CorpusView``'s items after it has been
        closed, the file object will be automatically re-opened.

    :warning: If the contents of the file are modified during the
        lifetime of the ``CorpusView``, then the ``CorpusView``'s behavior
        is undefined.

    :warning: If a unicode encoding is specified when constructing a
        ``CorpusView``, then the block reader may only call
        ``stream.seek()`` with offsets that have been returned by
        ``stream.tell()``; in particular, calling ``stream.seek()`` with
        relative offsets, or with offsets based on string lengths, may
        lead to incorrect behavior.

    :ivar _block_reader: The function used to read
        a single block from the underlying file stream.
    :ivar _toknum: A list containing the token index of each block
        that has been processed.  In particular, ``_toknum[i]`` is the
        token index of the first token in block ``i``.  Together
        with ``_filepos``, this forms a partial mapping between token
        indices and file positions.
    :ivar _filepos: A list containing the file position of each block
        that has been processed.  In particular, ``_toknum[i]`` is the
        file position of the first character in block ``i``.  Together
        with ``_toknum``, this forms a partial mapping between token
        indices and file positions.
    :ivar _stream: The stream used to access the underlying corpus file.
    :ivar _len: The total number of tokens in the corpus, if known;
        or None, if the number of tokens is not yet known.
    :ivar _eofpos: The character position of the last character in the
        file.  This is calculated when the corpus view is initialized,
        and is used to decide when the end of file has been reached.
    :ivar _cache: A cache of the most recently read block.  It
       is encoded as a tuple (start_toknum, end_toknum, tokens), where
       start_toknum is the token index of the first token in the block;
       end_toknum is the token index of the first token not in the
       block; and tokens is a list of the tokens in the block.
    """

    def __init__(self, fileid, block_reader=None, startpos=0, encoding="utf8"):
        """
        Create a new corpus view, based on the file ``fileid``, and
        read with ``block_reader``.  See the class documentation
        for more information.

        :param fileid: The path to the file that is read by this
            corpus view.  ``fileid`` can either be a string or a
            ``PathPointer``.

        :param startpos: The file position at which the view will
            start reading.  This can be used to skip over preface
            sections.

        :param encoding: The unicode encoding that should be used to
            read the file's contents.  If no encoding is specified,
            then the file's contents will be read as a non-unicode
            string (i.e., a str).
        """
        if block_reader:
            self.read_block = block_reader
        # Initialize our toknum/filepos mapping.
        self._toknum = [0]
        self._filepos = [startpos]
        self._encoding = encoding
        # We don't know our length (number of tokens) yet.
        self._len = None

        self._fileid = fileid
        self._stream = None

        self._current_toknum = None
        """This variable is set to the index of the next token that
           will be read, immediately before ``self.read_block()`` is
           called.  This is provided for the benefit of the block
           reader, which under rare circumstances may need to know
           the current token number."""

        self._current_blocknum = None
        """This variable is set to the index of the next block that
           will be read, immediately before ``self.read_block()`` is
           called.  This is provided for the benefit of the block
           reader, which under rare circumstances may need to know
           the current block number."""

        # Find the length of the file.
        try:
            if isinstance(self._fileid, PathPointer):
                self._eofpos = self._fileid.file_size()
            else:
                self._eofpos = os.stat(self._fileid).st_size
        except Exception as exc:
            raise ValueError(f"Unable to open or access {fileid!r} -- {exc}") from exc

        # Maintain a cache of the most recently read block, to
        # increase efficiency of random access.
        self._cache = (-1, -1, None)

    fileid = property(
        lambda self: self._fileid,
        doc="""
        The fileid of the file that is accessed by this view.

        :type: str or PathPointer""",
    )

    def read_block(self, stream):
        """
        Read a block from the input stream.

        :return: a block of tokens from the input stream
        :rtype: list(any)
        :param stream: an input stream
        :type stream: stream
        """
        raise NotImplementedError("Abstract Method")

    def _open(self):
        """
        Open the file stream associated with this corpus view.  This
        will be called performed if any value is read from the view
        while its file stream is closed.
        """
        if isinstance(self._fileid, PathPointer):
            self._stream = self._fileid.open(self._encoding)
        elif self._encoding:
            self._stream = SeekableUnicodeStreamReader(
                open(self._fileid, "rb"), self._encoding
            )
        else:
            self._stream = open(self._fileid, "rb")

    def close(self):
        """
        Close the file stream associated with this corpus view.  This
        can be useful if you are worried about running out of file
        handles (although the stream should automatically be closed
        upon garbage collection of the corpus view).  If the corpus
        view is accessed after it is closed, it will be automatically
        re-opened.
        """
        if self._stream is not None:
            self._stream.close()
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __len__(self):
        if self._len is None:
            # iterate_from() sets self._len when it reaches the end
            # of the file:
            for tok in self.iterate_from(self._toknum[-1]):
                pass
        return self._len

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, stop = slice_bounds(self, i)
            # Check if it's in the cache.
            offset = self._cache[0]
            if offset <= start and stop <= self._cache[1]:
                return self._cache[2][start - offset : stop - offset]
            # Construct & return the result.
            return LazySubsequence(self, start, stop)
        else:
            # Handle negative indices
            if i < 0:
                i += len(self)
            if i < 0:
                raise IndexError("index out of range")
            # Check if it's in the cache.
            offset = self._cache[0]
            if offset <= i < self._cache[1]:
                return self._cache[2][i - offset]
            # Use iterate_from to extract it.
            try:
                return next(self.iterate_from(i))
            except StopIteration as e:
                raise IndexError("index out of range") from e

    # If we wanted to be thread-safe, then this method would need to
    # do some locking.
    def iterate_from(self, start_tok):
        # Start by feeding from the cache, if possible.
        if self._cache[0] <= start_tok < self._cache[1]:
            for tok in self._cache[2][start_tok - self._cache[0] :]:
                yield tok
                start_tok += 1

        # Decide where in the file we should start.  If `start` is in
        # our mapping, then we can jump straight to the correct block;
        # otherwise, start at the last block we've processed.
        if start_tok < self._toknum[-1]:
            block_index = bisect.bisect_right(self._toknum, start_tok) - 1
            toknum = self._toknum[block_index]
            filepos = self._filepos[block_index]
        else:
            block_index = len(self._toknum) - 1
            toknum = self._toknum[-1]
            filepos = self._filepos[-1]

        # Open the stream, if it's not open already.
        if self._stream is None:
            self._open()

        # If the file is empty, the while loop will never run.
        # This *seems* to be all the state we need to set:
        if self._eofpos == 0:
            self._len = 0

        # Each iteration through this loop, we read a single block
        # from the stream.
        while filepos < self._eofpos:
            # Read the next block.
            self._stream.seek(filepos)
            self._current_toknum = toknum
            self._current_blocknum = block_index
            tokens = self.read_block(self._stream)
            assert isinstance(tokens, (tuple, list, AbstractLazySequence)), (
                "block reader %s() should return list or tuple."
                % self.read_block.__name__
            )
            num_toks = len(tokens)
            new_filepos = self._stream.tell()
            assert (
                new_filepos > filepos
            ), "block reader %s() should consume at least 1 byte (filepos=%d)" % (
                self.read_block.__name__,
                filepos,
            )

            # Update our cache.
            self._cache = (toknum, toknum + num_toks, list(tokens))

            # Update our mapping.
            assert toknum <= self._toknum[-1]
            if num_toks > 0:
                block_index += 1
                if toknum == self._toknum[-1]:
                    assert new_filepos > self._filepos[-1]  # monotonic!
                    self._filepos.append(new_filepos)
                    self._toknum.append(toknum + num_toks)
                else:
                    # Check for consistency:
                    assert (
                        new_filepos == self._filepos[block_index]
                    ), "inconsistent block reader (num chars read)"
                    assert (
                        toknum + num_toks == self._toknum[block_index]
                    ), "inconsistent block reader (num tokens returned)"

            # If we reached the end of the file, then update self._len
            if new_filepos == self._eofpos:
                self._len = toknum + num_toks
            # Generate the tokens in this block (but skip any tokens
            # before start_tok).  Note that between yields, our state
            # may be modified.
            for tok in tokens[max(0, start_tok - toknum) :]:
                yield tok
            # If we're at the end of the file, then we're done.
            assert new_filepos <= self._eofpos
            if new_filepos == self._eofpos:
                break
            # Update our indices
            toknum += num_toks
            filepos = new_filepos

        # If we reach this point, then we should know our length.
        assert self._len is not None
        # Enforce closing of stream once we reached end of file
        # We should have reached EOF once we're out of the while loop.
        self.close()

    # Use concat for these, so we can use a ConcatenatedCorpusView
    # when possible.
    def __add__(self, other):
        return concat([self, other])

    def __radd__(self, other):
        return concat([other, self])

    def __mul__(self, count):
        return concat([self] * count)

    def __rmul__(self, count):
        return concat([self] * count)


class ConcatenatedCorpusView(AbstractLazySequence):
    """
    A 'view' of a corpus file that joins together one or more
    ``StreamBackedCorpusViews<StreamBackedCorpusView>``.  At most
    one file handle is left open at any time.
    """

    def __init__(self, corpus_views):
        self._pieces = corpus_views
        """A list of the corpus subviews that make up this
        concatenation."""

        self._offsets = [0]
        """A list of offsets, indicating the index at which each
        subview begins.  In particular::
            offsets[i] = sum([len(p) for p in pieces[:i]])"""

        self._open_piece = None
        """The most recently accessed corpus subview (or None).
        Before a new subview is accessed, this subview will be closed."""

    def __len__(self):
        if len(self._offsets) <= len(self._pieces):
            # Iterate to the end of the corpus.
            for tok in self.iterate_from(self._offsets[-1]):
                pass

        return self._offsets[-1]

    def close(self):
        for piece in self._pieces:
            piece.close()

    def iterate_from(self, start_tok):
        piecenum = bisect.bisect_right(self._offsets, start_tok) - 1

        while piecenum < len(self._pieces):
            offset = self._offsets[piecenum]
            piece = self._pieces[piecenum]

            # If we've got another piece open, close it first.
            if self._open_piece is not piece:
                if self._open_piece is not None:
                    self._open_piece.close()
                self._open_piece = piece

            # Get everything we can from this piece.
            yield from piece.iterate_from(max(0, start_tok - offset))

            # Update the offset table.
            if piecenum + 1 == len(self._offsets):
                self._offsets.append(self._offsets[-1] + len(piece))

            # Move on to the next piece.
            piecenum += 1


def concat(docs):
    """
    Concatenate together the contents of multiple documents from a
    single corpus, using an appropriate concatenation function.  This
    utility function is used by corpus readers when the user requests
    more than one document at a time.
    """
    if len(docs) == 1:
        return docs[0]
    if len(docs) == 0:
        raise ValueError("concat() expects at least one object!")

    types = {d.__class__ for d in docs}

    # If they're all strings, use string concatenation.
    if all(isinstance(doc, str) for doc in docs):
        return "".join(docs)

    # If they're all corpus views, then use ConcatenatedCorpusView.
    for typ in types:
        if not issubclass(typ, (StreamBackedCorpusView, ConcatenatedCorpusView)):
            break
    else:
        return ConcatenatedCorpusView(docs)

    # If they're all lazy sequences, use a lazy concatenation
    for typ in types:
        if not issubclass(typ, AbstractLazySequence):
            break
    else:
        return LazyConcatenation(docs)

    # Otherwise, see what we can do:
    if len(types) == 1:
        typ = list(types)[0]

        if issubclass(typ, list):
            return reduce((lambda a, b: a + b), docs, [])

        if issubclass(typ, tuple):
            return reduce((lambda a, b: a + b), docs, ())

        if ElementTree.iselement(typ):
            xmltree = ElementTree.Element("documents")
            for doc in docs:
                xmltree.append(doc)
            return xmltree

    # No method found!
    raise ValueError("Don't know how to concatenate types: %r" % types)


######################################################################
# { Corpus View for Pickled Sequences
######################################################################


class PickleCorpusView(StreamBackedCorpusView):
    """
    A stream backed corpus view for corpus files that consist of
    sequences of serialized Python objects (serialized using
    ``pickle.dump``).  One use case for this class is to store the
    result of running feature detection on a corpus to disk.  This can
    be useful when performing feature detection is expensive (so we
    don't want to repeat it); but the corpus is too large to store in
    memory.  The following example illustrates this technique:

        >>> from nltk.corpus.reader.util import PickleCorpusView
        >>> from nltk.util import LazyMap
        >>> feature_corpus = LazyMap(detect_features, corpus) # doctest: +SKIP
        >>> PickleCorpusView.write(feature_corpus, some_fileid)  # doctest: +SKIP
        >>> pcv = PickleCorpusView(some_fileid) # doctest: +SKIP
    """

    BLOCK_SIZE = 100
    PROTOCOL = -1

    def __init__(self, fileid, delete_on_gc=False):
        """
        Create a new corpus view that reads the pickle corpus
        ``fileid``.

        :param delete_on_gc: If true, then ``fileid`` will be deleted
            whenever this object gets garbage-collected.
        """
        self._delete_on_gc = delete_on_gc
        StreamBackedCorpusView.__init__(self, fileid)

    def read_block(self, stream):
        result = []
        for i in range(self.BLOCK_SIZE):
            try:
                result.append(pickle.load(stream))
            except EOFError:
                break
        return result

    def __del__(self):
        """
        If ``delete_on_gc`` was set to true when this
        ``PickleCorpusView`` was created, then delete the corpus view's
        fileid.  (This method is called whenever a
        ``PickledCorpusView`` is garbage-collected.
        """
        if getattr(self, "_delete_on_gc"):
            if os.path.exists(self._fileid):
                try:
                    os.remove(self._fileid)
                except OSError:
                    pass
        self.__dict__.clear()  # make the garbage collector's job easier

    @classmethod
    def write(cls, sequence, output_file):
        if isinstance(output_file, str):
            output_file = open(output_file, "wb")
        for item in sequence:
            pickle.dump(item, output_file, cls.PROTOCOL)

    @classmethod
    def cache_to_tempfile(cls, sequence, delete_on_gc=True):
        """
        Write the given sequence to a temporary file as a pickle
        corpus; and then return a ``PickleCorpusView`` view for that
        temporary corpus file.

        :param delete_on_gc: If true, then the temporary file will be
            deleted whenever this object gets garbage-collected.
        """
        try:
            fd, output_file_name = tempfile.mkstemp(".pcv", "nltk-")
            output_file = os.fdopen(fd, "wb")
            cls.write(sequence, output_file)
            output_file.close()
            return PickleCorpusView(output_file_name, delete_on_gc)
        except OSError as e:
            raise ValueError("Error while creating temp file: %s" % e) from e


######################################################################
# { Block Readers
######################################################################


def read_whitespace_block(stream):
    toks = []
    for i in range(20):  # Read 20 lines at a time.
        toks.extend(stream.readline().split())
    return toks


def read_wordpunct_block(stream):
    toks = []
    for i in range(20):  # Read 20 lines at a time.
        toks.extend(wordpunct_tokenize(stream.readline()))
    return toks


def read_line_block(stream):
    toks = []
    for i in range(20):
        line = stream.readline()
        if not line:
            return toks
        toks.append(line.rstrip("\n"))
    return toks


def read_blankline_block(stream):
    s = ""
    while True:
        line = stream.readline()
        # End of file:
        if not line:
            if s:
                return [s]
            else:
                return []
        # Blank line:
        elif line and not line.strip():
            if s:
                return [s]
        # Other line:
        else:
            s += line


def read_alignedsent_block(stream):
    s = ""
    while True:
        line = stream.readline()
        if line[0] == "=" or line[0] == "\n" or line[:2] == "\r\n":
            continue
        # End of file:
        if not line:
            if s:
                return [s]
            else:
                return []
        # Other line:
        else:
            s += line
            if re.match(r"^\d+-\d+", line) is not None:
                return [s]


def read_regexp_block(stream, start_re, end_re=None):
    """
    Read a sequence of tokens from a stream, where tokens begin with
    lines that match ``start_re``.  If ``end_re`` is specified, then
    tokens end with lines that match ``end_re``; otherwise, tokens end
    whenever the next line matching ``start_re`` or EOF is found.
    """
    # Scan until we find a line matching the start regexp.
    while True:
        line = stream.readline()
        if not line:
            return []  # end of file.
        if re.match(start_re, line):
            break

    # Scan until we find another line matching the regexp, or EOF.
    lines = [line]
    while True:
        oldpos = stream.tell()
        line = stream.readline()
        # End of file:
        if not line:
            return ["".join(lines)]
        # End of token:
        if end_re is not None and re.match(end_re, line):
            return ["".join(lines)]
        # Start of new token: backup to just before it starts, and
        # return the token we've already collected.
        if end_re is None and re.match(start_re, line):
            stream.seek(oldpos)
            return ["".join(lines)]
        # Anything else is part of the token.
        lines.append(line)


def read_sexpr_block(stream, block_size=16384, comment_char=None):
    """
    Read a sequence of s-expressions from the stream, and leave the
    stream's file position at the end the last complete s-expression
    read.  This function will always return at least one s-expression,
    unless there are no more s-expressions in the file.

    If the file ends in in the middle of an s-expression, then that
    incomplete s-expression is returned when the end of the file is
    reached.

    :param block_size: The default block size for reading.  If an
        s-expression is longer than one block, then more than one
        block will be read.
    :param comment_char: A character that marks comments.  Any lines
        that begin with this character will be stripped out.
        (If spaces or tabs precede the comment character, then the
        line will not be stripped.)
    """
    start = stream.tell()
    block = stream.read(block_size)
    encoding = getattr(stream, "encoding", None)
    assert encoding is not None or isinstance(block, str)
    if encoding not in (None, "utf-8"):
        import warnings

        warnings.warn(
            "Parsing may fail, depending on the properties "
            "of the %s encoding!" % encoding
        )
        # (e.g., the utf-16 encoding does not work because it insists
        # on adding BOMs to the beginning of encoded strings.)

    if comment_char:
        COMMENT = re.compile("(?m)^%s.*$" % re.escape(comment_char))
    while True:
        try:
            # If we're stripping comments, then make sure our block ends
            # on a line boundary; and then replace any comments with
            # space characters.  (We can't just strip them out -- that
            # would make our offset wrong.)
            if comment_char:
                block += stream.readline()
                block = re.sub(COMMENT, _sub_space, block)
            # Read the block.
            tokens, offset = _parse_sexpr_block(block)
            # Skip whitespace
            offset = re.compile(r"\s*").search(block, offset).end()

            # Move to the end position.
            if encoding is None:
                stream.seek(start + offset)
            else:
                stream.seek(start + len(block[:offset].encode(encoding)))

            # Return the list of tokens we processed
            return tokens
        except ValueError as e:
            if e.args[0] == "Block too small":
                next_block = stream.read(block_size)
                if next_block:
                    block += next_block
                    continue
                else:
                    # The file ended mid-sexpr -- return what we got.
                    return [block.strip()]
            else:
                raise


def _sub_space(m):
    """Helper function: given a regexp match, return a string of
    spaces that's the same length as the matched string."""
    return " " * (m.end() - m.start())


def _parse_sexpr_block(block):
    tokens = []
    start = end = 0

    while end < len(block):
        m = re.compile(r"\S").search(block, end)
        if not m:
            return tokens, end

        start = m.start()

        # Case 1: sexpr is not parenthesized.
        if m.group() != "(":
            m2 = re.compile(r"[\s(]").search(block, start)
            if m2:
                end = m2.start()
            else:
                if tokens:
                    return tokens, end
                raise ValueError("Block too small")

        # Case 2: parenthesized sexpr.
        else:
            nesting = 0
            for m in re.compile(r"[()]").finditer(block, start):
                if m.group() == "(":
                    nesting += 1
                else:
                    nesting -= 1
                if nesting == 0:
                    end = m.end()
                    break
            else:
                if tokens:
                    return tokens, end
                raise ValueError("Block too small")

        tokens.append(block[start:end])

    return tokens, end


######################################################################
# { Finding Corpus Items
######################################################################


def find_corpus_fileids(root, regexp):
    if not isinstance(root, PathPointer):
        raise TypeError("find_corpus_fileids: expected a PathPointer")
    regexp += "$"

    # Find fileids in a zipfile: scan the zipfile's namelist.  Filter
    # out entries that end in '/' -- they're directories.
    if isinstance(root, ZipFilePathPointer):
        fileids = [
            name[len(root.entry) :]
            for name in root.zipfile.namelist()
            if not name.endswith("/")
        ]
        items = [name for name in fileids if re.match(regexp, name)]
        return sorted(items)

    # Find fileids in a directory: use os.walk to search all (proper
    # or symlinked) subdirectories, and match paths against the regexp.
    elif isinstance(root, FileSystemPathPointer):
        items = []
        for dirname, subdirs, fileids in os.walk(root.path):
            prefix = "".join("%s/" % p for p in _path_from(root.path, dirname))
            items += [
                prefix + fileid
                for fileid in fileids
                if re.match(regexp, prefix + fileid)
            ]
            # Don't visit svn directories:
            if ".svn" in subdirs:
                subdirs.remove(".svn")
        return sorted(items)

    else:
        raise AssertionError("Don't know how to handle %r" % root)


def _path_from(parent, child):
    if os.path.split(parent)[1] == "":
        parent = os.path.split(parent)[0]
    path = []
    while parent != child:
        child, dirname = os.path.split(child)
        path.insert(0, dirname)
        assert os.path.split(child)[0] != child
    return path


######################################################################
# { Paragraph structure in Treebank files
######################################################################


def tagged_treebank_para_block_reader(stream):
    # Read the next paragraph.
    para = ""
    while True:
        line = stream.readline()
        # End of paragraph:
        if re.match(r"======+\s*$", line):
            if para.strip():
                return [para]
        # End of file:
        elif line == "":
            if para.strip():
                return [para]
            else:
                return []
        # Content line:
        else:
            para += line
