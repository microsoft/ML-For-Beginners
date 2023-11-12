# Natural Language Toolkit: XML Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Corpus reader for corpora whose documents are xml files.

(note -- not named 'xml' to avoid conflicting w/ standard xml package)
"""

import codecs
from xml.etree import ElementTree

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import *
from nltk.data import SeekableUnicodeStreamReader
from nltk.internals import ElementWrapper
from nltk.tokenize import WordPunctTokenizer


class XMLCorpusReader(CorpusReader):
    """
    Corpus reader for corpora whose documents are xml files.

    Note that the ``XMLCorpusReader`` constructor does not take an
    ``encoding`` argument, because the unicode encoding is specified by
    the XML files themselves.  See the XML specs for more info.
    """

    def __init__(self, root, fileids, wrap_etree=False):
        self._wrap_etree = wrap_etree
        CorpusReader.__init__(self, root, fileids)

    def xml(self, fileid=None):
        # Make sure we have exactly one file -- no concatenating XML.
        if fileid is None and len(self._fileids) == 1:
            fileid = self._fileids[0]
        if not isinstance(fileid, str):
            raise TypeError("Expected a single file identifier string")
        # Read the XML in using ElementTree.
        with self.abspath(fileid).open() as fp:
            elt = ElementTree.parse(fp).getroot()
        # If requested, wrap it.
        if self._wrap_etree:
            elt = ElementWrapper(elt)
        # Return the ElementTree element.
        return elt

    def words(self, fileid=None):
        """
        Returns all of the words and punctuation symbols in the specified file
        that were in text nodes -- ie, tags are ignored. Like the xml() method,
        fileid can only specify one file.

        :return: the given file's text nodes as a list of words and punctuation symbols
        :rtype: list(str)
        """

        elt = self.xml(fileid)
        encoding = self.encoding(fileid)
        word_tokenizer = WordPunctTokenizer()
        try:
            iterator = elt.getiterator()
        except:
            iterator = elt.iter()
        out = []

        for node in iterator:
            text = node.text
            if text is not None:
                if isinstance(text, bytes):
                    text = text.decode(encoding)
                toks = word_tokenizer.tokenize(text)
                out.extend(toks)
        return out


class XMLCorpusView(StreamBackedCorpusView):
    """
    A corpus view that selects out specified elements from an XML
    file, and provides a flat list-like interface for accessing them.
    (Note: ``XMLCorpusView`` is not used by ``XMLCorpusReader`` itself,
    but may be used by subclasses of ``XMLCorpusReader``.)

    Every XML corpus view has a "tag specification", indicating what
    XML elements should be included in the view; and each (non-nested)
    element that matches this specification corresponds to one item in
    the view.  Tag specifications are regular expressions over tag
    paths, where a tag path is a list of element tag names, separated
    by '/', indicating the ancestry of the element.  Some examples:

      - ``'foo'``: A top-level element whose tag is ``foo``.
      - ``'foo/bar'``: An element whose tag is ``bar`` and whose parent
        is a top-level element whose tag is ``foo``.
      - ``'.*/foo'``: An element whose tag is ``foo``, appearing anywhere
        in the xml tree.
      - ``'.*/(foo|bar)'``: An wlement whose tag is ``foo`` or ``bar``,
        appearing anywhere in the xml tree.

    The view items are generated from the selected XML elements via
    the method ``handle_elt()``.  By default, this method returns the
    element as-is (i.e., as an ElementTree object); but it can be
    overridden, either via subclassing or via the ``elt_handler``
    constructor parameter.
    """

    #: If true, then display debugging output to stdout when reading
    #: blocks.
    _DEBUG = False

    #: The number of characters read at a time by this corpus reader.
    _BLOCK_SIZE = 1024

    def __init__(self, fileid, tagspec, elt_handler=None):
        """
        Create a new corpus view based on a specified XML file.

        Note that the ``XMLCorpusView`` constructor does not take an
        ``encoding`` argument, because the unicode encoding is
        specified by the XML files themselves.

        :type tagspec: str
        :param tagspec: A tag specification, indicating what XML
            elements should be included in the view.  Each non-nested
            element that matches this specification corresponds to one
            item in the view.

        :param elt_handler: A function used to transform each element
            to a value for the view.  If no handler is specified, then
            ``self.handle_elt()`` is called, which returns the element
            as an ElementTree object.  The signature of elt_handler is::

                elt_handler(elt, tagspec) -> value
        """
        if elt_handler:
            self.handle_elt = elt_handler

        self._tagspec = re.compile(tagspec + r"\Z")
        """The tag specification for this corpus view."""

        self._tag_context = {0: ()}
        """A dictionary mapping from file positions (as returned by
           ``stream.seek()`` to XML contexts.  An XML context is a
           tuple of XML tag names, indicating which tags have not yet
           been closed."""

        encoding = self._detect_encoding(fileid)
        StreamBackedCorpusView.__init__(self, fileid, encoding=encoding)

    def _detect_encoding(self, fileid):
        if isinstance(fileid, PathPointer):
            try:
                infile = fileid.open()
                s = infile.readline()
            finally:
                infile.close()
        else:
            with open(fileid, "rb") as infile:
                s = infile.readline()
        if s.startswith(codecs.BOM_UTF16_BE):
            return "utf-16-be"
        if s.startswith(codecs.BOM_UTF16_LE):
            return "utf-16-le"
        if s.startswith(codecs.BOM_UTF32_BE):
            return "utf-32-be"
        if s.startswith(codecs.BOM_UTF32_LE):
            return "utf-32-le"
        if s.startswith(codecs.BOM_UTF8):
            return "utf-8"
        m = re.match(rb'\s*<\?xml\b.*\bencoding="([^"]+)"', s)
        if m:
            return m.group(1).decode()
        m = re.match(rb"\s*<\?xml\b.*\bencoding='([^']+)'", s)
        if m:
            return m.group(1).decode()
        # No encoding found -- what should the default be?
        return "utf-8"

    def handle_elt(self, elt, context):
        """
        Convert an element into an appropriate value for inclusion in
        the view.  Unless overridden by a subclass or by the
        ``elt_handler`` constructor argument, this method simply
        returns ``elt``.

        :return: The view value corresponding to ``elt``.

        :type elt: ElementTree
        :param elt: The element that should be converted.

        :type context: str
        :param context: A string composed of element tags separated by
            forward slashes, indicating the XML context of the given
            element.  For example, the string ``'foo/bar/baz'``
            indicates that the element is a ``baz`` element whose
            parent is a ``bar`` element and whose grandparent is a
            top-level ``foo`` element.
        """
        return elt

    #: A regular expression that matches XML fragments that do not
    #: contain any un-closed tags.
    _VALID_XML_RE = re.compile(
        r"""
        [^<]*
        (
          ((<!--.*?-->)                         |  # comment
           (<![CDATA[.*?]])                     |  # raw character data
           (<!DOCTYPE\s+[^\[]*(\[[^\]]*])?\s*>) |  # doctype decl
           (<[^!>][^>]*>))                         # tag or PI
          [^<]*)*
        \Z""",
        re.DOTALL | re.VERBOSE,
    )

    #: A regular expression used to extract the tag name from a start tag,
    #: end tag, or empty-elt tag string.
    _XML_TAG_NAME = re.compile(r"<\s*(?:/\s*)?([^\s>]+)")

    #: A regular expression used to find all start-tags, end-tags, and
    #: empty-elt tags in an XML file.  This regexp is more lenient than
    #: the XML spec -- e.g., it allows spaces in some places where the
    #: spec does not.
    _XML_PIECE = re.compile(
        r"""
        # Include these so we can skip them:
        (?P<COMMENT>        <!--.*?-->                          )|
        (?P<CDATA>          <![CDATA[.*?]]>                     )|
        (?P<PI>             <\?.*?\?>                           )|
        (?P<DOCTYPE>        <!DOCTYPE\s+[^\[^>]*(\[[^\]]*])?\s*>)|
        # These are the ones we actually care about:
        (?P<EMPTY_ELT_TAG>  <\s*[^>/\?!\s][^>]*/\s*>            )|
        (?P<START_TAG>      <\s*[^>/\?!\s][^>]*>                )|
        (?P<END_TAG>        <\s*/[^>/\?!\s][^>]*>               )""",
        re.DOTALL | re.VERBOSE,
    )

    def _read_xml_fragment(self, stream):
        """
        Read a string from the given stream that does not contain any
        un-closed tags.  In particular, this function first reads a
        block from the stream of size ``self._BLOCK_SIZE``.  It then
        checks if that block contains an un-closed tag.  If it does,
        then this function either backtracks to the last '<', or reads
        another block.
        """
        fragment = ""

        if isinstance(stream, SeekableUnicodeStreamReader):
            startpos = stream.tell()
        while True:
            # Read a block and add it to the fragment.
            xml_block = stream.read(self._BLOCK_SIZE)
            fragment += xml_block

            # Do we have a well-formed xml fragment?
            if self._VALID_XML_RE.match(fragment):
                return fragment

            # Do we have a fragment that will never be well-formed?
            if re.search("[<>]", fragment).group(0) == ">":
                pos = stream.tell() - (
                    len(fragment) - re.search("[<>]", fragment).end()
                )
                raise ValueError('Unexpected ">" near char %s' % pos)

            # End of file?
            if not xml_block:
                raise ValueError("Unexpected end of file: tag not closed")

            # If not, then we must be in the middle of a <..tag..>.
            # If appropriate, backtrack to the most recent '<'
            # character.
            last_open_bracket = fragment.rfind("<")
            if last_open_bracket > 0:
                if self._VALID_XML_RE.match(fragment[:last_open_bracket]):
                    if isinstance(stream, SeekableUnicodeStreamReader):
                        stream.seek(startpos)
                        stream.char_seek_forward(last_open_bracket)
                    else:
                        stream.seek(-(len(fragment) - last_open_bracket), 1)
                    return fragment[:last_open_bracket]

            # Otherwise, read another block. (i.e., return to the
            # top of the loop.)

    def read_block(self, stream, tagspec=None, elt_handler=None):
        """
        Read from ``stream`` until we find at least one element that
        matches ``tagspec``, and return the result of applying
        ``elt_handler`` to each element found.
        """
        if tagspec is None:
            tagspec = self._tagspec
        if elt_handler is None:
            elt_handler = self.handle_elt

        # Use a stack of strings to keep track of our context:
        context = list(self._tag_context.get(stream.tell()))
        assert context is not None  # check this -- could it ever happen?

        elts = []

        elt_start = None  # where does the elt start
        elt_depth = None  # what context depth
        elt_text = ""

        while elts == [] or elt_start is not None:
            if isinstance(stream, SeekableUnicodeStreamReader):
                startpos = stream.tell()
            xml_fragment = self._read_xml_fragment(stream)

            # End of file.
            if not xml_fragment:
                if elt_start is None:
                    break
                else:
                    raise ValueError("Unexpected end of file")

            # Process each <tag> in the xml fragment.
            for piece in self._XML_PIECE.finditer(xml_fragment):
                if self._DEBUG:
                    print("{:>25} {}".format("/".join(context)[-20:], piece.group()))

                if piece.group("START_TAG"):
                    name = self._XML_TAG_NAME.match(piece.group()).group(1)
                    # Keep context up-to-date.
                    context.append(name)
                    # Is this one of the elts we're looking for?
                    if elt_start is None:
                        if re.match(tagspec, "/".join(context)):
                            elt_start = piece.start()
                            elt_depth = len(context)

                elif piece.group("END_TAG"):
                    name = self._XML_TAG_NAME.match(piece.group()).group(1)
                    # sanity checks:
                    if not context:
                        raise ValueError("Unmatched tag </%s>" % name)
                    if name != context[-1]:
                        raise ValueError(f"Unmatched tag <{context[-1]}>...</{name}>")
                    # Is this the end of an element?
                    if elt_start is not None and elt_depth == len(context):
                        elt_text += xml_fragment[elt_start : piece.end()]
                        elts.append((elt_text, "/".join(context)))
                        elt_start = elt_depth = None
                        elt_text = ""
                    # Keep context up-to-date
                    context.pop()

                elif piece.group("EMPTY_ELT_TAG"):
                    name = self._XML_TAG_NAME.match(piece.group()).group(1)
                    if elt_start is None:
                        if re.match(tagspec, "/".join(context) + "/" + name):
                            elts.append((piece.group(), "/".join(context) + "/" + name))

            if elt_start is not None:
                # If we haven't found any elements yet, then keep
                # looping until we do.
                if elts == []:
                    elt_text += xml_fragment[elt_start:]
                    elt_start = 0

                # If we've found at least one element, then try
                # backtracking to the start of the element that we're
                # inside of.
                else:
                    # take back the last start-tag, and return what
                    # we've gotten so far (elts is non-empty).
                    if self._DEBUG:
                        print(" " * 36 + "(backtrack)")
                    if isinstance(stream, SeekableUnicodeStreamReader):
                        stream.seek(startpos)
                        stream.char_seek_forward(elt_start)
                    else:
                        stream.seek(-(len(xml_fragment) - elt_start), 1)
                    context = context[: elt_depth - 1]
                    elt_start = elt_depth = None
                    elt_text = ""

        # Update the _tag_context dict.
        pos = stream.tell()
        if pos in self._tag_context:
            assert tuple(context) == self._tag_context[pos]
        else:
            self._tag_context[pos] = tuple(context)

        return [
            elt_handler(
                ElementTree.fromstring(elt.encode("ascii", "xmlcharrefreplace")),
                context,
            )
            for (elt, context) in elts
        ]
