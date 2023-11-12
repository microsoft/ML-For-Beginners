# Natural Language Toolkit: Toolbox Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Greg Aumann <greg_aumann@sil.org>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Module for reading, writing and manipulating
Toolbox databases and settings files.
"""

import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder

from nltk.data import PathPointer, find


class StandardFormat:
    """
    Class for reading and processing standard format marker files and strings.
    """

    def __init__(self, filename=None, encoding=None):
        self._encoding = encoding
        if filename is not None:
            self.open(filename)

    def open(self, sfm_file):
        """
        Open a standard format marker file for sequential reading.

        :param sfm_file: name of the standard format marker input file
        :type sfm_file: str
        """
        if isinstance(sfm_file, PathPointer):
            self._file = sfm_file.open(self._encoding)
        else:
            self._file = codecs.open(sfm_file, "r", self._encoding)

    def open_string(self, s):
        """
        Open a standard format marker string for sequential reading.

        :param s: string to parse as a standard format marker input file
        :type s: str
        """
        self._file = StringIO(s)

    def raw_fields(self):
        """
        Return an iterator that returns the next field in a (marker, value)
        tuple. Linebreaks and trailing white space are preserved except
        for the final newline in each field.

        :rtype: iter(tuple(str, str))
        """
        join_string = "\n"
        line_regexp = r"^%s(?:\\(\S+)\s*)?(.*)$"
        # discard a BOM in the first line
        first_line_pat = re.compile(line_regexp % "(?:\xef\xbb\xbf)?")
        line_pat = re.compile(line_regexp % "")
        # need to get first line outside the loop for correct handling
        # of the first marker if it spans multiple lines
        file_iter = iter(self._file)
        # PEP 479, prevent RuntimeError when StopIteration is raised inside generator
        try:
            line = next(file_iter)
        except StopIteration:
            # no more data is available, terminate the generator
            return
        mobj = re.match(first_line_pat, line)
        mkr, line_value = mobj.groups()
        value_lines = [line_value]
        self.line_num = 0
        for line in file_iter:
            self.line_num += 1
            mobj = re.match(line_pat, line)
            line_mkr, line_value = mobj.groups()
            if line_mkr:
                yield (mkr, join_string.join(value_lines))
                mkr = line_mkr
                value_lines = [line_value]
            else:
                value_lines.append(line_value)
        self.line_num += 1
        yield (mkr, join_string.join(value_lines))

    def fields(
        self,
        strip=True,
        unwrap=True,
        encoding=None,
        errors="strict",
        unicode_fields=None,
    ):
        """
        Return an iterator that returns the next field in a ``(marker, value)``
        tuple, where ``marker`` and ``value`` are unicode strings if an ``encoding``
        was specified in the ``fields()`` method. Otherwise they are non-unicode strings.

        :param strip: strip trailing whitespace from the last line of each field
        :type strip: bool
        :param unwrap: Convert newlines in a field to spaces.
        :type unwrap: bool
        :param encoding: Name of an encoding to use. If it is specified then
            the ``fields()`` method returns unicode strings rather than non
            unicode strings.
        :type encoding: str or None
        :param errors: Error handling scheme for codec. Same as the ``decode()``
            builtin string method.
        :type errors: str
        :param unicode_fields: Set of marker names whose values are UTF-8 encoded.
            Ignored if encoding is None. If the whole file is UTF-8 encoded set
            ``encoding='utf8'`` and leave ``unicode_fields`` with its default
            value of None.
        :type unicode_fields: sequence
        :rtype: iter(tuple(str, str))
        """
        if encoding is None and unicode_fields is not None:
            raise ValueError("unicode_fields is set but not encoding.")
        unwrap_pat = re.compile(r"\n+")
        for mkr, val in self.raw_fields():
            if unwrap:
                val = unwrap_pat.sub(" ", val)
            if strip:
                val = val.rstrip()
            yield (mkr, val)

    def close(self):
        """Close a previously opened standard format marker file or string."""
        self._file.close()
        try:
            del self.line_num
        except AttributeError:
            pass


class ToolboxData(StandardFormat):
    def parse(self, grammar=None, **kwargs):
        if grammar:
            return self._chunk_parse(grammar=grammar, **kwargs)
        else:
            return self._record_parse(**kwargs)

    def _record_parse(self, key=None, **kwargs):
        r"""
        Returns an element tree structure corresponding to a toolbox data file with
        all markers at the same level.

        Thus the following Toolbox database::
            \_sh v3.0  400  Rotokas Dictionary
            \_DateStampHasFourDigitYear

            \lx kaa
            \ps V.A
            \ge gag
            \gp nek i pas

            \lx kaa
            \ps V.B
            \ge strangle
            \gp pasim nek

        after parsing will end up with the same structure (ignoring the extra
        whitespace) as the following XML fragment after being parsed by
        ElementTree::
            <toolbox_data>
                <header>
                    <_sh>v3.0  400  Rotokas Dictionary</_sh>
                    <_DateStampHasFourDigitYear/>
                </header>

                <record>
                    <lx>kaa</lx>
                    <ps>V.A</ps>
                    <ge>gag</ge>
                    <gp>nek i pas</gp>
                </record>

                <record>
                    <lx>kaa</lx>
                    <ps>V.B</ps>
                    <ge>strangle</ge>
                    <gp>pasim nek</gp>
                </record>
            </toolbox_data>

        :param key: Name of key marker at the start of each record. If set to
            None (the default value) the first marker that doesn't begin with
            an underscore is assumed to be the key.
        :type key: str
        :param kwargs: Keyword arguments passed to ``StandardFormat.fields()``
        :type kwargs: dict
        :rtype: ElementTree._ElementInterface
        :return: contents of toolbox data divided into header and records
        """
        builder = TreeBuilder()
        builder.start("toolbox_data", {})
        builder.start("header", {})
        in_records = False
        for mkr, value in self.fields(**kwargs):
            if key is None and not in_records and mkr[0] != "_":
                key = mkr
            if mkr == key:
                if in_records:
                    builder.end("record")
                else:
                    builder.end("header")
                    in_records = True
                builder.start("record", {})
            builder.start(mkr, {})
            builder.data(value)
            builder.end(mkr)
        if in_records:
            builder.end("record")
        else:
            builder.end("header")
        builder.end("toolbox_data")
        return builder.close()

    def _tree2etree(self, parent):
        from nltk.tree import Tree

        root = Element(parent.label())
        for child in parent:
            if isinstance(child, Tree):
                root.append(self._tree2etree(child))
            else:
                text, tag = child
                e = SubElement(root, tag)
                e.text = text
        return root

    def _chunk_parse(self, grammar=None, root_label="record", trace=0, **kwargs):
        """
        Returns an element tree structure corresponding to a toolbox data file
        parsed according to the chunk grammar.

        :type grammar: str
        :param grammar: Contains the chunking rules used to parse the
            database.  See ``chunk.RegExp`` for documentation.
        :type root_label: str
        :param root_label: The node value that should be used for the
            top node of the chunk structure.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            higher will generate verbose tracing output.
        :type kwargs: dict
        :param kwargs: Keyword arguments passed to ``toolbox.StandardFormat.fields()``
        :rtype: ElementTree._ElementInterface
        """
        from nltk import chunk
        from nltk.tree import Tree

        cp = chunk.RegexpParser(grammar, root_label=root_label, trace=trace)
        db = self.parse(**kwargs)
        tb_etree = Element("toolbox_data")
        header = db.find("header")
        tb_etree.append(header)
        for record in db.findall("record"):
            parsed = cp.parse([(elem.text, elem.tag) for elem in record])
            tb_etree.append(self._tree2etree(parsed))
        return tb_etree


_is_value = re.compile(r"\S")


def to_sfm_string(tree, encoding=None, errors="strict", unicode_fields=None):
    """
    Return a string with a standard format representation of the toolbox
    data in tree (tree can be a toolbox database or a single record).

    :param tree: flat representation of toolbox data (whole database or single record)
    :type tree: ElementTree._ElementInterface
    :param encoding: Name of an encoding to use.
    :type encoding: str
    :param errors: Error handling scheme for codec. Same as the ``encode()``
        builtin string method.
    :type errors: str
    :param unicode_fields:
    :type unicode_fields: dict(str) or set(str)
    :rtype: str
    """
    if tree.tag == "record":
        root = Element("toolbox_data")
        root.append(tree)
        tree = root

    if tree.tag != "toolbox_data":
        raise ValueError("not a toolbox_data element structure")
    if encoding is None and unicode_fields is not None:
        raise ValueError(
            "if encoding is not specified then neither should unicode_fields"
        )
    l = []
    for rec in tree:
        l.append("\n")
        for field in rec:
            mkr = field.tag
            value = field.text
            if encoding is not None:
                if unicode_fields is not None and mkr in unicode_fields:
                    cur_encoding = "utf8"
                else:
                    cur_encoding = encoding
                if re.search(_is_value, value):
                    l.append((f"\\{mkr} {value}\n").encode(cur_encoding, errors))
                else:
                    l.append((f"\\{mkr}{value}\n").encode(cur_encoding, errors))
            else:
                if re.search(_is_value, value):
                    l.append(f"\\{mkr} {value}\n")
                else:
                    l.append(f"\\{mkr}{value}\n")
    return "".join(l[1:])


class ToolboxSettings(StandardFormat):
    """This class is the base class for settings files."""

    def __init__(self):
        super().__init__()

    def parse(self, encoding=None, errors="strict", **kwargs):
        """
        Return the contents of toolbox settings file with a nested structure.

        :param encoding: encoding used by settings file
        :type encoding: str
        :param errors: Error handling scheme for codec. Same as ``decode()`` builtin method.
        :type errors: str
        :param kwargs: Keyword arguments passed to ``StandardFormat.fields()``
        :type kwargs: dict
        :rtype: ElementTree._ElementInterface
        """
        builder = TreeBuilder()
        for mkr, value in self.fields(encoding=encoding, errors=errors, **kwargs):
            # Check whether the first char of the field marker
            # indicates a block start (+) or end (-)
            block = mkr[0]
            if block in ("+", "-"):
                mkr = mkr[1:]
            else:
                block = None
            # Build tree on the basis of block char
            if block == "+":
                builder.start(mkr, {})
                builder.data(value)
            elif block == "-":
                builder.end(mkr)
            else:
                builder.start(mkr, {})
                builder.data(value)
                builder.end(mkr)
        return builder.close()


def to_settings_string(tree, encoding=None, errors="strict", unicode_fields=None):
    # write XML to file
    l = list()
    _to_settings_string(
        tree.getroot(),
        l,
        encoding=encoding,
        errors=errors,
        unicode_fields=unicode_fields,
    )
    return "".join(l)


def _to_settings_string(node, l, **kwargs):
    # write XML to file
    tag = node.tag
    text = node.text
    if len(node) == 0:
        if text:
            l.append(f"\\{tag} {text}\n")
        else:
            l.append("\\%s\n" % tag)
    else:
        if text:
            l.append(f"\\+{tag} {text}\n")
        else:
            l.append("\\+%s\n" % tag)
        for n in node:
            _to_settings_string(n, l, **kwargs)
        l.append("\\-%s\n" % tag)
    return


def remove_blanks(elem):
    """
    Remove all elements and subelements with no text and no child elements.

    :param elem: toolbox data in an elementtree structure
    :type elem: ElementTree._ElementInterface
    """
    out = list()
    for child in elem:
        remove_blanks(child)
        if child.text or len(child) > 0:
            out.append(child)
    elem[:] = out


def add_default_fields(elem, default_fields):
    """
    Add blank elements and subelements specified in default_fields.

    :param elem: toolbox data in an elementtree structure
    :type elem: ElementTree._ElementInterface
    :param default_fields: fields to add to each type of element and subelement
    :type default_fields: dict(tuple)
    """
    for field in default_fields.get(elem.tag, []):
        if elem.find(field) is None:
            SubElement(elem, field)
    for child in elem:
        add_default_fields(child, default_fields)


def sort_fields(elem, field_orders):
    """
    Sort the elements and subelements in order specified in field_orders.

    :param elem: toolbox data in an elementtree structure
    :type elem: ElementTree._ElementInterface
    :param field_orders: order of fields for each type of element and subelement
    :type field_orders: dict(tuple)
    """
    order_dicts = dict()
    for field, order in field_orders.items():
        order_dicts[field] = order_key = dict()
        for i, subfield in enumerate(order):
            order_key[subfield] = i
    _sort_fields(elem, order_dicts)


def _sort_fields(elem, orders_dicts):
    """sort the children of elem"""
    try:
        order = orders_dicts[elem.tag]
    except KeyError:
        pass
    else:
        tmp = sorted(
            ((order.get(child.tag, 1e9), i), child) for i, child in enumerate(elem)
        )
        elem[:] = [child for key, child in tmp]
    for child in elem:
        if len(child):
            _sort_fields(child, orders_dicts)


def add_blank_lines(tree, blanks_before, blanks_between):
    """
    Add blank lines before all elements and subelements specified in blank_before.

    :param elem: toolbox data in an elementtree structure
    :type elem: ElementTree._ElementInterface
    :param blank_before: elements and subelements to add blank lines before
    :type blank_before: dict(tuple)
    """
    try:
        before = blanks_before[tree.tag]
        between = blanks_between[tree.tag]
    except KeyError:
        for elem in tree:
            if len(elem):
                add_blank_lines(elem, blanks_before, blanks_between)
    else:
        last_elem = None
        for elem in tree:
            tag = elem.tag
            if last_elem is not None and last_elem.tag != tag:
                if tag in before and last_elem is not None:
                    e = last_elem.getiterator()[-1]
                    e.text = (e.text or "") + "\n"
            else:
                if tag in between:
                    e = last_elem.getiterator()[-1]
                    e.text = (e.text or "") + "\n"
            if len(elem):
                add_blank_lines(elem, blanks_before, blanks_between)
            last_elem = elem


def demo():
    from itertools import islice

    #    zip_path = find('corpora/toolbox.zip')
    #    lexicon = ToolboxData(ZipFilePathPointer(zip_path, 'toolbox/rotokas.dic')).parse()
    file_path = find("corpora/toolbox/rotokas.dic")
    lexicon = ToolboxData(file_path).parse()
    print("first field in fourth record:")
    print(lexicon[3][0].tag)
    print(lexicon[3][0].text)

    print("\nfields in sequential order:")
    for field in islice(lexicon.find("record"), 10):
        print(field.tag, field.text)

    print("\nlx fields:")
    for field in islice(lexicon.findall("record/lx"), 10):
        print(field.text)

    settings = ToolboxSettings()
    file_path = find("corpora/toolbox/MDF/MDF_AltH.typ")
    settings.open(file_path)
    #    settings.open(ZipFilePathPointer(zip_path, entry='toolbox/MDF/MDF_AltH.typ'))
    tree = settings.parse(unwrap=False, encoding="cp1252")
    print(tree.find("expset/expMDF/rtfPageSetup/paperSize").text)
    settings_tree = ElementTree(tree)
    print(to_settings_string(settings_tree).encode("utf8"))


if __name__ == "__main__":
    demo()
