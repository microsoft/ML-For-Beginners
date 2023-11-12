# Natural Language Toolkit: Verbnet Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
An NLTK interface to the VerbNet verb lexicon

For details about VerbNet see:
https://verbs.colorado.edu/~mpalmer/projects/verbnet.html
"""

import re
import textwrap
from collections import defaultdict

from nltk.corpus.reader.xmldocs import XMLCorpusReader


class VerbnetCorpusReader(XMLCorpusReader):
    """
    An NLTK interface to the VerbNet verb lexicon.

    From the VerbNet site: "VerbNet (VN) (Kipper-Schuler 2006) is the largest
    on-line verb lexicon currently available for English. It is a hierarchical
    domain-independent, broad-coverage verb lexicon with mappings to other
    lexical resources such as WordNet (Miller, 1990; Fellbaum, 1998), XTAG
    (XTAG Research Group, 2001), and FrameNet (Baker et al., 1998)."

    For details about VerbNet see:
    https://verbs.colorado.edu/~mpalmer/projects/verbnet.html
    """

    # No unicode encoding param, since the data files are all XML.
    def __init__(self, root, fileids, wrap_etree=False):
        XMLCorpusReader.__init__(self, root, fileids, wrap_etree)

        self._lemma_to_class = defaultdict(list)
        """A dictionary mapping from verb lemma strings to lists of
        VerbNet class identifiers."""

        self._wordnet_to_class = defaultdict(list)
        """A dictionary mapping from wordnet identifier strings to
        lists of VerbNet class identifiers."""

        self._class_to_fileid = {}
        """A dictionary mapping from class identifiers to
        corresponding file identifiers.  The keys of this dictionary
        provide a complete list of all classes and subclasses."""

        self._shortid_to_longid = {}

        # Initialize the dictionaries.  Use the quick (regexp-based)
        # method instead of the slow (xml-based) method, because it
        # runs 2-30 times faster.
        self._quick_index()

    _LONGID_RE = re.compile(r"([^\-\.]*)-([\d+.\-]+)$")
    """Regular expression that matches (and decomposes) longids"""

    _SHORTID_RE = re.compile(r"[\d+.\-]+$")
    """Regular expression that matches shortids"""

    _INDEX_RE = re.compile(
        r'<MEMBER name="\??([^"]+)" wn="([^"]*)"[^>]+>|' r'<VNSUBCLASS ID="([^"]+)"/?>'
    )
    """Regular expression used by ``_index()`` to quickly scan the corpus
       for basic information."""

    def lemmas(self, vnclass=None):
        """
        Return a list of all verb lemmas that appear in any class, or
        in the ``classid`` if specified.
        """
        if vnclass is None:
            return sorted(self._lemma_to_class.keys())
        else:
            # [xx] should this include subclass members?
            if isinstance(vnclass, str):
                vnclass = self.vnclass(vnclass)
            return [member.get("name") for member in vnclass.findall("MEMBERS/MEMBER")]

    def wordnetids(self, vnclass=None):
        """
        Return a list of all wordnet identifiers that appear in any
        class, or in ``classid`` if specified.
        """
        if vnclass is None:
            return sorted(self._wordnet_to_class.keys())
        else:
            # [xx] should this include subclass members?
            if isinstance(vnclass, str):
                vnclass = self.vnclass(vnclass)
            return sum(
                (
                    member.get("wn", "").split()
                    for member in vnclass.findall("MEMBERS/MEMBER")
                ),
                [],
            )

    def classids(self, lemma=None, wordnetid=None, fileid=None, classid=None):
        """
        Return a list of the VerbNet class identifiers.  If a file
        identifier is specified, then return only the VerbNet class
        identifiers for classes (and subclasses) defined by that file.
        If a lemma is specified, then return only VerbNet class
        identifiers for classes that contain that lemma as a member.
        If a wordnetid is specified, then return only identifiers for
        classes that contain that wordnetid as a member.  If a classid
        is specified, then return only identifiers for subclasses of
        the specified VerbNet class.
        If nothing is specified, return all classids within VerbNet
        """
        if fileid is not None:
            return [c for (c, f) in self._class_to_fileid.items() if f == fileid]
        elif lemma is not None:
            return self._lemma_to_class[lemma]
        elif wordnetid is not None:
            return self._wordnet_to_class[wordnetid]
        elif classid is not None:
            xmltree = self.vnclass(classid)
            return [
                subclass.get("ID")
                for subclass in xmltree.findall("SUBCLASSES/VNSUBCLASS")
            ]
        else:
            return sorted(self._class_to_fileid.keys())

    def vnclass(self, fileid_or_classid):
        """Returns VerbNet class ElementTree

        Return an ElementTree containing the xml for the specified
        VerbNet class.

        :param fileid_or_classid: An identifier specifying which class
            should be returned.  Can be a file identifier (such as
            ``'put-9.1.xml'``), or a VerbNet class identifier (such as
            ``'put-9.1'``) or a short VerbNet class identifier (such as
            ``'9.1'``).
        """
        # File identifier: just return the xml.
        if fileid_or_classid in self._fileids:
            return self.xml(fileid_or_classid)

        # Class identifier: get the xml, and find the right elt.
        classid = self.longid(fileid_or_classid)
        if classid in self._class_to_fileid:
            fileid = self._class_to_fileid[self.longid(classid)]
            tree = self.xml(fileid)
            if classid == tree.get("ID"):
                return tree
            else:
                for subclass in tree.findall(".//VNSUBCLASS"):
                    if classid == subclass.get("ID"):
                        return subclass
                else:
                    assert False  # we saw it during _index()!

        else:
            raise ValueError(f"Unknown identifier {fileid_or_classid}")

    def fileids(self, vnclass_ids=None):
        """
        Return a list of fileids that make up this corpus.  If
        ``vnclass_ids`` is specified, then return the fileids that make
        up the specified VerbNet class(es).
        """
        if vnclass_ids is None:
            return self._fileids
        elif isinstance(vnclass_ids, str):
            return [self._class_to_fileid[self.longid(vnclass_ids)]]
        else:
            return [
                self._class_to_fileid[self.longid(vnclass_id)]
                for vnclass_id in vnclass_ids
            ]

    def frames(self, vnclass):
        """Given a VerbNet class, this method returns VerbNet frames

        The members returned are:
        1) Example
        2) Description
        3) Syntax
        4) Semantics

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        :return: frames - a list of frame dictionaries
        """
        if isinstance(vnclass, str):
            vnclass = self.vnclass(vnclass)
        frames = []
        vnframes = vnclass.findall("FRAMES/FRAME")
        for vnframe in vnframes:
            frames.append(
                {
                    "example": self._get_example_within_frame(vnframe),
                    "description": self._get_description_within_frame(vnframe),
                    "syntax": self._get_syntactic_list_within_frame(vnframe),
                    "semantics": self._get_semantics_within_frame(vnframe),
                }
            )
        return frames

    def subclasses(self, vnclass):
        """Returns subclass ids, if any exist

        Given a VerbNet class, this method returns subclass ids (if they exist)
        in a list of strings.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        :return: list of subclasses
        """
        if isinstance(vnclass, str):
            vnclass = self.vnclass(vnclass)

        subclasses = [
            subclass.get("ID") for subclass in vnclass.findall("SUBCLASSES/VNSUBCLASS")
        ]
        return subclasses

    def themroles(self, vnclass):
        """Returns thematic roles participating in a VerbNet class

        Members returned as part of roles are-
        1) Type
        2) Modifiers

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        :return: themroles: A list of thematic roles in the VerbNet class
        """
        if isinstance(vnclass, str):
            vnclass = self.vnclass(vnclass)

        themroles = []
        for trole in vnclass.findall("THEMROLES/THEMROLE"):
            themroles.append(
                {
                    "type": trole.get("type"),
                    "modifiers": [
                        {"value": restr.get("Value"), "type": restr.get("type")}
                        for restr in trole.findall("SELRESTRS/SELRESTR")
                    ],
                }
            )
        return themroles

    ######################################################################
    # { Index Initialization
    ######################################################################

    def _index(self):
        """
        Initialize the indexes ``_lemma_to_class``,
        ``_wordnet_to_class``, and ``_class_to_fileid`` by scanning
        through the corpus fileids.  This is fast if ElementTree
        uses the C implementation (<0.1 secs), but quite slow (>10 secs)
        if only the python implementation is available.
        """
        for fileid in self._fileids:
            self._index_helper(self.xml(fileid), fileid)

    def _index_helper(self, xmltree, fileid):
        """Helper for ``_index()``"""
        vnclass = xmltree.get("ID")
        self._class_to_fileid[vnclass] = fileid
        self._shortid_to_longid[self.shortid(vnclass)] = vnclass
        for member in xmltree.findall("MEMBERS/MEMBER"):
            self._lemma_to_class[member.get("name")].append(vnclass)
            for wn in member.get("wn", "").split():
                self._wordnet_to_class[wn].append(vnclass)
        for subclass in xmltree.findall("SUBCLASSES/VNSUBCLASS"):
            self._index_helper(subclass, fileid)

    def _quick_index(self):
        """
        Initialize the indexes ``_lemma_to_class``,
        ``_wordnet_to_class``, and ``_class_to_fileid`` by scanning
        through the corpus fileids.  This doesn't do proper xml parsing,
        but is good enough to find everything in the standard VerbNet
        corpus -- and it runs about 30 times faster than xml parsing
        (with the python ElementTree; only 2-3 times faster
        if ElementTree uses the C implementation).
        """
        # nb: if we got rid of wordnet_to_class, this would run 2-3
        # times faster.
        for fileid in self._fileids:
            vnclass = fileid[:-4]  # strip the '.xml'
            self._class_to_fileid[vnclass] = fileid
            self._shortid_to_longid[self.shortid(vnclass)] = vnclass
            with self.open(fileid) as fp:
                for m in self._INDEX_RE.finditer(fp.read()):
                    groups = m.groups()
                    if groups[0] is not None:
                        self._lemma_to_class[groups[0]].append(vnclass)
                        for wn in groups[1].split():
                            self._wordnet_to_class[wn].append(vnclass)
                    elif groups[2] is not None:
                        self._class_to_fileid[groups[2]] = fileid
                        vnclass = groups[2]  # for <MEMBER> elts.
                        self._shortid_to_longid[self.shortid(vnclass)] = vnclass
                    else:
                        assert False, "unexpected match condition"

    ######################################################################
    # { Identifier conversion
    ######################################################################

    def longid(self, shortid):
        """Returns longid of a VerbNet class

        Given a short VerbNet class identifier (eg '37.10'), map it
        to a long id (eg 'confess-37.10').  If ``shortid`` is already a
        long id, then return it as-is"""
        if self._LONGID_RE.match(shortid):
            return shortid  # it's already a longid.
        elif not self._SHORTID_RE.match(shortid):
            raise ValueError("vnclass identifier %r not found" % shortid)
        try:
            return self._shortid_to_longid[shortid]
        except KeyError as e:
            raise ValueError("vnclass identifier %r not found" % shortid) from e

    def shortid(self, longid):
        """Returns shortid of a VerbNet class

        Given a long VerbNet class identifier (eg 'confess-37.10'),
        map it to a short id (eg '37.10').  If ``longid`` is already a
        short id, then return it as-is."""
        if self._SHORTID_RE.match(longid):
            return longid  # it's already a shortid.
        m = self._LONGID_RE.match(longid)
        if m:
            return m.group(2)
        else:
            raise ValueError("vnclass identifier %r not found" % longid)

    ######################################################################
    # { Frame access utility functions
    ######################################################################

    def _get_semantics_within_frame(self, vnframe):
        """Returns semantics within a single frame

        A utility function to retrieve semantics within a frame in VerbNet
        Members of the semantics dictionary:
        1) Predicate value
        2) Arguments

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        :return: semantics: semantics dictionary
        """
        semantics_within_single_frame = []
        for pred in vnframe.findall("SEMANTICS/PRED"):
            arguments = [
                {"type": arg.get("type"), "value": arg.get("value")}
                for arg in pred.findall("ARGS/ARG")
            ]
            semantics_within_single_frame.append(
                {
                    "predicate_value": pred.get("value"),
                    "arguments": arguments,
                    "negated": pred.get("bool") == "!",
                }
            )
        return semantics_within_single_frame

    def _get_example_within_frame(self, vnframe):
        """Returns example within a frame

        A utility function to retrieve an example within a frame in VerbNet.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        :return: example_text: The example sentence for this particular frame
        """
        example_element = vnframe.find("EXAMPLES/EXAMPLE")
        if example_element is not None:
            example_text = example_element.text
        else:
            example_text = ""
        return example_text

    def _get_description_within_frame(self, vnframe):
        """Returns member description within frame

        A utility function to retrieve a description of participating members
        within a frame in VerbNet.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        :return: description: a description dictionary with members - primary and secondary
        """
        description_element = vnframe.find("DESCRIPTION")
        return {
            "primary": description_element.attrib["primary"],
            "secondary": description_element.get("secondary", ""),
        }

    def _get_syntactic_list_within_frame(self, vnframe):
        """Returns semantics within a frame

        A utility function to retrieve semantics within a frame in VerbNet.
        Members of the syntactic dictionary:
        1) POS Tag
        2) Modifiers

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        :return: syntax_within_single_frame
        """
        syntax_within_single_frame = []
        for elt in vnframe.find("SYNTAX"):
            pos_tag = elt.tag
            modifiers = dict()
            modifiers["value"] = elt.get("value") if "value" in elt.attrib else ""
            modifiers["selrestrs"] = [
                {"value": restr.get("Value"), "type": restr.get("type")}
                for restr in elt.findall("SELRESTRS/SELRESTR")
            ]
            modifiers["synrestrs"] = [
                {"value": restr.get("Value"), "type": restr.get("type")}
                for restr in elt.findall("SYNRESTRS/SYNRESTR")
            ]
            syntax_within_single_frame.append(
                {"pos_tag": pos_tag, "modifiers": modifiers}
            )
        return syntax_within_single_frame

    ######################################################################
    # { Pretty Printing
    ######################################################################

    def pprint(self, vnclass):
        """Returns pretty printed version of a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet class.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        """
        if isinstance(vnclass, str):
            vnclass = self.vnclass(vnclass)

        s = vnclass.get("ID") + "\n"
        s += self.pprint_subclasses(vnclass, indent="  ") + "\n"
        s += self.pprint_members(vnclass, indent="  ") + "\n"
        s += "  Thematic roles:\n"
        s += self.pprint_themroles(vnclass, indent="    ") + "\n"
        s += "  Frames:\n"
        s += self.pprint_frames(vnclass, indent="    ")
        return s

    def pprint_subclasses(self, vnclass, indent=""):
        """Returns pretty printed version of subclasses of VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet class's subclasses.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        """
        if isinstance(vnclass, str):
            vnclass = self.vnclass(vnclass)

        subclasses = self.subclasses(vnclass)
        if not subclasses:
            subclasses = ["(none)"]
        s = "Subclasses: " + " ".join(subclasses)
        return textwrap.fill(
            s, 70, initial_indent=indent, subsequent_indent=indent + "  "
        )

    def pprint_members(self, vnclass, indent=""):
        """Returns pretty printed version of members in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet class's member verbs.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        """
        if isinstance(vnclass, str):
            vnclass = self.vnclass(vnclass)

        members = self.lemmas(vnclass)
        if not members:
            members = ["(none)"]
        s = "Members: " + " ".join(members)
        return textwrap.fill(
            s, 70, initial_indent=indent, subsequent_indent=indent + "  "
        )

    def pprint_themroles(self, vnclass, indent=""):
        """Returns pretty printed version of thematic roles in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet class's thematic roles.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        """
        if isinstance(vnclass, str):
            vnclass = self.vnclass(vnclass)

        pieces = []
        for themrole in self.themroles(vnclass):
            piece = indent + "* " + themrole.get("type")
            modifiers = [
                modifier["value"] + modifier["type"]
                for modifier in themrole["modifiers"]
            ]
            if modifiers:
                piece += "[{}]".format(" ".join(modifiers))
            pieces.append(piece)
        return "\n".join(pieces)

    def pprint_frames(self, vnclass, indent=""):
        """Returns pretty version of all frames in a VerbNet class

        Return a string containing a pretty-printed representation of
        the list of frames within the VerbNet class.

        :param vnclass: A VerbNet class identifier; or an ElementTree
            containing the xml contents of a VerbNet class.
        """
        if isinstance(vnclass, str):
            vnclass = self.vnclass(vnclass)
        pieces = []
        for vnframe in self.frames(vnclass):
            pieces.append(self._pprint_single_frame(vnframe, indent))
        return "\n".join(pieces)

    def _pprint_single_frame(self, vnframe, indent=""):
        """Returns pretty printed version of a single frame in a VerbNet class

        Returns a string containing a pretty-printed representation of
        the given frame.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        """
        frame_string = self._pprint_description_within_frame(vnframe, indent) + "\n"
        frame_string += self._pprint_example_within_frame(vnframe, indent + " ") + "\n"
        frame_string += (
            self._pprint_syntax_within_frame(vnframe, indent + "  Syntax: ") + "\n"
        )
        frame_string += indent + "  Semantics:\n"
        frame_string += self._pprint_semantics_within_frame(vnframe, indent + "    ")
        return frame_string

    def _pprint_example_within_frame(self, vnframe, indent=""):
        """Returns pretty printed version of example within frame in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet frame example.

        :param vnframe: An ElementTree containing the xml contents of
            a Verbnet frame.
        """
        if vnframe["example"]:
            return indent + " Example: " + vnframe["example"]

    def _pprint_description_within_frame(self, vnframe, indent=""):
        """Returns pretty printed version of a VerbNet frame description

        Return a string containing a pretty-printed representation of
        the given VerbNet frame description.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        """
        description = indent + vnframe["description"]["primary"]
        if vnframe["description"]["secondary"]:
            description += " ({})".format(vnframe["description"]["secondary"])
        return description

    def _pprint_syntax_within_frame(self, vnframe, indent=""):
        """Returns pretty printed version of syntax within a frame in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet frame syntax.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        """
        pieces = []
        for element in vnframe["syntax"]:
            piece = element["pos_tag"]
            modifier_list = []
            if "value" in element["modifiers"] and element["modifiers"]["value"]:
                modifier_list.append(element["modifiers"]["value"])
            modifier_list += [
                "{}{}".format(restr["value"], restr["type"])
                for restr in (
                    element["modifiers"]["selrestrs"]
                    + element["modifiers"]["synrestrs"]
                )
            ]
            if modifier_list:
                piece += "[{}]".format(" ".join(modifier_list))
            pieces.append(piece)

        return indent + " ".join(pieces)

    def _pprint_semantics_within_frame(self, vnframe, indent=""):
        """Returns a pretty printed version of semantics within frame in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet frame semantics.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        """
        pieces = []
        for predicate in vnframe["semantics"]:
            arguments = [argument["value"] for argument in predicate["arguments"]]
            pieces.append(
                f"{'¬' if predicate['negated'] else ''}{predicate['predicate_value']}({', '.join(arguments)})"
            )
        return "\n".join(f"{indent}* {piece}" for piece in pieces)
