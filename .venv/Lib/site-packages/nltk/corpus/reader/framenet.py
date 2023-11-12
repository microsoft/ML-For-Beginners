# Natural Language Toolkit: Framenet Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Chuck Wooters <wooters@icsi.berkeley.edu>,
#          Nathan Schneider <nathan.schneider@georgetown.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


"""
Corpus reader for the FrameNet 1.7 lexicon and corpus.
"""

import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint

from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap

__docformat__ = "epytext en"


def mimic_wrap(lines, wrap_at=65, **kwargs):
    """
    Wrap the first of 'lines' with textwrap and the remaining lines at exactly the same
    positions as the first.
    """
    l0 = textwrap.fill(lines[0], wrap_at, drop_whitespace=False).split("\n")
    yield l0

    def _(line):
        il0 = 0
        while line and il0 < len(l0) - 1:
            yield line[: len(l0[il0])]
            line = line[len(l0[il0]) :]
            il0 += 1
        if line:  # Remaining stuff on this line past the end of the mimicked line.
            # So just textwrap this line.
            yield from textwrap.fill(line, wrap_at, drop_whitespace=False).split("\n")

    for l in lines[1:]:
        yield list(_(l))


def _pretty_longstring(defstr, prefix="", wrap_at=65):

    """
    Helper function for pretty-printing a long string.

    :param defstr: The string to be printed.
    :type defstr: str
    :return: A nicely formatted string representation of the long string.
    :rtype: str
    """

    outstr = ""
    for line in textwrap.fill(defstr, wrap_at).split("\n"):
        outstr += prefix + line + "\n"
    return outstr


def _pretty_any(obj):

    """
    Helper function for pretty-printing any AttrDict object.

    :param obj: The obj to be printed.
    :type obj: AttrDict
    :return: A nicely formatted string representation of the AttrDict object.
    :rtype: str
    """

    outstr = ""
    for k in obj:
        if isinstance(obj[k], str) and len(obj[k]) > 65:
            outstr += f"[{k}]\n"
            outstr += "{}".format(_pretty_longstring(obj[k], prefix="  "))
            outstr += "\n"
        else:
            outstr += f"[{k}] {obj[k]}\n"

    return outstr


def _pretty_semtype(st):

    """
    Helper function for pretty-printing a semantic type.

    :param st: The semantic type to be printed.
    :type st: AttrDict
    :return: A nicely formatted string representation of the semantic type.
    :rtype: str
    """

    semkeys = st.keys()
    if len(semkeys) == 1:
        return "<None>"

    outstr = ""
    outstr += "semantic type ({0.ID}): {0.name}\n".format(st)
    if "abbrev" in semkeys:
        outstr += f"[abbrev] {st.abbrev}\n"
    if "definition" in semkeys:
        outstr += "[definition]\n"
        outstr += _pretty_longstring(st.definition, "  ")
    outstr += f"[rootType] {st.rootType.name}({st.rootType.ID})\n"
    if st.superType is None:
        outstr += "[superType] <None>\n"
    else:
        outstr += f"[superType] {st.superType.name}({st.superType.ID})\n"
    outstr += f"[subTypes] {len(st.subTypes)} subtypes\n"
    outstr += (
        "  "
        + ", ".join(f"{x.name}({x.ID})" for x in st.subTypes)
        + "\n" * (len(st.subTypes) > 0)
    )
    return outstr


def _pretty_frame_relation_type(freltyp):

    """
    Helper function for pretty-printing a frame relation type.

    :param freltyp: The frame relation type to be printed.
    :type freltyp: AttrDict
    :return: A nicely formatted string representation of the frame relation type.
    :rtype: str
    """
    outstr = "<frame relation type ({0.ID}): {0.superFrameName} -- {0.name} -> {0.subFrameName}>".format(
        freltyp
    )
    return outstr


def _pretty_frame_relation(frel):

    """
    Helper function for pretty-printing a frame relation.

    :param frel: The frame relation to be printed.
    :type frel: AttrDict
    :return: A nicely formatted string representation of the frame relation.
    :rtype: str
    """
    outstr = "<{0.type.superFrameName}={0.superFrameName} -- {0.type.name} -> {0.type.subFrameName}={0.subFrameName}>".format(
        frel
    )
    return outstr


def _pretty_fe_relation(ferel):

    """
    Helper function for pretty-printing an FE relation.

    :param ferel: The FE relation to be printed.
    :type ferel: AttrDict
    :return: A nicely formatted string representation of the FE relation.
    :rtype: str
    """
    outstr = "<{0.type.superFrameName}={0.frameRelation.superFrameName}.{0.superFEName} -- {0.type.name} -> {0.type.subFrameName}={0.frameRelation.subFrameName}.{0.subFEName}>".format(
        ferel
    )
    return outstr


def _pretty_lu(lu):

    """
    Helper function for pretty-printing a lexical unit.

    :param lu: The lu to be printed.
    :type lu: AttrDict
    :return: A nicely formatted string representation of the lexical unit.
    :rtype: str
    """

    lukeys = lu.keys()
    outstr = ""
    outstr += "lexical unit ({0.ID}): {0.name}\n\n".format(lu)
    if "definition" in lukeys:
        outstr += "[definition]\n"
        outstr += _pretty_longstring(lu.definition, "  ")
    if "frame" in lukeys:
        outstr += f"\n[frame] {lu.frame.name}({lu.frame.ID})\n"
    if "incorporatedFE" in lukeys:
        outstr += f"\n[incorporatedFE] {lu.incorporatedFE}\n"
    if "POS" in lukeys:
        outstr += f"\n[POS] {lu.POS}\n"
    if "status" in lukeys:
        outstr += f"\n[status] {lu.status}\n"
    if "totalAnnotated" in lukeys:
        outstr += f"\n[totalAnnotated] {lu.totalAnnotated} annotated examples\n"
    if "lexemes" in lukeys:
        outstr += "\n[lexemes] {}\n".format(
            " ".join(f"{lex.name}/{lex.POS}" for lex in lu.lexemes)
        )
    if "semTypes" in lukeys:
        outstr += f"\n[semTypes] {len(lu.semTypes)} semantic types\n"
        outstr += (
            "  " * (len(lu.semTypes) > 0)
            + ", ".join(f"{x.name}({x.ID})" for x in lu.semTypes)
            + "\n" * (len(lu.semTypes) > 0)
        )
    if "URL" in lukeys:
        outstr += f"\n[URL] {lu.URL}\n"
    if "subCorpus" in lukeys:
        subc = [x.name for x in lu.subCorpus]
        outstr += f"\n[subCorpus] {len(lu.subCorpus)} subcorpora\n"
        for line in textwrap.fill(", ".join(sorted(subc)), 60).split("\n"):
            outstr += f"  {line}\n"
    if "exemplars" in lukeys:
        outstr += "\n[exemplars] {} sentences across all subcorpora\n".format(
            len(lu.exemplars)
        )

    return outstr


def _pretty_exemplars(exemplars, lu):
    """
    Helper function for pretty-printing a list of exemplar sentences for a lexical unit.

    :param sent: The list of exemplar sentences to be printed.
    :type sent: list(AttrDict)
    :return: An index of the text of the exemplar sentences.
    :rtype: str
    """

    outstr = ""
    outstr += "exemplar sentences for {0.name} in {0.frame.name}:\n\n".format(lu)
    for i, sent in enumerate(exemplars):
        outstr += f"[{i}] {sent.text}\n"
    outstr += "\n"
    return outstr


def _pretty_fulltext_sentences(sents):
    """
    Helper function for pretty-printing a list of annotated sentences for a full-text document.

    :param sent: The list of sentences to be printed.
    :type sent: list(AttrDict)
    :return: An index of the text of the sentences.
    :rtype: str
    """

    outstr = ""
    outstr += "full-text document ({0.ID}) {0.name}:\n\n".format(sents)
    outstr += "[corpid] {0.corpid}\n[corpname] {0.corpname}\n[description] {0.description}\n[URL] {0.URL}\n\n".format(
        sents
    )
    outstr += f"[sentence]\n"
    for i, sent in enumerate(sents.sentence):
        outstr += f"[{i}] {sent.text}\n"
    outstr += "\n"
    return outstr


def _pretty_fulltext_sentence(sent):
    """
    Helper function for pretty-printing an annotated sentence from a full-text document.

    :param sent: The sentence to be printed.
    :type sent: list(AttrDict)
    :return: The text of the sentence with annotation set indices on frame targets.
    :rtype: str
    """

    outstr = ""
    outstr += "full-text sentence ({0.ID}) in {1}:\n\n".format(
        sent, sent.doc.get("name", sent.doc.description)
    )
    outstr += f"\n[POS] {len(sent.POS)} tags\n"
    outstr += f"\n[POS_tagset] {sent.POS_tagset}\n\n"
    outstr += "[text] + [annotationSet]\n\n"
    outstr += sent._ascii()  # -> _annotation_ascii()
    outstr += "\n"
    return outstr


def _pretty_pos(aset):
    """
    Helper function for pretty-printing a sentence with its POS tags.

    :param aset: The POS annotation set of the sentence to be printed.
    :type sent: list(AttrDict)
    :return: The text of the sentence and its POS tags.
    :rtype: str
    """

    outstr = ""
    outstr += "POS annotation set ({0.ID}) {0.POS_tagset} in sentence {0.sent.ID}:\n\n".format(
        aset
    )

    # list the target spans and their associated aset index
    overt = sorted(aset.POS)

    sent = aset.sent
    s0 = sent.text
    s1 = ""
    s2 = ""
    i = 0
    adjust = 0
    for j, k, lbl in overt:
        assert j >= i, ("Overlapping targets?", (j, k, lbl))
        s1 += " " * (j - i) + "-" * (k - j)
        if len(lbl) > (k - j):
            # add space in the sentence to make room for the annotation index
            amt = len(lbl) - (k - j)
            s0 = (
                s0[: k + adjust] + "~" * amt + s0[k + adjust :]
            )  # '~' to prevent line wrapping
            s1 = s1[: k + adjust] + " " * amt + s1[k + adjust :]
            adjust += amt
        s2 += " " * (j - i) + lbl.ljust(k - j)
        i = k

    long_lines = [s0, s1, s2]

    outstr += "\n\n".join(
        map("\n".join, zip_longest(*mimic_wrap(long_lines), fillvalue=" "))
    ).replace("~", " ")
    outstr += "\n"
    return outstr


def _pretty_annotation(sent, aset_level=False):
    """
    Helper function for pretty-printing an exemplar sentence for a lexical unit.

    :param sent: An annotation set or exemplar sentence to be printed.
    :param aset_level: If True, 'sent' is actually an annotation set within a sentence.
    :type sent: AttrDict
    :return: A nicely formatted string representation of the exemplar sentence
    with its target, frame, and FE annotations.
    :rtype: str
    """

    sentkeys = sent.keys()
    outstr = "annotation set" if aset_level else "exemplar sentence"
    outstr += f" ({sent.ID}):\n"
    if aset_level:  # TODO: any UNANN exemplars?
        outstr += f"\n[status] {sent.status}\n"
    for k in ("corpID", "docID", "paragNo", "sentNo", "aPos"):
        if k in sentkeys:
            outstr += f"[{k}] {sent[k]}\n"
    outstr += (
        "\n[LU] ({0.ID}) {0.name} in {0.frame.name}\n".format(sent.LU)
        if sent.LU
        else "\n[LU] Not found!"
    )
    outstr += "\n[frame] ({0.ID}) {0.name}\n".format(
        sent.frame
    )  # redundant with above, but .frame is convenient
    if not aset_level:
        outstr += "\n[annotationSet] {} annotation sets\n".format(
            len(sent.annotationSet)
        )
        outstr += f"\n[POS] {len(sent.POS)} tags\n"
        outstr += f"\n[POS_tagset] {sent.POS_tagset}\n"
    outstr += "\n[GF] {} relation{}\n".format(
        len(sent.GF), "s" if len(sent.GF) != 1 else ""
    )
    outstr += "\n[PT] {} phrase{}\n".format(
        len(sent.PT), "s" if len(sent.PT) != 1 else ""
    )
    """
    Special Layers
    --------------

    The 'NER' layer contains, for some of the data, named entity labels.

    The 'WSL' (word status layer) contains, for some of the data,
    spans which should not in principle be considered targets (NT).

    The 'Other' layer records relative clause constructions (Rel=relativizer, Ant=antecedent),
    pleonastic 'it' (Null), and existential 'there' (Exist).
    On occasion they are duplicated by accident (e.g., annotationSet 1467275 in lu6700.xml).

    The 'Sent' layer appears to contain labels that the annotator has flagged the
    sentence with for their convenience: values include
    'sense1', 'sense2', 'sense3', etc.;
    'Blend', 'Canonical', 'Idiom', 'Metaphor', 'Special-Sent',
    'keepS', 'deleteS', 'reexamine'
    (sometimes they are duplicated for no apparent reason).

    The POS-specific layers may contain the following kinds of spans:
    Asp (aspectual particle), Non-Asp (non-aspectual particle),
    Cop (copula), Supp (support), Ctrlr (controller),
    Gov (governor), X. Gov and X always cooccur.

    >>> from nltk.corpus import framenet as fn
    >>> def f(luRE, lyr, ignore=set()):
    ...   for i,ex in enumerate(fn.exemplars(luRE)):
    ...     if lyr in ex and ex[lyr] and set(zip(*ex[lyr])[2]) - ignore:
    ...       print(i,ex[lyr])

    - Verb: Asp, Non-Asp
    - Noun: Cop, Supp, Ctrlr, Gov, X
    - Adj: Cop, Supp, Ctrlr, Gov, X
    - Prep: Cop, Supp, Ctrlr
    - Adv: Ctrlr
    - Scon: (none)
    - Art: (none)
    """
    for lyr in ("NER", "WSL", "Other", "Sent"):
        if lyr in sent and sent[lyr]:
            outstr += "\n[{}] {} entr{}\n".format(
                lyr, len(sent[lyr]), "ies" if len(sent[lyr]) != 1 else "y"
            )
    outstr += "\n[text] + [Target] + [FE]"
    # POS-specific layers: syntactically important words that are neither the target
    # nor the FEs. Include these along with the first FE layer but with '^' underlining.
    for lyr in ("Verb", "Noun", "Adj", "Adv", "Prep", "Scon", "Art"):
        if lyr in sent and sent[lyr]:
            outstr += f" + [{lyr}]"
    if "FE2" in sentkeys:
        outstr += " + [FE2]"
        if "FE3" in sentkeys:
            outstr += " + [FE3]"
    outstr += "\n\n"
    outstr += sent._ascii()  # -> _annotation_ascii()
    outstr += "\n"

    return outstr


def _annotation_ascii(sent):
    """
    Given a sentence or FE annotation set, construct the width-limited string showing
    an ASCII visualization of the sentence's annotations, calling either
    _annotation_ascii_frames() or _annotation_ascii_FEs() as appropriate.
    This will be attached as a method to appropriate AttrDict instances
    and called in the full pretty-printing of the instance.
    """
    if sent._type == "fulltext_sentence" or (
        "annotationSet" in sent and len(sent.annotationSet) > 2
    ):
        # a full-text sentence OR sentence with multiple targets.
        # (multiple targets = >2 annotation sets, because the first annotation set is POS.)
        return _annotation_ascii_frames(sent)
    else:  # an FE annotation set, or an LU sentence with 1 target
        return _annotation_ascii_FEs(sent)


def _annotation_ascii_frames(sent):
    """
    ASCII string rendering of the sentence along with its targets and frame names.
    Called for all full-text sentences, as well as the few LU sentences with multiple
    targets (e.g., fn.lu(6412).exemplars[82] has two want.v targets).
    Line-wrapped to limit the display width.
    """
    # list the target spans and their associated aset index
    overt = []
    for a, aset in enumerate(sent.annotationSet[1:]):
        for j, k in aset.Target:
            indexS = f"[{a + 1}]"
            if aset.status == "UNANN" or aset.LU.status == "Problem":
                indexS += " "
                if aset.status == "UNANN":
                    indexS += "!"  # warning indicator that there is a frame annotation but no FE annotation
                if aset.LU.status == "Problem":
                    indexS += "?"  # warning indicator that there is a missing LU definition (because the LU has Problem status)
            overt.append((j, k, aset.LU.frame.name, indexS))
    overt = sorted(overt)

    duplicates = set()
    for o, (j, k, fname, asetIndex) in enumerate(overt):
        if o > 0 and j <= overt[o - 1][1]:
            # multiple annotation sets on the same target
            # (e.g. due to a coordination construction or multiple annotators)
            if (
                overt[o - 1][:2] == (j, k) and overt[o - 1][2] == fname
            ):  # same target, same frame
                # splice indices together
                combinedIndex = (
                    overt[o - 1][3] + asetIndex
                )  # e.g., '[1][2]', '[1]! [2]'
                combinedIndex = combinedIndex.replace(" !", "! ").replace(" ?", "? ")
                overt[o - 1] = overt[o - 1][:3] + (combinedIndex,)
                duplicates.add(o)
            else:  # different frames, same or overlapping targets
                s = sent.text
                for j, k, fname, asetIndex in overt:
                    s += "\n" + asetIndex + " " + sent.text[j:k] + " :: " + fname
                s += "\n(Unable to display sentence with targets marked inline due to overlap)"
                return s
    for o in reversed(sorted(duplicates)):
        del overt[o]

    s0 = sent.text
    s1 = ""
    s11 = ""
    s2 = ""
    i = 0
    adjust = 0
    fAbbrevs = OrderedDict()
    for j, k, fname, asetIndex in overt:
        if not j >= i:
            assert j >= i, (
                "Overlapping targets?"
                + (
                    " UNANN"
                    if any(aset.status == "UNANN" for aset in sent.annotationSet[1:])
                    else ""
                ),
                (j, k, asetIndex),
            )
        s1 += " " * (j - i) + "*" * (k - j)
        short = fname[: k - j]
        if (k - j) < len(fname):
            r = 0
            while short in fAbbrevs:
                if fAbbrevs[short] == fname:
                    break
                r += 1
                short = fname[: k - j - 1] + str(r)
            else:  # short not in fAbbrevs
                fAbbrevs[short] = fname
        s11 += " " * (j - i) + short.ljust(k - j)
        if len(asetIndex) > (k - j):
            # add space in the sentence to make room for the annotation index
            amt = len(asetIndex) - (k - j)
            s0 = (
                s0[: k + adjust] + "~" * amt + s0[k + adjust :]
            )  # '~' to prevent line wrapping
            s1 = s1[: k + adjust] + " " * amt + s1[k + adjust :]
            s11 = s11[: k + adjust] + " " * amt + s11[k + adjust :]
            adjust += amt
        s2 += " " * (j - i) + asetIndex.ljust(k - j)
        i = k

    long_lines = [s0, s1, s11, s2]

    outstr = "\n\n".join(
        map("\n".join, zip_longest(*mimic_wrap(long_lines), fillvalue=" "))
    ).replace("~", " ")
    outstr += "\n"
    if fAbbrevs:
        outstr += " (" + ", ".join("=".join(pair) for pair in fAbbrevs.items()) + ")"
        assert len(fAbbrevs) == len(dict(fAbbrevs)), "Abbreviation clash"

    return outstr


def _annotation_ascii_FE_layer(overt, ni, feAbbrevs):
    """Helper for _annotation_ascii_FEs()."""
    s1 = ""
    s2 = ""
    i = 0
    for j, k, fename in overt:
        s1 += " " * (j - i) + ("^" if fename.islower() else "-") * (k - j)
        short = fename[: k - j]
        if len(fename) > len(short):
            r = 0
            while short in feAbbrevs:
                if feAbbrevs[short] == fename:
                    break
                r += 1
                short = fename[: k - j - 1] + str(r)
            else:  # short not in feAbbrevs
                feAbbrevs[short] = fename
        s2 += " " * (j - i) + short.ljust(k - j)
        i = k

    sNI = ""
    if ni:
        sNI += " [" + ", ".join(":".join(x) for x in sorted(ni.items())) + "]"
    return [s1, s2, sNI]


def _annotation_ascii_FEs(sent):
    """
    ASCII string rendering of the sentence along with a single target and its FEs.
    Secondary and tertiary FE layers are included if present.
    'sent' can be an FE annotation set or an LU sentence with a single target.
    Line-wrapped to limit the display width.
    """
    feAbbrevs = OrderedDict()
    posspec = []  # POS-specific layer spans (e.g., Supp[ort], Cop[ula])
    posspec_separate = False
    for lyr in ("Verb", "Noun", "Adj", "Adv", "Prep", "Scon", "Art"):
        if lyr in sent and sent[lyr]:
            for a, b, lbl in sent[lyr]:
                if (
                    lbl == "X"
                ):  # skip this, which covers an entire phrase typically containing the target and all its FEs
                    # (but do display the Gov)
                    continue
                if any(1 for x, y, felbl in sent.FE[0] if x <= a < y or a <= x < b):
                    # overlap between one of the POS-specific layers and first FE layer
                    posspec_separate = (
                        True  # show POS-specific layers on a separate line
                    )
                posspec.append(
                    (a, b, lbl.lower().replace("-", ""))
                )  # lowercase Cop=>cop, Non-Asp=>nonasp, etc. to distinguish from FE names
    if posspec_separate:
        POSSPEC = _annotation_ascii_FE_layer(posspec, {}, feAbbrevs)
    FE1 = _annotation_ascii_FE_layer(
        sorted(sent.FE[0] + (posspec if not posspec_separate else [])),
        sent.FE[1],
        feAbbrevs,
    )
    FE2 = FE3 = None
    if "FE2" in sent:
        FE2 = _annotation_ascii_FE_layer(sent.FE2[0], sent.FE2[1], feAbbrevs)
        if "FE3" in sent:
            FE3 = _annotation_ascii_FE_layer(sent.FE3[0], sent.FE3[1], feAbbrevs)

    for i, j in sent.Target:
        FE1span, FE1name, FE1exp = FE1
        if len(FE1span) < j:
            FE1span += " " * (j - len(FE1span))
        if len(FE1name) < j:
            FE1name += " " * (j - len(FE1name))
            FE1[1] = FE1name
        FE1[0] = (
            FE1span[:i] + FE1span[i:j].replace(" ", "*").replace("-", "=") + FE1span[j:]
        )
    long_lines = [sent.text]
    if posspec_separate:
        long_lines.extend(POSSPEC[:2])
    long_lines.extend([FE1[0], FE1[1] + FE1[2]])  # lines with no length limit
    if FE2:
        long_lines.extend([FE2[0], FE2[1] + FE2[2]])
        if FE3:
            long_lines.extend([FE3[0], FE3[1] + FE3[2]])
    long_lines.append("")
    outstr = "\n".join(
        map("\n".join, zip_longest(*mimic_wrap(long_lines), fillvalue=" "))
    )
    if feAbbrevs:
        outstr += "(" + ", ".join("=".join(pair) for pair in feAbbrevs.items()) + ")"
        assert len(feAbbrevs) == len(dict(feAbbrevs)), "Abbreviation clash"
    outstr += "\n"

    return outstr


def _pretty_fe(fe):

    """
    Helper function for pretty-printing a frame element.

    :param fe: The frame element to be printed.
    :type fe: AttrDict
    :return: A nicely formatted string representation of the frame element.
    :rtype: str
    """
    fekeys = fe.keys()
    outstr = ""
    outstr += "frame element ({0.ID}): {0.name}\n    of {1.name}({1.ID})\n".format(
        fe, fe.frame
    )
    if "definition" in fekeys:
        outstr += "[definition]\n"
        outstr += _pretty_longstring(fe.definition, "  ")
    if "abbrev" in fekeys:
        outstr += f"[abbrev] {fe.abbrev}\n"
    if "coreType" in fekeys:
        outstr += f"[coreType] {fe.coreType}\n"
    if "requiresFE" in fekeys:
        outstr += "[requiresFE] "
        if fe.requiresFE is None:
            outstr += "<None>\n"
        else:
            outstr += f"{fe.requiresFE.name}({fe.requiresFE.ID})\n"
    if "excludesFE" in fekeys:
        outstr += "[excludesFE] "
        if fe.excludesFE is None:
            outstr += "<None>\n"
        else:
            outstr += f"{fe.excludesFE.name}({fe.excludesFE.ID})\n"
    if "semType" in fekeys:
        outstr += "[semType] "
        if fe.semType is None:
            outstr += "<None>\n"
        else:
            outstr += "\n  " + f"{fe.semType.name}({fe.semType.ID})" + "\n"

    return outstr


def _pretty_frame(frame):

    """
    Helper function for pretty-printing a frame.

    :param frame: The frame to be printed.
    :type frame: AttrDict
    :return: A nicely formatted string representation of the frame.
    :rtype: str
    """

    outstr = ""
    outstr += "frame ({0.ID}): {0.name}\n\n".format(frame)
    outstr += f"[URL] {frame.URL}\n\n"
    outstr += "[definition]\n"
    outstr += _pretty_longstring(frame.definition, "  ") + "\n"

    outstr += f"[semTypes] {len(frame.semTypes)} semantic types\n"
    outstr += (
        "  " * (len(frame.semTypes) > 0)
        + ", ".join(f"{x.name}({x.ID})" for x in frame.semTypes)
        + "\n" * (len(frame.semTypes) > 0)
    )

    outstr += "\n[frameRelations] {} frame relations\n".format(
        len(frame.frameRelations)
    )
    outstr += "  " + "\n  ".join(repr(frel) for frel in frame.frameRelations) + "\n"

    outstr += f"\n[lexUnit] {len(frame.lexUnit)} lexical units\n"
    lustrs = []
    for luName, lu in sorted(frame.lexUnit.items()):
        tmpstr = f"{luName} ({lu.ID})"
        lustrs.append(tmpstr)
    outstr += "{}\n".format(_pretty_longstring(", ".join(lustrs), prefix="  "))

    outstr += f"\n[FE] {len(frame.FE)} frame elements\n"
    fes = {}
    for feName, fe in sorted(frame.FE.items()):
        try:
            fes[fe.coreType].append(f"{feName} ({fe.ID})")
        except KeyError:
            fes[fe.coreType] = []
            fes[fe.coreType].append(f"{feName} ({fe.ID})")
    for ct in sorted(
        fes.keys(),
        key=lambda ct2: [
            "Core",
            "Core-Unexpressed",
            "Peripheral",
            "Extra-Thematic",
        ].index(ct2),
    ):
        outstr += "{:>16}: {}\n".format(ct, ", ".join(sorted(fes[ct])))

    outstr += "\n[FEcoreSets] {} frame element core sets\n".format(
        len(frame.FEcoreSets)
    )
    outstr += (
        "  "
        + "\n  ".join(
            ", ".join([x.name for x in coreSet]) for coreSet in frame.FEcoreSets
        )
        + "\n"
    )

    return outstr


class FramenetError(Exception):

    """An exception class for framenet-related errors."""


class AttrDict(dict):

    """A class that wraps a dict and allows accessing the keys of the
    dict as if they were attributes. Taken from here:
    https://stackoverflow.com/a/14620633/8879

    >>> foo = {'a':1, 'b':2, 'c':3}
    >>> bar = AttrDict(foo)
    >>> pprint(dict(bar))
    {'a': 1, 'b': 2, 'c': 3}
    >>> bar.b
    2
    >>> bar.d = 4
    >>> pprint(dict(bar))
    {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.__dict__ = self

    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        if name == "_short_repr":
            return self._short_repr
        return self[name]

    def __getitem__(self, name):
        v = super().__getitem__(name)
        if isinstance(v, Future):
            return v._data()
        return v

    def _short_repr(self):
        if "_type" in self:
            if self["_type"].endswith("relation"):
                return self.__repr__()
            try:
                return "<{} ID={} name={}>".format(
                    self["_type"], self["ID"], self["name"]
                )
            except KeyError:
                try:  # no ID--e.g., for _type=lusubcorpus
                    return "<{} name={}>".format(self["_type"], self["name"])
                except KeyError:  # no name--e.g., for _type=lusentence
                    return "<{} ID={}>".format(self["_type"], self["ID"])
        else:
            return self.__repr__()

    def _str(self):
        outstr = ""

        if "_type" not in self:
            outstr = _pretty_any(self)
        elif self["_type"] == "frame":
            outstr = _pretty_frame(self)
        elif self["_type"] == "fe":
            outstr = _pretty_fe(self)
        elif self["_type"] == "lu":
            outstr = _pretty_lu(self)
        elif self["_type"] == "luexemplars":  # list of ALL exemplars for LU
            outstr = _pretty_exemplars(self, self[0].LU)
        elif (
            self["_type"] == "fulltext_annotation"
        ):  # list of all sentences for full-text doc
            outstr = _pretty_fulltext_sentences(self)
        elif self["_type"] == "lusentence":
            outstr = _pretty_annotation(self)
        elif self["_type"] == "fulltext_sentence":
            outstr = _pretty_fulltext_sentence(self)
        elif self["_type"] in ("luannotationset", "fulltext_annotationset"):
            outstr = _pretty_annotation(self, aset_level=True)
        elif self["_type"] == "posannotationset":
            outstr = _pretty_pos(self)
        elif self["_type"] == "semtype":
            outstr = _pretty_semtype(self)
        elif self["_type"] == "framerelationtype":
            outstr = _pretty_frame_relation_type(self)
        elif self["_type"] == "framerelation":
            outstr = _pretty_frame_relation(self)
        elif self["_type"] == "ferelation":
            outstr = _pretty_fe_relation(self)
        else:
            outstr = _pretty_any(self)

        # ensure result is unicode string prior to applying the
        #  decorator (because non-ASCII characters
        # could in principle occur in the data and would trigger an encoding error when
        # passed as arguments to str.format()).
        # assert isinstance(outstr, unicode) # not in Python 3.2
        return outstr

    def __str__(self):
        return self._str()

    def __repr__(self):
        return self.__str__()


class SpecialList(list):
    """
    A list subclass which adds a '_type' attribute for special printing
    (similar to an AttrDict, though this is NOT an AttrDict subclass).
    """

    def __init__(self, typ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._type = typ

    def _str(self):
        outstr = ""

        assert self._type
        if len(self) == 0:
            outstr = "[]"
        elif self._type == "luexemplars":  # list of ALL exemplars for LU
            outstr = _pretty_exemplars(self, self[0].LU)
        else:
            assert False, self._type
        return outstr

    def __str__(self):
        return self._str()

    def __repr__(self):
        return self.__str__()


class Future:
    """
    Wraps and acts as a proxy for a value to be loaded lazily (on demand).
    Adapted from https://gist.github.com/sergey-miryanov/2935416
    """

    def __init__(self, loader, *args, **kwargs):
        """
        :param loader: when called with no arguments, returns the value to be stored
        :type loader: callable
        """
        super().__init__(*args, **kwargs)
        self._loader = loader
        self._d = None

    def _data(self):
        if callable(self._loader):
            self._d = self._loader()
            self._loader = None  # the data is now cached
        return self._d

    def __nonzero__(self):
        return bool(self._data())

    def __len__(self):
        return len(self._data())

    def __setitem__(self, key, value):
        return self._data().__setitem__(key, value)

    def __getitem__(self, key):
        return self._data().__getitem__(key)

    def __getattr__(self, key):
        return self._data().__getattr__(key)

    def __str__(self):
        return self._data().__str__()

    def __repr__(self):
        return self._data().__repr__()


class PrettyDict(AttrDict):
    """
    Displays an abbreviated repr of values where possible.
    Inherits from AttrDict, so a callable value will
    be lazily converted to an actual value.
    """

    def __init__(self, *args, **kwargs):
        _BREAK_LINES = kwargs.pop("breakLines", False)
        super().__init__(*args, **kwargs)
        dict.__setattr__(self, "_BREAK_LINES", _BREAK_LINES)

    def __repr__(self):
        parts = []
        for k, v in sorted(self.items()):
            kv = repr(k) + ": "
            try:
                kv += v._short_repr()
            except AttributeError:
                kv += repr(v)
            parts.append(kv)
        return "{" + (",\n " if self._BREAK_LINES else ", ").join(parts) + "}"


class PrettyList(list):
    """
    Displays an abbreviated repr of only the first several elements, not the whole list.
    """

    # from nltk.util
    def __init__(self, *args, **kwargs):
        self._MAX_REPR_SIZE = kwargs.pop("maxReprSize", 60)
        self._BREAK_LINES = kwargs.pop("breakLines", False)
        super().__init__(*args, **kwargs)

    def __repr__(self):
        """
        Return a string representation for this corpus view that is
        similar to a list's representation; but if it would be more
        than 60 characters long, it is truncated.
        """
        pieces = []
        length = 5

        for elt in self:
            pieces.append(
                elt._short_repr()
            )  # key difference from inherited version: call to _short_repr()
            length += len(pieces[-1]) + 2
            if self._MAX_REPR_SIZE and length > self._MAX_REPR_SIZE and len(pieces) > 2:
                return "[%s, ...]" % str(",\n " if self._BREAK_LINES else ", ").join(
                    pieces[:-1]
                )
        return "[%s]" % str(",\n " if self._BREAK_LINES else ", ").join(pieces)


class PrettyLazyMap(LazyMap):
    """
    Displays an abbreviated repr of only the first several elements, not the whole list.
    """

    # from nltk.util
    _MAX_REPR_SIZE = 60

    def __repr__(self):
        """
        Return a string representation for this corpus view that is
        similar to a list's representation; but if it would be more
        than 60 characters long, it is truncated.
        """
        pieces = []
        length = 5
        for elt in self:
            pieces.append(
                elt._short_repr()
            )  # key difference from inherited version: call to _short_repr()
            length += len(pieces[-1]) + 2
            if length > self._MAX_REPR_SIZE and len(pieces) > 2:
                return "[%s, ...]" % ", ".join(pieces[:-1])
        return "[%s]" % ", ".join(pieces)


class PrettyLazyIteratorList(LazyIteratorList):
    """
    Displays an abbreviated repr of only the first several elements, not the whole list.
    """

    # from nltk.util
    _MAX_REPR_SIZE = 60

    def __repr__(self):
        """
        Return a string representation for this corpus view that is
        similar to a list's representation; but if it would be more
        than 60 characters long, it is truncated.
        """
        pieces = []
        length = 5
        for elt in self:
            pieces.append(
                elt._short_repr()
            )  # key difference from inherited version: call to _short_repr()
            length += len(pieces[-1]) + 2
            if length > self._MAX_REPR_SIZE and len(pieces) > 2:
                return "[%s, ...]" % ", ".join(pieces[:-1])
        return "[%s]" % ", ".join(pieces)


class PrettyLazyConcatenation(LazyConcatenation):
    """
    Displays an abbreviated repr of only the first several elements, not the whole list.
    """

    # from nltk.util
    _MAX_REPR_SIZE = 60

    def __repr__(self):
        """
        Return a string representation for this corpus view that is
        similar to a list's representation; but if it would be more
        than 60 characters long, it is truncated.
        """
        pieces = []
        length = 5
        for elt in self:
            pieces.append(
                elt._short_repr()
            )  # key difference from inherited version: call to _short_repr()
            length += len(pieces[-1]) + 2
            if length > self._MAX_REPR_SIZE and len(pieces) > 2:
                return "[%s, ...]" % ", ".join(pieces[:-1])
        return "[%s]" % ", ".join(pieces)

    def __add__(self, other):
        """Return a list concatenating self with other."""
        return PrettyLazyIteratorList(itertools.chain(self, other))

    def __radd__(self, other):
        """Return a list concatenating other with self."""
        return PrettyLazyIteratorList(itertools.chain(other, self))


class FramenetCorpusReader(XMLCorpusReader):
    """A corpus reader for the Framenet Corpus.

    >>> from nltk.corpus import framenet as fn
    >>> fn.lu(3238).frame.lexUnit['glint.v'] is fn.lu(3238)
    True
    >>> fn.frame_by_name('Replacing') is fn.lus('replace.v')[0].frame
    True
    >>> fn.lus('prejudice.n')[0].frame.frameRelations == fn.frame_relations('Partiality')
    True
    """

    _bad_statuses = ["Problem"]
    """
    When loading LUs for a frame, those whose status is in this list will be ignored.
    Due to caching, if user code modifies this, it should do so before loading any data.
    'Problem' should always be listed for FrameNet 1.5, as these LUs are not included
    in the XML index.
    """

    _warnings = False

    def warnings(self, v):
        """Enable or disable warnings of data integrity issues as they are encountered.
        If v is truthy, warnings will be enabled.

        (This is a function rather than just an attribute/property to ensure that if
        enabling warnings is the first action taken, the corpus reader is instantiated first.)
        """
        self._warnings = v

    def __init__(self, root, fileids):
        XMLCorpusReader.__init__(self, root, fileids)

        # framenet corpus sub dirs
        # sub dir containing the xml files for frames
        self._frame_dir = "frame"
        # sub dir containing the xml files for lexical units
        self._lu_dir = "lu"
        # sub dir containing the xml files for fulltext annotation files
        self._fulltext_dir = "fulltext"

        # location of latest development version of FrameNet
        self._fnweb_url = "https://framenet2.icsi.berkeley.edu/fnReports/data"

        # Indexes used for faster look-ups
        self._frame_idx = None
        self._cached_frames = {}  # name -> ID
        self._lu_idx = None
        self._fulltext_idx = None
        self._semtypes = None
        self._freltyp_idx = None  # frame relation types (Inheritance, Using, etc.)
        self._frel_idx = None  # frame-to-frame relation instances
        self._ferel_idx = None  # FE-to-FE relation instances
        self._frel_f_idx = None  # frame-to-frame relations associated with each frame

        self._readme = "README.txt"

    def help(self, attrname=None):
        """Display help information summarizing the main methods."""

        if attrname is not None:
            return help(self.__getattribute__(attrname))

        # No need to mention frame_by_name() or frame_by_id(),
        # as it's easier to just call frame().
        # Also not mentioning lu_basic().

        msg = """
Citation: Nathan Schneider and Chuck Wooters (2017),
"The NLTK FrameNet API: Designing for Discoverability with a Rich Linguistic Resource".
Proceedings of EMNLP: System Demonstrations. https://arxiv.org/abs/1703.07438

Use the following methods to access data in FrameNet.
Provide a method name to `help()` for more information.

FRAMES
======

frame() to look up a frame by its exact name or ID
frames() to get frames matching a name pattern
frames_by_lemma() to get frames containing an LU matching a name pattern
frame_ids_and_names() to get a mapping from frame IDs to names

FRAME ELEMENTS
==============

fes() to get frame elements (a.k.a. roles) matching a name pattern, optionally constrained
  by a frame name pattern

LEXICAL UNITS
=============

lu() to look up an LU by its ID
lus() to get lexical units matching a name pattern, optionally constrained by frame
lu_ids_and_names() to get a mapping from LU IDs to names

RELATIONS
=========

frame_relation_types() to get the different kinds of frame-to-frame relations
  (Inheritance, Subframe, Using, etc.).
frame_relations() to get the relation instances, optionally constrained by
  frame(s) or relation type
fe_relations() to get the frame element pairs belonging to a frame-to-frame relation

SEMANTIC TYPES
==============

semtypes() to get the different kinds of semantic types that can be applied to
  FEs, LUs, and entire frames
semtype() to look up a particular semtype by name, ID, or abbreviation
semtype_inherits() to check whether two semantic types have a subtype-supertype
  relationship in the semtype hierarchy
propagate_semtypes() to apply inference rules that distribute semtypes over relations
  between FEs

ANNOTATIONS
===========

annotations() to get annotation sets, in which a token in a sentence is annotated
  with a lexical unit in a frame, along with its frame elements and their syntactic properties;
  can be constrained by LU name pattern and limited to lexicographic exemplars or full-text.
  Sentences of full-text annotation can have multiple annotation sets.
sents() to get annotated sentences illustrating one or more lexical units
exemplars() to get sentences of lexicographic annotation, most of which have
  just 1 annotation set; can be constrained by LU name pattern, frame, and overt FE(s)
doc() to look up a document of full-text annotation by its ID
docs() to get documents of full-text annotation that match a name pattern
docs_metadata() to get metadata about all full-text documents without loading them
ft_sents() to iterate over sentences of full-text annotation

UTILITIES
=========

buildindexes() loads metadata about all frames, LUs, etc. into memory to avoid
  delay when one is accessed for the first time. It does not load annotations.
readme() gives the text of the FrameNet README file
warnings(True) to display corpus consistency warnings when loading data
        """
        print(msg)

    def _buildframeindex(self):
        # The total number of Frames in Framenet is fairly small (~1200) so
        # this index should not be very large
        if not self._frel_idx:
            self._buildrelationindex()  # always load frame relations before frames,
            # otherwise weird ordering effects might result in incomplete information
        self._frame_idx = {}
        with XMLCorpusView(
            self.abspath("frameIndex.xml"), "frameIndex/frame", self._handle_elt
        ) as view:
            for f in view:
                self._frame_idx[f["ID"]] = f

    def _buildcorpusindex(self):
        # The total number of fulltext annotated documents in Framenet
        # is fairly small (~90) so this index should not be very large
        self._fulltext_idx = {}
        with XMLCorpusView(
            self.abspath("fulltextIndex.xml"),
            "fulltextIndex/corpus",
            self._handle_fulltextindex_elt,
        ) as view:
            for doclist in view:
                for doc in doclist:
                    self._fulltext_idx[doc.ID] = doc

    def _buildluindex(self):
        # The number of LUs in Framenet is about 13,000 so this index
        # should not be very large
        self._lu_idx = {}
        with XMLCorpusView(
            self.abspath("luIndex.xml"), "luIndex/lu", self._handle_elt
        ) as view:
            for lu in view:
                self._lu_idx[
                    lu["ID"]
                ] = lu  # populate with LU index entries. if any of these
                # are looked up they will be replaced by full LU objects.

    def _buildrelationindex(self):
        # print('building relation index...', file=sys.stderr)
        self._freltyp_idx = {}
        self._frel_idx = {}
        self._frel_f_idx = defaultdict(set)
        self._ferel_idx = {}

        with XMLCorpusView(
            self.abspath("frRelation.xml"),
            "frameRelations/frameRelationType",
            self._handle_framerelationtype_elt,
        ) as view:
            for freltyp in view:
                self._freltyp_idx[freltyp.ID] = freltyp
                for frel in freltyp.frameRelations:
                    supF = frel.superFrame = frel[freltyp.superFrameName] = Future(
                        (lambda fID: lambda: self.frame_by_id(fID))(frel.supID)
                    )
                    subF = frel.subFrame = frel[freltyp.subFrameName] = Future(
                        (lambda fID: lambda: self.frame_by_id(fID))(frel.subID)
                    )
                    self._frel_idx[frel.ID] = frel
                    self._frel_f_idx[frel.supID].add(frel.ID)
                    self._frel_f_idx[frel.subID].add(frel.ID)
                    for ferel in frel.feRelations:
                        ferel.superFrame = supF
                        ferel.subFrame = subF
                        ferel.superFE = Future(
                            (lambda fer: lambda: fer.superFrame.FE[fer.superFEName])(
                                ferel
                            )
                        )
                        ferel.subFE = Future(
                            (lambda fer: lambda: fer.subFrame.FE[fer.subFEName])(ferel)
                        )
                        self._ferel_idx[ferel.ID] = ferel
        # print('...done building relation index', file=sys.stderr)

    def _warn(self, *message, **kwargs):
        if self._warnings:
            kwargs.setdefault("file", sys.stderr)
            print(*message, **kwargs)

    def buildindexes(self):
        """
        Build the internal indexes to make look-ups faster.
        """
        # Frames
        self._buildframeindex()
        # LUs
        self._buildluindex()
        # Fulltext annotation corpora index
        self._buildcorpusindex()
        # frame and FE relations
        self._buildrelationindex()

    def doc(self, fn_docid):
        """
        Returns the annotated document whose id number is
        ``fn_docid``. This id number can be obtained by calling the
        Documents() function.

        The dict that is returned from this function will contain the
        following keys:

        - '_type'      : 'fulltextannotation'
        - 'sentence'   : a list of sentences in the document
           - Each item in the list is a dict containing the following keys:
              - 'ID'    : the ID number of the sentence
              - '_type' : 'sentence'
              - 'text'  : the text of the sentence
              - 'paragNo' : the paragraph number
              - 'sentNo'  : the sentence number
              - 'docID'   : the document ID number
              - 'corpID'  : the corpus ID number
              - 'aPos'    : the annotation position
              - 'annotationSet' : a list of annotation layers for the sentence
                 - Each item in the list is a dict containing the following keys:
                    - 'ID'       : the ID number of the annotation set
                    - '_type'    : 'annotationset'
                    - 'status'   : either 'MANUAL' or 'UNANN'
                    - 'luName'   : (only if status is 'MANUAL')
                    - 'luID'     : (only if status is 'MANUAL')
                    - 'frameID'  : (only if status is 'MANUAL')
                    - 'frameName': (only if status is 'MANUAL')
                    - 'layer' : a list of labels for the layer
                       - Each item in the layer is a dict containing the following keys:
                          - '_type': 'layer'
                          - 'rank'
                          - 'name'
                          - 'label' : a list of labels in the layer
                             - Each item is a dict containing the following keys:
                                - 'start'
                                - 'end'
                                - 'name'
                                - 'feID' (optional)

        :param fn_docid: The Framenet id number of the document
        :type fn_docid: int
        :return: Information about the annotated document
        :rtype: dict
        """
        try:
            xmlfname = self._fulltext_idx[fn_docid].filename
        except TypeError:  # happens when self._fulltext_idx == None
            # build the index
            self._buildcorpusindex()
            xmlfname = self._fulltext_idx[fn_docid].filename
        except KeyError as e:  # probably means that fn_docid was not in the index
            raise FramenetError(f"Unknown document id: {fn_docid}") from e

        # construct the path name for the xml file containing the document info
        locpath = os.path.join(f"{self._root}", self._fulltext_dir, xmlfname)

        # Grab the top-level xml element containing the fulltext annotation
        with XMLCorpusView(locpath, "fullTextAnnotation") as view:
            elt = view[0]
        info = self._handle_fulltextannotation_elt(elt)
        # add metadata
        for k, v in self._fulltext_idx[fn_docid].items():
            info[k] = v
        return info

    def frame_by_id(self, fn_fid, ignorekeys=[]):
        """
        Get the details for the specified Frame using the frame's id
        number.

        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> f = fn.frame_by_id(256)
        >>> f.ID
        256
        >>> f.name
        'Medical_specialties'
        >>> f.definition # doctest: +NORMALIZE_WHITESPACE
        "This frame includes words that name medical specialties and is closely related to the
        Medical_professionals frame.  The FE Type characterizing a sub-are in a Specialty may also be
        expressed. 'Ralph practices paediatric oncology.'"

        :param fn_fid: The Framenet id number of the frame
        :type fn_fid: int
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: Information about a frame
        :rtype: dict

        Also see the ``frame()`` function for details about what is
        contained in the dict that is returned.
        """

        # get the name of the frame with this id number
        try:
            fentry = self._frame_idx[fn_fid]
            if "_type" in fentry:
                return fentry  # full frame object is cached
            name = fentry["name"]
        except TypeError:
            self._buildframeindex()
            name = self._frame_idx[fn_fid]["name"]
        except KeyError as e:
            raise FramenetError(f"Unknown frame id: {fn_fid}") from e

        return self.frame_by_name(name, ignorekeys, check_cache=False)

    def frame_by_name(self, fn_fname, ignorekeys=[], check_cache=True):
        """
        Get the details for the specified Frame using the frame's name.

        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> f = fn.frame_by_name('Medical_specialties')
        >>> f.ID
        256
        >>> f.name
        'Medical_specialties'
        >>> f.definition # doctest: +NORMALIZE_WHITESPACE
         "This frame includes words that name medical specialties and is closely related to the
          Medical_professionals frame.  The FE Type characterizing a sub-are in a Specialty may also be
          expressed. 'Ralph practices paediatric oncology.'"

        :param fn_fname: The name of the frame
        :type fn_fname: str
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: Information about a frame
        :rtype: dict

        Also see the ``frame()`` function for details about what is
        contained in the dict that is returned.
        """

        if check_cache and fn_fname in self._cached_frames:
            return self._frame_idx[self._cached_frames[fn_fname]]
        elif not self._frame_idx:
            self._buildframeindex()

        # construct the path name for the xml file containing the Frame info
        locpath = os.path.join(f"{self._root}", self._frame_dir, fn_fname + ".xml")
        # print(locpath, file=sys.stderr)
        # Grab the xml for the frame
        try:
            with XMLCorpusView(locpath, "frame") as view:
                elt = view[0]
        except OSError as e:
            raise FramenetError(f"Unknown frame: {fn_fname}") from e

        fentry = self._handle_frame_elt(elt, ignorekeys)
        assert fentry

        fentry.URL = self._fnweb_url + "/" + self._frame_dir + "/" + fn_fname + ".xml"

        # INFERENCE RULE: propagate lexical semtypes from the frame to all its LUs
        for st in fentry.semTypes:
            if st.rootType.name == "Lexical_type":
                for lu in fentry.lexUnit.values():
                    if not any(
                        x is st for x in lu.semTypes
                    ):  # identity containment check
                        lu.semTypes.append(st)

        self._frame_idx[fentry.ID] = fentry
        self._cached_frames[fentry.name] = fentry.ID
        """
        # now set up callables to resolve the LU pointers lazily.
        # (could also do this here--caching avoids infinite recursion.)
        for luName,luinfo in fentry.lexUnit.items():
            fentry.lexUnit[luName] = (lambda luID: Future(lambda: self.lu(luID)))(luinfo.ID)
        """
        return fentry

    def frame(self, fn_fid_or_fname, ignorekeys=[]):
        """
        Get the details for the specified Frame using the frame's name
        or id number.

        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> f = fn.frame(256)
        >>> f.name
        'Medical_specialties'
        >>> f = fn.frame('Medical_specialties')
        >>> f.ID
        256
        >>> # ensure non-ASCII character in definition doesn't trigger an encoding error:
        >>> fn.frame('Imposing_obligation') # doctest: +ELLIPSIS
        frame (1494): Imposing_obligation...


        The dict that is returned from this function will contain the
        following information about the Frame:

        - 'name'       : the name of the Frame (e.g. 'Birth', 'Apply_heat', etc.)
        - 'definition' : textual definition of the Frame
        - 'ID'         : the internal ID number of the Frame
        - 'semTypes'   : a list of semantic types for this frame
           - Each item in the list is a dict containing the following keys:
              - 'name' : can be used with the semtype() function
              - 'ID'   : can be used with the semtype() function

        - 'lexUnit'    : a dict containing all of the LUs for this frame.
                         The keys in this dict are the names of the LUs and
                         the value for each key is itself a dict containing
                         info about the LU (see the lu() function for more info.)

        - 'FE' : a dict containing the Frame Elements that are part of this frame
                 The keys in this dict are the names of the FEs (e.g. 'Body_system')
                 and the values are dicts containing the following keys

              - 'definition' : The definition of the FE
              - 'name'       : The name of the FE e.g. 'Body_system'
              - 'ID'         : The id number
              - '_type'      : 'fe'
              - 'abbrev'     : Abbreviation e.g. 'bod'
              - 'coreType'   : one of "Core", "Peripheral", or "Extra-Thematic"
              - 'semType'    : if not None, a dict with the following two keys:
                 - 'name' : name of the semantic type. can be used with
                            the semtype() function
                 - 'ID'   : id number of the semantic type. can be used with
                            the semtype() function
              - 'requiresFE' : if not None, a dict with the following two keys:
                 - 'name' : the name of another FE in this frame
                 - 'ID'   : the id of the other FE in this frame
              - 'excludesFE' : if not None, a dict with the following two keys:
                 - 'name' : the name of another FE in this frame
                 - 'ID'   : the id of the other FE in this frame

        - 'frameRelation'      : a list of objects describing frame relations
        - 'FEcoreSets'  : a list of Frame Element core sets for this frame
           - Each item in the list is a list of FE objects

        :param fn_fid_or_fname: The Framenet name or id number of the frame
        :type fn_fid_or_fname: int or str
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: Information about a frame
        :rtype: dict
        """

        # get the frame info by name or id number
        if isinstance(fn_fid_or_fname, str):
            f = self.frame_by_name(fn_fid_or_fname, ignorekeys)
        else:
            f = self.frame_by_id(fn_fid_or_fname, ignorekeys)

        return f

    def frames_by_lemma(self, pat):
        """
        Returns a list of all frames that contain LUs in which the
        ``name`` attribute of the LU matches the given regular expression
        ``pat``. Note that LU names are composed of "lemma.POS", where
        the "lemma" part can be made up of either a single lexeme
        (e.g. 'run') or multiple lexemes (e.g. 'a little').

        Note: if you are going to be doing a lot of this type of
        searching, you'd want to build an index that maps from lemmas to
        frames because each time frames_by_lemma() is called, it has to
        search through ALL of the frame XML files in the db.

        >>> from nltk.corpus import framenet as fn
        >>> from nltk.corpus.reader.framenet import PrettyList
        >>> PrettyList(sorted(fn.frames_by_lemma(r'(?i)a little'), key=itemgetter('ID'))) # doctest: +ELLIPSIS
        [<frame ID=189 name=Quanti...>, <frame ID=2001 name=Degree>]

        :return: A list of frame objects.
        :rtype: list(AttrDict)
        """
        return PrettyList(
            f
            for f in self.frames()
            if any(re.search(pat, luName) for luName in f.lexUnit)
        )

    def lu_basic(self, fn_luid):
        """
        Returns basic information about the LU whose id is
        ``fn_luid``. This is basically just a wrapper around the
        ``lu()`` function with "subCorpus" info excluded.

        >>> from nltk.corpus import framenet as fn
        >>> lu = PrettyDict(fn.lu_basic(256), breakLines=True)
        >>> # ellipses account for differences between FN 1.5 and 1.7
        >>> lu # doctest: +ELLIPSIS
        {'ID': 256,
         'POS': 'V',
         'URL': 'https://framenet2.icsi.berkeley.edu/fnReports/data/lu/lu256.xml',
         '_type': 'lu',
         'cBy': ...,
         'cDate': '02/08/2001 01:27:50 PST Thu',
         'definition': 'COD: be aware of beforehand; predict.',
         'definitionMarkup': 'COD: be aware of beforehand; predict.',
         'frame': <frame ID=26 name=Expectation>,
         'lemmaID': 15082,
         'lexemes': [{'POS': 'V', 'breakBefore': 'false', 'headword': 'false', 'name': 'foresee', 'order': 1}],
         'name': 'foresee.v',
         'semTypes': [],
         'sentenceCount': {'annotated': ..., 'total': ...},
         'status': 'FN1_Sent'}

        :param fn_luid: The id number of the desired LU
        :type fn_luid: int
        :return: Basic information about the lexical unit
        :rtype: dict
        """
        return self.lu(fn_luid, ignorekeys=["subCorpus", "exemplars"])

    def lu(self, fn_luid, ignorekeys=[], luName=None, frameID=None, frameName=None):
        """
        Access a lexical unit by its ID. luName, frameID, and frameName are used
        only in the event that the LU does not have a file in the database
        (which is the case for LUs with "Problem" status); in this case,
        a placeholder LU is created which just contains its name, ID, and frame.


        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> fn.lu(256).name
        'foresee.v'
        >>> fn.lu(256).definition
        'COD: be aware of beforehand; predict.'
        >>> fn.lu(256).frame.name
        'Expectation'
        >>> list(map(PrettyDict, fn.lu(256).lexemes))
        [{'POS': 'V', 'breakBefore': 'false', 'headword': 'false', 'name': 'foresee', 'order': 1}]

        >>> fn.lu(227).exemplars[23] # doctest: +NORMALIZE_WHITESPACE
        exemplar sentence (352962):
        [sentNo] 0
        [aPos] 59699508
        <BLANKLINE>
        [LU] (227) guess.v in Coming_to_believe
        <BLANKLINE>
        [frame] (23) Coming_to_believe
        <BLANKLINE>
        [annotationSet] 2 annotation sets
        <BLANKLINE>
        [POS] 18 tags
        <BLANKLINE>
        [POS_tagset] BNC
        <BLANKLINE>
        [GF] 3 relations
        <BLANKLINE>
        [PT] 3 phrases
        <BLANKLINE>
        [Other] 1 entry
        <BLANKLINE>
        [text] + [Target] + [FE]
        <BLANKLINE>
        When he was inside the house , Culley noticed the characteristic
                                                      ------------------
                                                      Content
        <BLANKLINE>
        he would n't have guessed at .
        --                ******* --
        Co                        C1 [Evidence:INI]
         (Co=Cognizer, C1=Content)
        <BLANKLINE>
        <BLANKLINE>

        The dict that is returned from this function will contain most of the
        following information about the LU. Note that some LUs do not contain
        all of these pieces of information - particularly 'totalAnnotated' and
        'incorporatedFE' may be missing in some LUs:

        - 'name'       : the name of the LU (e.g. 'merger.n')
        - 'definition' : textual definition of the LU
        - 'ID'         : the internal ID number of the LU
        - '_type'      : 'lu'
        - 'status'     : e.g. 'Created'
        - 'frame'      : Frame that this LU belongs to
        - 'POS'        : the part of speech of this LU (e.g. 'N')
        - 'totalAnnotated' : total number of examples annotated with this LU
        - 'incorporatedFE' : FE that incorporates this LU (e.g. 'Ailment')
        - 'sentenceCount'  : a dict with the following two keys:
                 - 'annotated': number of sentences annotated with this LU
                 - 'total'    : total number of sentences with this LU

        - 'lexemes'  : a list of dicts describing the lemma of this LU.
           Each dict in the list contains these keys:

           - 'POS'     : part of speech e.g. 'N'
           - 'name'    : either single-lexeme e.g. 'merger' or
                         multi-lexeme e.g. 'a little'
           - 'order': the order of the lexeme in the lemma (starting from 1)
           - 'headword': a boolean ('true' or 'false')
           - 'breakBefore': Can this lexeme be separated from the previous lexeme?
                Consider: "take over.v" as in::

                         Germany took over the Netherlands in 2 days.
                         Germany took the Netherlands over in 2 days.

                In this case, 'breakBefore' would be "true" for the lexeme
                "over". Contrast this with "take after.v" as in::

                         Mary takes after her grandmother.
                        *Mary takes her grandmother after.

                In this case, 'breakBefore' would be "false" for the lexeme "after"

        - 'lemmaID'    : Can be used to connect lemmas in different LUs
        - 'semTypes'   : a list of semantic type objects for this LU
        - 'subCorpus'  : a list of subcorpora
           - Each item in the list is a dict containing the following keys:
              - 'name' :
              - 'sentence' : a list of sentences in the subcorpus
                 - each item in the list is a dict with the following keys:
                    - 'ID':
                    - 'sentNo':
                    - 'text': the text of the sentence
                    - 'aPos':
                    - 'annotationSet': a list of annotation sets
                       - each item in the list is a dict with the following keys:
                          - 'ID':
                          - 'status':
                          - 'layer': a list of layers
                             - each layer is a dict containing the following keys:
                                - 'name': layer name (e.g. 'BNC')
                                - 'rank':
                                - 'label': a list of labels for the layer
                                   - each label is a dict containing the following keys:
                                      - 'start': start pos of label in sentence 'text' (0-based)
                                      - 'end': end pos of label in sentence 'text' (0-based)
                                      - 'name': name of label (e.g. 'NN1')

        Under the hood, this implementation looks up the lexical unit information
        in the *frame* definition file. That file does not contain
        corpus annotations, so the LU files will be accessed on demand if those are
        needed. In principle, valence patterns could be loaded here too,
        though these are not currently supported.

        :param fn_luid: The id number of the lexical unit
        :type fn_luid: int
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: All information about the lexical unit
        :rtype: dict
        """
        # look for this LU in cache
        if not self._lu_idx:
            self._buildluindex()
        OOV = object()
        luinfo = self._lu_idx.get(fn_luid, OOV)
        if luinfo is OOV:
            # LU not in the index. We create a placeholder by falling back to
            # luName, frameID, and frameName. However, this will not be listed
            # among the LUs for its frame.
            self._warn(
                "LU ID not found: {} ({}) in {} ({})".format(
                    luName, fn_luid, frameName, frameID
                )
            )
            luinfo = AttrDict(
                {
                    "_type": "lu",
                    "ID": fn_luid,
                    "name": luName,
                    "frameID": frameID,
                    "status": "Problem",
                }
            )
            f = self.frame_by_id(luinfo.frameID)
            assert f.name == frameName, (f.name, frameName)
            luinfo["frame"] = f
            self._lu_idx[fn_luid] = luinfo
        elif "_type" not in luinfo:
            # we only have an index entry for the LU. loading the frame will replace this.
            f = self.frame_by_id(luinfo.frameID)
            luinfo = self._lu_idx[fn_luid]
        if ignorekeys:
            return AttrDict({k: v for k, v in luinfo.items() if k not in ignorekeys})

        return luinfo

    def _lu_file(self, lu, ignorekeys=[]):
        """
        Augment the LU information that was loaded from the frame file
        with additional information from the LU file.
        """
        fn_luid = lu.ID

        fname = f"lu{fn_luid}.xml"
        locpath = os.path.join(f"{self._root}", self._lu_dir, fname)
        # print(locpath, file=sys.stderr)
        if not self._lu_idx:
            self._buildluindex()

        try:
            with XMLCorpusView(locpath, "lexUnit") as view:
                elt = view[0]
        except OSError as e:
            raise FramenetError(f"Unknown LU id: {fn_luid}") from e

        lu2 = self._handle_lexunit_elt(elt, ignorekeys)
        lu.URL = self._fnweb_url + "/" + self._lu_dir + "/" + fname
        lu.subCorpus = lu2.subCorpus
        lu.exemplars = SpecialList(
            "luexemplars", [sent for subc in lu.subCorpus for sent in subc.sentence]
        )
        for sent in lu.exemplars:
            sent["LU"] = lu
            sent["frame"] = lu.frame
            for aset in sent.annotationSet:
                aset["LU"] = lu
                aset["frame"] = lu.frame

        return lu

    def _loadsemtypes(self):
        """Create the semantic types index."""
        self._semtypes = AttrDict()
        with XMLCorpusView(
            self.abspath("semTypes.xml"),
            "semTypes/semType",
            self._handle_semtype_elt,
        ) as view:
            for st in view:
                n = st["name"]
                a = st["abbrev"]
                i = st["ID"]
                # Both name and abbrev should be able to retrieve the
                # ID. The ID will retrieve the semantic type dict itself.
                self._semtypes[n] = i
                self._semtypes[a] = i
                self._semtypes[i] = st
        # now that all individual semtype XML is loaded, we can link them together
        roots = []
        for st in self.semtypes():
            if st.superType:
                st.superType = self.semtype(st.superType.supID)
                st.superType.subTypes.append(st)
            else:
                if st not in roots:
                    roots.append(st)
                st.rootType = st
        queue = list(roots)
        assert queue
        while queue:
            st = queue.pop(0)
            for child in st.subTypes:
                child.rootType = st.rootType
                queue.append(child)
        # self.propagate_semtypes()  # apply inferencing over FE relations

    def propagate_semtypes(self):
        """
        Apply inference rules to distribute semtypes over relations between FEs.
        For FrameNet 1.5, this results in 1011 semtypes being propagated.
        (Not done by default because it requires loading all frame files,
        which takes several seconds. If this needed to be fast, it could be rewritten
        to traverse the neighboring relations on demand for each FE semtype.)

        >>> from nltk.corpus import framenet as fn
        >>> x = sum(1 for f in fn.frames() for fe in f.FE.values() if fe.semType)
        >>> fn.propagate_semtypes()
        >>> y = sum(1 for f in fn.frames() for fe in f.FE.values() if fe.semType)
        >>> y-x > 1000
        True
        """
        if not self._semtypes:
            self._loadsemtypes()
        if not self._ferel_idx:
            self._buildrelationindex()
        changed = True
        i = 0
        nPropagations = 0
        while changed:
            # make a pass and see if anything needs to be propagated
            i += 1
            changed = False
            for ferel in self.fe_relations():
                superST = ferel.superFE.semType
                subST = ferel.subFE.semType
                try:
                    if superST and superST is not subST:
                        # propagate downward
                        assert subST is None or self.semtype_inherits(subST, superST), (
                            superST.name,
                            ferel,
                            subST.name,
                        )
                        if subST is None:
                            ferel.subFE.semType = subST = superST
                            changed = True
                            nPropagations += 1
                    if (
                        ferel.type.name in ["Perspective_on", "Subframe", "Precedes"]
                        and subST
                        and subST is not superST
                    ):
                        # propagate upward
                        assert superST is None, (superST.name, ferel, subST.name)
                        ferel.superFE.semType = superST = subST
                        changed = True
                        nPropagations += 1
                except AssertionError as ex:
                    # bug in the data! ignore
                    # print(ex, file=sys.stderr)
                    continue
            # print(i, nPropagations, file=sys.stderr)

    def semtype(self, key):
        """
        >>> from nltk.corpus import framenet as fn
        >>> fn.semtype(233).name
        'Temperature'
        >>> fn.semtype(233).abbrev
        'Temp'
        >>> fn.semtype('Temperature').ID
        233

        :param key: The name, abbreviation, or id number of the semantic type
        :type key: string or int
        :return: Information about a semantic type
        :rtype: dict
        """
        if isinstance(key, int):
            stid = key
        else:
            try:
                stid = self._semtypes[key]
            except TypeError:
                self._loadsemtypes()
                stid = self._semtypes[key]

        try:
            st = self._semtypes[stid]
        except TypeError:
            self._loadsemtypes()
            st = self._semtypes[stid]

        return st

    def semtype_inherits(self, st, superST):
        if not isinstance(st, dict):
            st = self.semtype(st)
        if not isinstance(superST, dict):
            superST = self.semtype(superST)
        par = st.superType
        while par:
            if par is superST:
                return True
            par = par.superType
        return False

    def frames(self, name=None):
        """
        Obtain details for a specific frame.

        >>> from nltk.corpus import framenet as fn
        >>> len(fn.frames()) in (1019, 1221)    # FN 1.5 and 1.7, resp.
        True
        >>> x = PrettyList(fn.frames(r'(?i)crim'), maxReprSize=0, breakLines=True)
        >>> x.sort(key=itemgetter('ID'))
        >>> x
        [<frame ID=200 name=Criminal_process>,
         <frame ID=500 name=Criminal_investigation>,
         <frame ID=692 name=Crime_scenario>,
         <frame ID=700 name=Committing_crime>]

        A brief intro to Frames (excerpted from "FrameNet II: Extended
        Theory and Practice" by Ruppenhofer et. al., 2010):

        A Frame is a script-like conceptual structure that describes a
        particular type of situation, object, or event along with the
        participants and props that are needed for that Frame. For
        example, the "Apply_heat" frame describes a common situation
        involving a Cook, some Food, and a Heating_Instrument, and is
        evoked by words such as bake, blanch, boil, broil, brown,
        simmer, steam, etc.

        We call the roles of a Frame "frame elements" (FEs) and the
        frame-evoking words are called "lexical units" (LUs).

        FrameNet includes relations between Frames. Several types of
        relations are defined, of which the most important are:

           - Inheritance: An IS-A relation. The child frame is a subtype
             of the parent frame, and each FE in the parent is bound to
             a corresponding FE in the child. An example is the
             "Revenge" frame which inherits from the
             "Rewards_and_punishments" frame.

           - Using: The child frame presupposes the parent frame as
             background, e.g the "Speed" frame "uses" (or presupposes)
             the "Motion" frame; however, not all parent FEs need to be
             bound to child FEs.

           - Subframe: The child frame is a subevent of a complex event
             represented by the parent, e.g. the "Criminal_process" frame
             has subframes of "Arrest", "Arraignment", "Trial", and
             "Sentencing".

           - Perspective_on: The child frame provides a particular
             perspective on an un-perspectivized parent frame. A pair of
             examples consists of the "Hiring" and "Get_a_job" frames,
             which perspectivize the "Employment_start" frame from the
             Employer's and the Employee's point of view, respectively.

        :param name: A regular expression pattern used to match against
            Frame names. If 'name' is None, then a list of all
            Framenet Frames will be returned.
        :type name: str
        :return: A list of matching Frames (or all Frames).
        :rtype: list(AttrDict)
        """
        try:
            fIDs = list(self._frame_idx.keys())
        except AttributeError:
            self._buildframeindex()
            fIDs = list(self._frame_idx.keys())

        if name is not None:
            return PrettyList(
                self.frame(fID) for fID, finfo in self.frame_ids_and_names(name).items()
            )
        else:
            return PrettyLazyMap(self.frame, fIDs)

    def frame_ids_and_names(self, name=None):
        """
        Uses the frame index, which is much faster than looking up each frame definition
        if only the names and IDs are needed.
        """
        if not self._frame_idx:
            self._buildframeindex()
        return {
            fID: finfo.name
            for fID, finfo in self._frame_idx.items()
            if name is None or re.search(name, finfo.name) is not None
        }

    def fes(self, name=None, frame=None):
        """
        Lists frame element objects. If 'name' is provided, this is treated as
        a case-insensitive regular expression to filter by frame name.
        (Case-insensitivity is because casing of frame element names is not always
        consistent across frames.) Specify 'frame' to filter by a frame name pattern,
        ID, or object.

        >>> from nltk.corpus import framenet as fn
        >>> fn.fes('Noise_maker')
        [<fe ID=6043 name=Noise_maker>]
        >>> sorted([(fe.frame.name,fe.name) for fe in fn.fes('sound')]) # doctest: +NORMALIZE_WHITESPACE
        [('Cause_to_make_noise', 'Sound_maker'), ('Make_noise', 'Sound'),
         ('Make_noise', 'Sound_source'), ('Sound_movement', 'Location_of_sound_source'),
         ('Sound_movement', 'Sound'), ('Sound_movement', 'Sound_source'),
         ('Sounds', 'Component_sound'), ('Sounds', 'Location_of_sound_source'),
         ('Sounds', 'Sound_source'), ('Vocalizations', 'Location_of_sound_source'),
         ('Vocalizations', 'Sound_source')]
        >>> sorted([(fe.frame.name,fe.name) for fe in fn.fes('sound',r'(?i)make_noise')]) # doctest: +NORMALIZE_WHITESPACE
        [('Cause_to_make_noise', 'Sound_maker'),
         ('Make_noise', 'Sound'),
         ('Make_noise', 'Sound_source')]
        >>> sorted(set(fe.name for fe in fn.fes('^sound')))
        ['Sound', 'Sound_maker', 'Sound_source']
        >>> len(fn.fes('^sound$'))
        2

        :param name: A regular expression pattern used to match against
            frame element names. If 'name' is None, then a list of all
            frame elements will be returned.
        :type name: str
        :return: A list of matching frame elements
        :rtype: list(AttrDict)
        """
        # what frames are we searching in?
        if frame is not None:
            if isinstance(frame, int):
                frames = [self.frame(frame)]
            elif isinstance(frame, str):
                frames = self.frames(frame)
            else:
                frames = [frame]
        else:
            frames = self.frames()

        return PrettyList(
            fe
            for f in frames
            for fename, fe in f.FE.items()
            if name is None or re.search(name, fename, re.I)
        )

    def lus(self, name=None, frame=None):
        """
        Obtain details for lexical units.
        Optionally restrict by lexical unit name pattern, and/or to a certain frame
        or frames whose name matches a pattern.

        >>> from nltk.corpus import framenet as fn
        >>> len(fn.lus()) in (11829, 13572) # FN 1.5 and 1.7, resp.
        True
        >>> PrettyList(sorted(fn.lus(r'(?i)a little'), key=itemgetter('ID')), maxReprSize=0, breakLines=True)
        [<lu ID=14733 name=a little.n>,
         <lu ID=14743 name=a little.adv>,
         <lu ID=14744 name=a little bit.adv>]
        >>> PrettyList(sorted(fn.lus(r'interest', r'(?i)stimulus'), key=itemgetter('ID')))
        [<lu ID=14894 name=interested.a>, <lu ID=14920 name=interesting.a>]

        A brief intro to Lexical Units (excerpted from "FrameNet II:
        Extended Theory and Practice" by Ruppenhofer et. al., 2010):

        A lexical unit (LU) is a pairing of a word with a meaning. For
        example, the "Apply_heat" Frame describes a common situation
        involving a Cook, some Food, and a Heating Instrument, and is
        _evoked_ by words such as bake, blanch, boil, broil, brown,
        simmer, steam, etc. These frame-evoking words are the LUs in the
        Apply_heat frame. Each sense of a polysemous word is a different
        LU.

        We have used the word "word" in talking about LUs. The reality
        is actually rather complex. When we say that the word "bake" is
        polysemous, we mean that the lemma "bake.v" (which has the
        word-forms "bake", "bakes", "baked", and "baking") is linked to
        three different frames:

           - Apply_heat: "Michelle baked the potatoes for 45 minutes."

           - Cooking_creation: "Michelle baked her mother a cake for her birthday."

           - Absorb_heat: "The potatoes have to bake for more than 30 minutes."

        These constitute three different LUs, with different
        definitions.

        Multiword expressions such as "given name" and hyphenated words
        like "shut-eye" can also be LUs. Idiomatic phrases such as
        "middle of nowhere" and "give the slip (to)" are also defined as
        LUs in the appropriate frames ("Isolated_places" and "Evading",
        respectively), and their internal structure is not analyzed.

        Framenet provides multiple annotated examples of each sense of a
        word (i.e. each LU).  Moreover, the set of examples
        (approximately 20 per LU) illustrates all of the combinatorial
        possibilities of the lexical unit.

        Each LU is linked to a Frame, and hence to the other words which
        evoke that Frame. This makes the FrameNet database similar to a
        thesaurus, grouping together semantically similar words.

        In the simplest case, frame-evoking words are verbs such as
        "fried" in:

           "Matilde fried the catfish in a heavy iron skillet."

        Sometimes event nouns may evoke a Frame. For example,
        "reduction" evokes "Cause_change_of_scalar_position" in:

           "...the reduction of debt levels to $665 million from $2.6 billion."

        Adjectives may also evoke a Frame. For example, "asleep" may
        evoke the "Sleep" frame as in:

           "They were asleep for hours."

        Many common nouns, such as artifacts like "hat" or "tower",
        typically serve as dependents rather than clearly evoking their
        own frames.

        :param name: A regular expression pattern used to search the LU
            names. Note that LU names take the form of a dotted
            string (e.g. "run.v" or "a little.adv") in which a
            lemma precedes the "." and a POS follows the
            dot. The lemma may be composed of a single lexeme
            (e.g. "run") or of multiple lexemes (e.g. "a
            little"). If 'name' is not given, then all LUs will
            be returned.

            The valid POSes are:

                   v    - verb
                   n    - noun
                   a    - adjective
                   adv  - adverb
                   prep - preposition
                   num  - numbers
                   intj - interjection
                   art  - article
                   c    - conjunction
                   scon - subordinating conjunction

        :type name: str
        :type frame: str or int or frame
        :return: A list of selected (or all) lexical units
        :rtype: list of LU objects (dicts). See the lu() function for info
          about the specifics of LU objects.

        """
        if not self._lu_idx:
            self._buildluindex()

        if name is not None:  # match LUs, then restrict by frame
            result = PrettyList(
                self.lu(luID) for luID, luName in self.lu_ids_and_names(name).items()
            )
            if frame is not None:
                if isinstance(frame, int):
                    frameIDs = {frame}
                elif isinstance(frame, str):
                    frameIDs = {f.ID for f in self.frames(frame)}
                else:
                    frameIDs = {frame.ID}
                result = PrettyList(lu for lu in result if lu.frame.ID in frameIDs)
        elif frame is not None:  # all LUs in matching frames
            if isinstance(frame, int):
                frames = [self.frame(frame)]
            elif isinstance(frame, str):
                frames = self.frames(frame)
            else:
                frames = [frame]
            result = PrettyLazyIteratorList(
                iter(LazyConcatenation(list(f.lexUnit.values()) for f in frames))
            )
        else:  # all LUs
            luIDs = [
                luID
                for luID, lu in self._lu_idx.items()
                if lu.status not in self._bad_statuses
            ]
            result = PrettyLazyMap(self.lu, luIDs)
        return result

    def lu_ids_and_names(self, name=None):
        """
        Uses the LU index, which is much faster than looking up each LU definition
        if only the names and IDs are needed.
        """
        if not self._lu_idx:
            self._buildluindex()
        return {
            luID: luinfo.name
            for luID, luinfo in self._lu_idx.items()
            if luinfo.status not in self._bad_statuses
            and (name is None or re.search(name, luinfo.name) is not None)
        }

    def docs_metadata(self, name=None):
        """
        Return an index of the annotated documents in Framenet.

        Details for a specific annotated document can be obtained using this
        class's doc() function and pass it the value of the 'ID' field.

        >>> from nltk.corpus import framenet as fn
        >>> len(fn.docs()) in (78, 107) # FN 1.5 and 1.7, resp.
        True
        >>> set([x.corpname for x in fn.docs_metadata()])>=set(['ANC', 'KBEval', \
                    'LUCorpus-v0.3', 'Miscellaneous', 'NTI', 'PropBank'])
        True

        :param name: A regular expression pattern used to search the
            file name of each annotated document. The document's
            file name contains the name of the corpus that the
            document is from, followed by two underscores "__"
            followed by the document name. So, for example, the
            file name "LUCorpus-v0.3__20000410_nyt-NEW.xml" is
            from the corpus named "LUCorpus-v0.3" and the
            document name is "20000410_nyt-NEW.xml".
        :type name: str
        :return: A list of selected (or all) annotated documents
        :rtype: list of dicts, where each dict object contains the following
                keys:

                - 'name'
                - 'ID'
                - 'corpid'
                - 'corpname'
                - 'description'
                - 'filename'
        """
        try:
            ftlist = PrettyList(self._fulltext_idx.values())
        except AttributeError:
            self._buildcorpusindex()
            ftlist = PrettyList(self._fulltext_idx.values())

        if name is None:
            return ftlist
        else:
            return PrettyList(
                x for x in ftlist if re.search(name, x["filename"]) is not None
            )

    def docs(self, name=None):
        """
        Return a list of the annotated full-text documents in FrameNet,
        optionally filtered by a regex to be matched against the document name.
        """
        return PrettyLazyMap((lambda x: self.doc(x.ID)), self.docs_metadata(name))

    def sents(self, exemplars=True, full_text=True):
        """
        Annotated sentences matching the specified criteria.
        """
        if exemplars:
            if full_text:
                return self.exemplars() + self.ft_sents()
            else:
                return self.exemplars()
        elif full_text:
            return self.ft_sents()

    def annotations(self, luNamePattern=None, exemplars=True, full_text=True):
        """
        Frame annotation sets matching the specified criteria.
        """

        if exemplars:
            epart = PrettyLazyIteratorList(
                sent.frameAnnotation for sent in self.exemplars(luNamePattern)
            )
        else:
            epart = []

        if full_text:
            if luNamePattern is not None:
                matchedLUIDs = set(self.lu_ids_and_names(luNamePattern).keys())
            ftpart = PrettyLazyIteratorList(
                aset
                for sent in self.ft_sents()
                for aset in sent.annotationSet[1:]
                if luNamePattern is None or aset.get("luID", "CXN_ASET") in matchedLUIDs
            )
        else:
            ftpart = []

        if exemplars:
            if full_text:
                return epart + ftpart
            else:
                return epart
        elif full_text:
            return ftpart

    def exemplars(self, luNamePattern=None, frame=None, fe=None, fe2=None):
        """
        Lexicographic exemplar sentences, optionally filtered by LU name and/or 1-2 FEs that
        are realized overtly. 'frame' may be a name pattern, frame ID, or frame instance.
        'fe' may be a name pattern or FE instance; if specified, 'fe2' may also
        be specified to retrieve sentences with both overt FEs (in either order).
        """
        if fe is None and fe2 is not None:
            raise FramenetError("exemplars(..., fe=None, fe2=<value>) is not allowed")
        elif fe is not None and fe2 is not None:
            if not isinstance(fe2, str):
                if isinstance(fe, str):
                    # fe2 is specific to a particular frame. swap fe and fe2 so fe is always used to determine the frame.
                    fe, fe2 = fe2, fe
                elif fe.frame is not fe2.frame:  # ensure frames match
                    raise FramenetError(
                        "exemplars() call with inconsistent `fe` and `fe2` specification (frames must match)"
                    )
        if frame is None and fe is not None and not isinstance(fe, str):
            frame = fe.frame

        # narrow down to frames matching criteria

        lusByFrame = defaultdict(
            list
        )  # frame name -> matching LUs, if luNamePattern is specified
        if frame is not None or luNamePattern is not None:
            if frame is None or isinstance(frame, str):
                if luNamePattern is not None:
                    frames = set()
                    for lu in self.lus(luNamePattern, frame=frame):
                        frames.add(lu.frame.ID)
                        lusByFrame[lu.frame.name].append(lu)
                    frames = LazyMap(self.frame, list(frames))
                else:
                    frames = self.frames(frame)
            else:
                if isinstance(frame, int):
                    frames = [self.frame(frame)]
                else:  # frame object
                    frames = [frame]

                if luNamePattern is not None:
                    lusByFrame = {frame.name: self.lus(luNamePattern, frame=frame)}

            if fe is not None:  # narrow to frames that define this FE
                if isinstance(fe, str):
                    frames = PrettyLazyIteratorList(
                        f
                        for f in frames
                        if fe in f.FE
                        or any(re.search(fe, ffe, re.I) for ffe in f.FE.keys())
                    )
                else:
                    if fe.frame not in frames:
                        raise FramenetError(
                            "exemplars() call with inconsistent `frame` and `fe` specification"
                        )
                    frames = [fe.frame]

                if fe2 is not None:  # narrow to frames that ALSO define this FE
                    if isinstance(fe2, str):
                        frames = PrettyLazyIteratorList(
                            f
                            for f in frames
                            if fe2 in f.FE
                            or any(re.search(fe2, ffe, re.I) for ffe in f.FE.keys())
                        )
                    # else we already narrowed it to a single frame
        else:  # frame, luNamePattern are None. fe, fe2 are None or strings
            if fe is not None:
                frames = {ffe.frame.ID for ffe in self.fes(fe)}
                if fe2 is not None:
                    frames2 = {ffe.frame.ID for ffe in self.fes(fe2)}
                    frames = frames & frames2
                frames = LazyMap(self.frame, list(frames))
            else:
                frames = self.frames()

        # we've narrowed down 'frames'
        # now get exemplars for relevant LUs in those frames

        def _matching_exs():
            for f in frames:
                fes = fes2 = None  # FEs of interest
                if fe is not None:
                    fes = (
                        {ffe for ffe in f.FE.keys() if re.search(fe, ffe, re.I)}
                        if isinstance(fe, str)
                        else {fe.name}
                    )
                    if fe2 is not None:
                        fes2 = (
                            {ffe for ffe in f.FE.keys() if re.search(fe2, ffe, re.I)}
                            if isinstance(fe2, str)
                            else {fe2.name}
                        )

                for lu in (
                    lusByFrame[f.name]
                    if luNamePattern is not None
                    else f.lexUnit.values()
                ):
                    for ex in lu.exemplars:
                        if (fes is None or self._exemplar_of_fes(ex, fes)) and (
                            fes2 is None or self._exemplar_of_fes(ex, fes2)
                        ):
                            yield ex

        return PrettyLazyIteratorList(_matching_exs())

    def _exemplar_of_fes(self, ex, fes=None):
        """
        Given an exemplar sentence and a set of FE names, return the subset of FE names
        that are realized overtly in the sentence on the FE, FE2, or FE3 layer.

        If 'fes' is None, returns all overt FE names.
        """
        overtNames = set(list(zip(*ex.FE[0]))[2]) if ex.FE[0] else set()
        if "FE2" in ex:
            overtNames |= set(list(zip(*ex.FE2[0]))[2]) if ex.FE2[0] else set()
            if "FE3" in ex:
                overtNames |= set(list(zip(*ex.FE3[0]))[2]) if ex.FE3[0] else set()
        return overtNames & fes if fes is not None else overtNames

    def ft_sents(self, docNamePattern=None):
        """
        Full-text annotation sentences, optionally filtered by document name.
        """
        return PrettyLazyIteratorList(
            sent for d in self.docs(docNamePattern) for sent in d.sentence
        )

    def frame_relation_types(self):
        """
        Obtain a list of frame relation types.

        >>> from nltk.corpus import framenet as fn
        >>> frts = sorted(fn.frame_relation_types(), key=itemgetter('ID'))
        >>> isinstance(frts, list)
        True
        >>> len(frts) in (9, 10)    # FN 1.5 and 1.7, resp.
        True
        >>> PrettyDict(frts[0], breakLines=True)
        {'ID': 1,
         '_type': 'framerelationtype',
         'frameRelations': [<Parent=Event -- Inheritance -> Child=Change_of_consistency>, <Parent=Event -- Inheritance -> Child=Rotting>, ...],
         'name': 'Inheritance',
         'subFrameName': 'Child',
         'superFrameName': 'Parent'}

        :return: A list of all of the frame relation types in framenet
        :rtype: list(dict)
        """
        if not self._freltyp_idx:
            self._buildrelationindex()
        return self._freltyp_idx.values()

    def frame_relations(self, frame=None, frame2=None, type=None):
        """
        :param frame: (optional) frame object, name, or ID; only relations involving
            this frame will be returned
        :param frame2: (optional; 'frame' must be a different frame) only show relations
            between the two specified frames, in either direction
        :param type: (optional) frame relation type (name or object); show only relations
            of this type
        :type frame: int or str or AttrDict
        :return: A list of all of the frame relations in framenet
        :rtype: list(dict)

        >>> from nltk.corpus import framenet as fn
        >>> frels = fn.frame_relations()
        >>> isinstance(frels, list)
        True
        >>> len(frels) in (1676, 2070)  # FN 1.5 and 1.7, resp.
        True
        >>> PrettyList(fn.frame_relations('Cooking_creation'), maxReprSize=0, breakLines=True)
        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>,
         <Parent=Apply_heat -- Using -> Child=Cooking_creation>,
         <MainEntry=Apply_heat -- See_also -> ReferringEntry=Cooking_creation>]
        >>> PrettyList(fn.frame_relations(274), breakLines=True)
        [<Parent=Avoiding -- Inheritance -> Child=Dodging>,
         <Parent=Avoiding -- Inheritance -> Child=Evading>, ...]
        >>> PrettyList(fn.frame_relations(fn.frame('Cooking_creation')), breakLines=True)
        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>,
         <Parent=Apply_heat -- Using -> Child=Cooking_creation>, ...]
        >>> PrettyList(fn.frame_relations('Cooking_creation', type='Inheritance'))
        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>]
        >>> PrettyList(fn.frame_relations('Cooking_creation', 'Apply_heat'), breakLines=True) # doctest: +NORMALIZE_WHITESPACE
        [<Parent=Apply_heat -- Using -> Child=Cooking_creation>,
        <MainEntry=Apply_heat -- See_also -> ReferringEntry=Cooking_creation>]
        """
        relation_type = type

        if not self._frel_idx:
            self._buildrelationindex()

        rels = None

        if relation_type is not None:
            if not isinstance(relation_type, dict):
                type = [rt for rt in self.frame_relation_types() if rt.name == type][0]
                assert isinstance(type, dict)

        # lookup by 'frame'
        if frame is not None:
            if isinstance(frame, dict) and "frameRelations" in frame:
                rels = PrettyList(frame.frameRelations)
            else:
                if not isinstance(frame, int):
                    if isinstance(frame, dict):
                        frame = frame.ID
                    else:
                        frame = self.frame_by_name(frame).ID
                rels = [self._frel_idx[frelID] for frelID in self._frel_f_idx[frame]]

            # filter by 'type'
            if type is not None:
                rels = [rel for rel in rels if rel.type is type]
        elif type is not None:
            # lookup by 'type'
            rels = type.frameRelations
        else:
            rels = self._frel_idx.values()

        # filter by 'frame2'
        if frame2 is not None:
            if frame is None:
                raise FramenetError(
                    "frame_relations(frame=None, frame2=<value>) is not allowed"
                )
            if not isinstance(frame2, int):
                if isinstance(frame2, dict):
                    frame2 = frame2.ID
                else:
                    frame2 = self.frame_by_name(frame2).ID
            if frame == frame2:
                raise FramenetError(
                    "The two frame arguments to frame_relations() must be different frames"
                )
            rels = [
                rel
                for rel in rels
                if rel.superFrame.ID == frame2 or rel.subFrame.ID == frame2
            ]

        return PrettyList(
            sorted(
                rels,
                key=lambda frel: (frel.type.ID, frel.superFrameName, frel.subFrameName),
            )
        )

    def fe_relations(self):
        """
        Obtain a list of frame element relations.

        >>> from nltk.corpus import framenet as fn
        >>> ferels = fn.fe_relations()
        >>> isinstance(ferels, list)
        True
        >>> len(ferels) in (10020, 12393)   # FN 1.5 and 1.7, resp.
        True
        >>> PrettyDict(ferels[0], breakLines=True) # doctest: +NORMALIZE_WHITESPACE
        {'ID': 14642,
        '_type': 'ferelation',
        'frameRelation': <Parent=Abounding_with -- Inheritance -> Child=Lively_place>,
        'subFE': <fe ID=11370 name=Degree>,
        'subFEName': 'Degree',
        'subFrame': <frame ID=1904 name=Lively_place>,
        'subID': 11370,
        'supID': 2271,
        'superFE': <fe ID=2271 name=Degree>,
        'superFEName': 'Degree',
        'superFrame': <frame ID=262 name=Abounding_with>,
        'type': <framerelationtype ID=1 name=Inheritance>}

        :return: A list of all of the frame element relations in framenet
        :rtype: list(dict)
        """
        if not self._ferel_idx:
            self._buildrelationindex()
        return PrettyList(
            sorted(
                self._ferel_idx.values(),
                key=lambda ferel: (
                    ferel.type.ID,
                    ferel.frameRelation.superFrameName,
                    ferel.superFEName,
                    ferel.frameRelation.subFrameName,
                    ferel.subFEName,
                ),
            )
        )

    def semtypes(self):
        """
        Obtain a list of semantic types.

        >>> from nltk.corpus import framenet as fn
        >>> stypes = fn.semtypes()
        >>> len(stypes) in (73, 109) # FN 1.5 and 1.7, resp.
        True
        >>> sorted(stypes[0].keys())
        ['ID', '_type', 'abbrev', 'definition', 'definitionMarkup', 'name', 'rootType', 'subTypes', 'superType']

        :return: A list of all of the semantic types in framenet
        :rtype: list(dict)
        """
        if not self._semtypes:
            self._loadsemtypes()
        return PrettyList(
            self._semtypes[i] for i in self._semtypes if isinstance(i, int)
        )

    def _load_xml_attributes(self, d, elt):
        """
        Extracts a subset of the attributes from the given element and
        returns them in a dictionary.

        :param d: A dictionary in which to store the attributes.
        :type d: dict
        :param elt: An ElementTree Element
        :type elt: Element
        :return: Returns the input dict ``d`` possibly including attributes from ``elt``
        :rtype: dict
        """

        d = type(d)(d)

        try:
            attr_dict = elt.attrib
        except AttributeError:
            return d

        if attr_dict is None:
            return d

        # Ignore these attributes when loading attributes from an xml node
        ignore_attrs = [  #'cBy', 'cDate', 'mDate', # <-- annotation metadata that could be of interest
            "xsi",
            "schemaLocation",
            "xmlns",
            "bgColor",
            "fgColor",
        ]

        for attr in attr_dict:

            if any(attr.endswith(x) for x in ignore_attrs):
                continue

            val = attr_dict[attr]
            if val.isdigit():
                d[attr] = int(val)
            else:
                d[attr] = val

        return d

    def _strip_tags(self, data):
        """
        Gets rid of all tags and newline characters from the given input

        :return: A cleaned-up version of the input string
        :rtype: str
        """

        try:
            r"""
            # Look for boundary issues in markup. (Sometimes FEs are pluralized in definitions.)
            m = re.search(r'\w[<][^/]|[<][/][^>]+[>](s\w|[a-rt-z0-9])', data)
            if m:
                print('Markup boundary:', data[max(0,m.start(0)-10):m.end(0)+10].replace('\n',' '), file=sys.stderr)
            """

            data = data.replace("<t>", "")
            data = data.replace("</t>", "")
            data = re.sub('<fex name="[^"]+">', "", data)
            data = data.replace("</fex>", "")
            data = data.replace("<fen>", "")
            data = data.replace("</fen>", "")
            data = data.replace("<m>", "")
            data = data.replace("</m>", "")
            data = data.replace("<ment>", "")
            data = data.replace("</ment>", "")
            data = data.replace("<ex>", "'")
            data = data.replace("</ex>", "'")
            data = data.replace("<gov>", "")
            data = data.replace("</gov>", "")
            data = data.replace("<x>", "")
            data = data.replace("</x>", "")

            # Get rid of <def-root> and </def-root> tags
            data = data.replace("<def-root>", "")
            data = data.replace("</def-root>", "")

            data = data.replace("\n", " ")
        except AttributeError:
            pass

        return data

    def _handle_elt(self, elt, tagspec=None):
        """Extracts and returns the attributes of the given element"""
        return self._load_xml_attributes(AttrDict(), elt)

    def _handle_fulltextindex_elt(self, elt, tagspec=None):
        """
        Extracts corpus/document info from the fulltextIndex.xml file.

        Note that this function "flattens" the information contained
        in each of the "corpus" elements, so that each "document"
        element will contain attributes for the corpus and
        corpusid. Also, each of the "document" items will contain a
        new attribute called "filename" that is the base file name of
        the xml file for the document in the "fulltext" subdir of the
        Framenet corpus.
        """
        ftinfo = self._load_xml_attributes(AttrDict(), elt)
        corpname = ftinfo.name
        corpid = ftinfo.ID
        retlist = []
        for sub in elt:
            if sub.tag.endswith("document"):
                doc = self._load_xml_attributes(AttrDict(), sub)
                if "name" in doc:
                    docname = doc.name
                else:
                    docname = doc.description
                doc.filename = f"{corpname}__{docname}.xml"
                doc.URL = (
                    self._fnweb_url + "/" + self._fulltext_dir + "/" + doc.filename
                )
                doc.corpname = corpname
                doc.corpid = corpid
                retlist.append(doc)

        return retlist

    def _handle_frame_elt(self, elt, ignorekeys=[]):
        """Load the info for a Frame from a frame xml file"""
        frinfo = self._load_xml_attributes(AttrDict(), elt)

        frinfo["_type"] = "frame"
        frinfo["definition"] = ""
        frinfo["definitionMarkup"] = ""
        frinfo["FE"] = PrettyDict()
        frinfo["FEcoreSets"] = []
        frinfo["lexUnit"] = PrettyDict()
        frinfo["semTypes"] = []
        for k in ignorekeys:
            if k in frinfo:
                del frinfo[k]

        for sub in elt:
            if sub.tag.endswith("definition") and "definition" not in ignorekeys:
                frinfo["definitionMarkup"] = sub.text
                frinfo["definition"] = self._strip_tags(sub.text)
            elif sub.tag.endswith("FE") and "FE" not in ignorekeys:
                feinfo = self._handle_fe_elt(sub)
                frinfo["FE"][feinfo.name] = feinfo
                feinfo["frame"] = frinfo  # backpointer
            elif sub.tag.endswith("FEcoreSet") and "FEcoreSet" not in ignorekeys:
                coreset = self._handle_fecoreset_elt(sub)
                # assumes all FEs have been loaded before coresets
                frinfo["FEcoreSets"].append(
                    PrettyList(frinfo["FE"][fe.name] for fe in coreset)
                )
            elif sub.tag.endswith("lexUnit") and "lexUnit" not in ignorekeys:
                luentry = self._handle_framelexunit_elt(sub)
                if luentry["status"] in self._bad_statuses:
                    # problematic LU entry; ignore it
                    continue
                luentry["frame"] = frinfo
                luentry["URL"] = (
                    self._fnweb_url
                    + "/"
                    + self._lu_dir
                    + "/"
                    + "lu{}.xml".format(luentry["ID"])
                )
                luentry["subCorpus"] = Future(
                    (lambda lu: lambda: self._lu_file(lu).subCorpus)(luentry)
                )
                luentry["exemplars"] = Future(
                    (lambda lu: lambda: self._lu_file(lu).exemplars)(luentry)
                )
                frinfo["lexUnit"][luentry.name] = luentry
                if not self._lu_idx:
                    self._buildluindex()
                self._lu_idx[luentry.ID] = luentry
            elif sub.tag.endswith("semType") and "semTypes" not in ignorekeys:
                semtypeinfo = self._load_xml_attributes(AttrDict(), sub)
                frinfo["semTypes"].append(self.semtype(semtypeinfo.ID))

        frinfo["frameRelations"] = self.frame_relations(frame=frinfo)

        # resolve 'requires' and 'excludes' links between FEs of this frame
        for fe in frinfo.FE.values():
            if fe.requiresFE:
                name, ID = fe.requiresFE.name, fe.requiresFE.ID
                fe.requiresFE = frinfo.FE[name]
                assert fe.requiresFE.ID == ID
            if fe.excludesFE:
                name, ID = fe.excludesFE.name, fe.excludesFE.ID
                fe.excludesFE = frinfo.FE[name]
                assert fe.excludesFE.ID == ID

        return frinfo

    def _handle_fecoreset_elt(self, elt):
        """Load fe coreset info from xml."""
        info = self._load_xml_attributes(AttrDict(), elt)
        tmp = []
        for sub in elt:
            tmp.append(self._load_xml_attributes(AttrDict(), sub))

        return tmp

    def _handle_framerelationtype_elt(self, elt, *args):
        """Load frame-relation element and its child fe-relation elements from frRelation.xml."""
        info = self._load_xml_attributes(AttrDict(), elt)
        info["_type"] = "framerelationtype"
        info["frameRelations"] = PrettyList()

        for sub in elt:
            if sub.tag.endswith("frameRelation"):
                frel = self._handle_framerelation_elt(sub)
                frel["type"] = info  # backpointer
                for ferel in frel.feRelations:
                    ferel["type"] = info
                info["frameRelations"].append(frel)

        return info

    def _handle_framerelation_elt(self, elt):
        """Load frame-relation element and its child fe-relation elements from frRelation.xml."""
        info = self._load_xml_attributes(AttrDict(), elt)
        assert info["superFrameName"] != info["subFrameName"], (elt, info)
        info["_type"] = "framerelation"
        info["feRelations"] = PrettyList()

        for sub in elt:
            if sub.tag.endswith("FERelation"):
                ferel = self._handle_elt(sub)
                ferel["_type"] = "ferelation"
                ferel["frameRelation"] = info  # backpointer
                info["feRelations"].append(ferel)

        return info

    def _handle_fulltextannotation_elt(self, elt):
        """Load full annotation info for a document from its xml
        file. The main element (fullTextAnnotation) contains a 'header'
        element (which we ignore here) and a bunch of 'sentence'
        elements."""
        info = AttrDict()
        info["_type"] = "fulltext_annotation"
        info["sentence"] = []

        for sub in elt:
            if sub.tag.endswith("header"):
                continue  # not used
            elif sub.tag.endswith("sentence"):
                s = self._handle_fulltext_sentence_elt(sub)
                s.doc = info
                info["sentence"].append(s)

        return info

    def _handle_fulltext_sentence_elt(self, elt):
        """Load information from the given 'sentence' element. Each
        'sentence' element contains a "text" and "annotationSet" sub
        elements."""
        info = self._load_xml_attributes(AttrDict(), elt)
        info["_type"] = "fulltext_sentence"
        info["annotationSet"] = []
        info["targets"] = []
        target_spans = set()
        info["_ascii"] = types.MethodType(
            _annotation_ascii, info
        )  # attach a method for this instance
        info["text"] = ""

        for sub in elt:
            if sub.tag.endswith("text"):
                info["text"] = self._strip_tags(sub.text)
            elif sub.tag.endswith("annotationSet"):
                a = self._handle_fulltextannotationset_elt(
                    sub, is_pos=(len(info["annotationSet"]) == 0)
                )
                if "cxnID" in a:  # ignoring construction annotations for now
                    continue
                a.sent = info
                a.text = info.text
                info["annotationSet"].append(a)
                if "Target" in a:
                    for tspan in a.Target:
                        if tspan in target_spans:
                            self._warn(
                                'Duplicate target span "{}"'.format(
                                    info.text[slice(*tspan)]
                                ),
                                tspan,
                                "in sentence",
                                info["ID"],
                                info.text,
                            )
                            # this can happen in cases like "chemical and biological weapons"
                            # being annotated as "chemical weapons" and "biological weapons"
                        else:
                            target_spans.add(tspan)
                    info["targets"].append((a.Target, a.luName, a.frameName))

        assert info["annotationSet"][0].status == "UNANN"
        info["POS"] = info["annotationSet"][0].POS
        info["POS_tagset"] = info["annotationSet"][0].POS_tagset
        return info

    def _handle_fulltextannotationset_elt(self, elt, is_pos=False):
        """Load information from the given 'annotationSet' element. Each
        'annotationSet' contains several "layer" elements."""

        info = self._handle_luannotationset_elt(elt, is_pos=is_pos)
        if not is_pos:
            info["_type"] = "fulltext_annotationset"
            if "cxnID" not in info:  # ignoring construction annotations for now
                info["LU"] = self.lu(
                    info.luID,
                    luName=info.luName,
                    frameID=info.frameID,
                    frameName=info.frameName,
                )
                info["frame"] = info.LU.frame
        return info

    def _handle_fulltextlayer_elt(self, elt):
        """Load information from the given 'layer' element. Each
        'layer' contains several "label" elements."""
        info = self._load_xml_attributes(AttrDict(), elt)
        info["_type"] = "layer"
        info["label"] = []

        for sub in elt:
            if sub.tag.endswith("label"):
                l = self._load_xml_attributes(AttrDict(), sub)
                info["label"].append(l)

        return info

    def _handle_framelexunit_elt(self, elt):
        """Load the lexical unit info from an xml element in a frame's xml file."""
        luinfo = AttrDict()
        luinfo["_type"] = "lu"
        luinfo = self._load_xml_attributes(luinfo, elt)
        luinfo["definition"] = ""
        luinfo["definitionMarkup"] = ""
        luinfo["sentenceCount"] = PrettyDict()
        luinfo["lexemes"] = PrettyList()  # multiword LUs have multiple lexemes
        luinfo["semTypes"] = PrettyList()  # an LU can have multiple semtypes

        for sub in elt:
            if sub.tag.endswith("definition"):
                luinfo["definitionMarkup"] = sub.text
                luinfo["definition"] = self._strip_tags(sub.text)
            elif sub.tag.endswith("sentenceCount"):
                luinfo["sentenceCount"] = self._load_xml_attributes(PrettyDict(), sub)
            elif sub.tag.endswith("lexeme"):
                lexemeinfo = self._load_xml_attributes(PrettyDict(), sub)
                if not isinstance(lexemeinfo.name, str):
                    # some lexeme names are ints by default: e.g.,
                    # thousand.num has lexeme with name="1000"
                    lexemeinfo.name = str(lexemeinfo.name)
                luinfo["lexemes"].append(lexemeinfo)
            elif sub.tag.endswith("semType"):
                semtypeinfo = self._load_xml_attributes(PrettyDict(), sub)
                luinfo["semTypes"].append(self.semtype(semtypeinfo.ID))

        # sort lexemes by 'order' attribute
        # otherwise, e.g., 'write down.v' may have lexemes in wrong order
        luinfo["lexemes"].sort(key=lambda x: x.order)

        return luinfo

    def _handle_lexunit_elt(self, elt, ignorekeys):
        """
        Load full info for a lexical unit from its xml file.
        This should only be called when accessing corpus annotations
        (which are not included in frame files).
        """
        luinfo = self._load_xml_attributes(AttrDict(), elt)
        luinfo["_type"] = "lu"
        luinfo["definition"] = ""
        luinfo["definitionMarkup"] = ""
        luinfo["subCorpus"] = PrettyList()
        luinfo["lexemes"] = PrettyList()  # multiword LUs have multiple lexemes
        luinfo["semTypes"] = PrettyList()  # an LU can have multiple semtypes
        for k in ignorekeys:
            if k in luinfo:
                del luinfo[k]

        for sub in elt:
            if sub.tag.endswith("header"):
                continue  # not used
            elif sub.tag.endswith("valences"):
                continue  # not used
            elif sub.tag.endswith("definition") and "definition" not in ignorekeys:
                luinfo["definitionMarkup"] = sub.text
                luinfo["definition"] = self._strip_tags(sub.text)
            elif sub.tag.endswith("subCorpus") and "subCorpus" not in ignorekeys:
                sc = self._handle_lusubcorpus_elt(sub)
                if sc is not None:
                    luinfo["subCorpus"].append(sc)
            elif sub.tag.endswith("lexeme") and "lexeme" not in ignorekeys:
                luinfo["lexemes"].append(self._load_xml_attributes(PrettyDict(), sub))
            elif sub.tag.endswith("semType") and "semType" not in ignorekeys:
                semtypeinfo = self._load_xml_attributes(AttrDict(), sub)
                luinfo["semTypes"].append(self.semtype(semtypeinfo.ID))

        return luinfo

    def _handle_lusubcorpus_elt(self, elt):
        """Load a subcorpus of a lexical unit from the given xml."""
        sc = AttrDict()
        try:
            sc["name"] = elt.get("name")
        except AttributeError:
            return None
        sc["_type"] = "lusubcorpus"
        sc["sentence"] = []

        for sub in elt:
            if sub.tag.endswith("sentence"):
                s = self._handle_lusentence_elt(sub)
                if s is not None:
                    sc["sentence"].append(s)

        return sc

    def _handle_lusentence_elt(self, elt):
        """Load a sentence from a subcorpus of an LU from xml."""
        info = self._load_xml_attributes(AttrDict(), elt)
        info["_type"] = "lusentence"
        info["annotationSet"] = []
        info["_ascii"] = types.MethodType(
            _annotation_ascii, info
        )  # attach a method for this instance
        for sub in elt:
            if sub.tag.endswith("text"):
                info["text"] = self._strip_tags(sub.text)
            elif sub.tag.endswith("annotationSet"):
                annset = self._handle_luannotationset_elt(
                    sub, is_pos=(len(info["annotationSet"]) == 0)
                )
                if annset is not None:
                    assert annset.status == "UNANN" or "FE" in annset, annset
                    if annset.status != "UNANN":
                        info["frameAnnotation"] = annset
                    # copy layer info up to current level
                    for k in (
                        "Target",
                        "FE",
                        "FE2",
                        "FE3",
                        "GF",
                        "PT",
                        "POS",
                        "POS_tagset",
                        "Other",
                        "Sent",
                        "Verb",
                        "Noun",
                        "Adj",
                        "Adv",
                        "Prep",
                        "Scon",
                        "Art",
                    ):
                        if k in annset:
                            info[k] = annset[k]
                    info["annotationSet"].append(annset)
                    annset["sent"] = info
                    annset["text"] = info.text
        return info

    def _handle_luannotationset_elt(self, elt, is_pos=False):
        """Load an annotation set from a sentence in an subcorpus of an LU"""
        info = self._load_xml_attributes(AttrDict(), elt)
        info["_type"] = "posannotationset" if is_pos else "luannotationset"
        info["layer"] = []
        info["_ascii"] = types.MethodType(
            _annotation_ascii, info
        )  # attach a method for this instance

        if "cxnID" in info:  # ignoring construction annotations for now.
            return info

        for sub in elt:
            if sub.tag.endswith("layer"):
                l = self._handle_lulayer_elt(sub)
                if l is not None:
                    overt = []
                    ni = {}  # null instantiations

                    info["layer"].append(l)
                    for lbl in l.label:
                        if "start" in lbl:
                            thespan = (lbl.start, lbl.end + 1, lbl.name)
                            if l.name not in (
                                "Sent",
                                "Other",
                            ):  # 'Sent' and 'Other' layers sometimes contain accidental duplicate spans
                                assert thespan not in overt, (info.ID, l.name, thespan)
                            overt.append(thespan)
                        else:  # null instantiation
                            if lbl.name in ni:
                                self._warn(
                                    "FE with multiple NI entries:",
                                    lbl.name,
                                    ni[lbl.name],
                                    lbl.itype,
                                )
                            else:
                                ni[lbl.name] = lbl.itype
                    overt = sorted(overt)

                    if l.name == "Target":
                        if not overt:
                            self._warn(
                                "Skipping empty Target layer in annotation set ID={}".format(
                                    info.ID
                                )
                            )
                            continue
                        assert all(lblname == "Target" for i, j, lblname in overt)
                        if "Target" in info:
                            self._warn(
                                "Annotation set {} has multiple Target layers".format(
                                    info.ID
                                )
                            )
                        else:
                            info["Target"] = [(i, j) for (i, j, _) in overt]
                    elif l.name == "FE":
                        if l.rank == 1:
                            assert "FE" not in info
                            info["FE"] = (overt, ni)
                            # assert False,info
                        else:
                            # sometimes there are 3 FE layers! e.g. Change_position_on_a_scale.fall.v
                            assert 2 <= l.rank <= 3, l.rank
                            k = "FE" + str(l.rank)
                            assert k not in info
                            info[k] = (overt, ni)
                    elif l.name in ("GF", "PT"):
                        assert l.rank == 1
                        info[l.name] = overt
                    elif l.name in ("BNC", "PENN"):
                        assert l.rank == 1
                        info["POS"] = overt
                        info["POS_tagset"] = l.name
                    else:
                        if is_pos:
                            if l.name not in ("NER", "WSL"):
                                self._warn(
                                    "Unexpected layer in sentence annotationset:",
                                    l.name,
                                )
                        else:
                            if l.name not in (
                                "Sent",
                                "Verb",
                                "Noun",
                                "Adj",
                                "Adv",
                                "Prep",
                                "Scon",
                                "Art",
                                "Other",
                            ):
                                self._warn(
                                    "Unexpected layer in frame annotationset:", l.name
                                )
                        info[l.name] = overt
        if not is_pos and "cxnID" not in info:
            if "Target" not in info:
                self._warn(f"Missing target in annotation set ID={info.ID}")
            assert "FE" in info
            if "FE3" in info:
                assert "FE2" in info

        return info

    def _handle_lulayer_elt(self, elt):
        """Load a layer from an annotation set"""
        layer = self._load_xml_attributes(AttrDict(), elt)
        layer["_type"] = "lulayer"
        layer["label"] = []

        for sub in elt:
            if sub.tag.endswith("label"):
                l = self._load_xml_attributes(AttrDict(), sub)
                if l is not None:
                    layer["label"].append(l)
        return layer

    def _handle_fe_elt(self, elt):
        feinfo = self._load_xml_attributes(AttrDict(), elt)
        feinfo["_type"] = "fe"
        feinfo["definition"] = ""
        feinfo["definitionMarkup"] = ""
        feinfo["semType"] = None
        feinfo["requiresFE"] = None
        feinfo["excludesFE"] = None
        for sub in elt:
            if sub.tag.endswith("definition"):
                feinfo["definitionMarkup"] = sub.text
                feinfo["definition"] = self._strip_tags(sub.text)
            elif sub.tag.endswith("semType"):
                stinfo = self._load_xml_attributes(AttrDict(), sub)
                feinfo["semType"] = self.semtype(stinfo.ID)
            elif sub.tag.endswith("requiresFE"):
                feinfo["requiresFE"] = self._load_xml_attributes(AttrDict(), sub)
            elif sub.tag.endswith("excludesFE"):
                feinfo["excludesFE"] = self._load_xml_attributes(AttrDict(), sub)

        return feinfo

    def _handle_semtype_elt(self, elt, tagspec=None):
        semt = self._load_xml_attributes(AttrDict(), elt)
        semt["_type"] = "semtype"
        semt["superType"] = None
        semt["subTypes"] = PrettyList()
        for sub in elt:
            if sub.text is not None:
                semt["definitionMarkup"] = sub.text
                semt["definition"] = self._strip_tags(sub.text)
            else:
                supertypeinfo = self._load_xml_attributes(AttrDict(), sub)
                semt["superType"] = supertypeinfo
                # the supertype may not have been loaded yet

        return semt


#
# Demo
#
def demo():
    from nltk.corpus import framenet as fn

    #
    # It is not necessary to explicitly build the indexes by calling
    # buildindexes(). We do this here just for demo purposes. If the
    # indexes are not built explicitly, they will be built as needed.
    #
    print("Building the indexes...")
    fn.buildindexes()

    #
    # Get some statistics about the corpus
    #
    print("Number of Frames:", len(fn.frames()))
    print("Number of Lexical Units:", len(fn.lus()))
    print("Number of annotated documents:", len(fn.docs()))
    print()

    #
    # Frames
    #
    print(
        'getting frames whose name matches the (case insensitive) regex: "(?i)medical"'
    )
    medframes = fn.frames(r"(?i)medical")
    print(f'Found {len(medframes)} Frames whose name matches "(?i)medical":')
    print([(f.name, f.ID) for f in medframes])

    #
    # store the first frame in the list of frames
    #
    tmp_id = medframes[0].ID
    m_frame = fn.frame(tmp_id)  # reads all info for the frame

    #
    # get the frame relations
    #
    print(
        '\nNumber of frame relations for the "{}" ({}) frame:'.format(
            m_frame.name, m_frame.ID
        ),
        len(m_frame.frameRelations),
    )
    for fr in m_frame.frameRelations:
        print("   ", fr)

    #
    # get the names of the Frame Elements
    #
    print(
        f'\nNumber of Frame Elements in the "{m_frame.name}" frame:',
        len(m_frame.FE),
    )
    print("   ", [x for x in m_frame.FE])

    #
    # get the names of the "Core" Frame Elements
    #
    print(f'\nThe "core" Frame Elements in the "{m_frame.name}" frame:')
    print("   ", [x.name for x in m_frame.FE.values() if x.coreType == "Core"])

    #
    # get all of the Lexical Units that are incorporated in the
    # 'Ailment' FE of the 'Medical_conditions' frame (id=239)
    #
    print('\nAll Lexical Units that are incorporated in the "Ailment" FE:')
    m_frame = fn.frame(239)
    ailment_lus = [
        x
        for x in m_frame.lexUnit.values()
        if "incorporatedFE" in x and x.incorporatedFE == "Ailment"
    ]
    print("   ", [x.name for x in ailment_lus])

    #
    # get all of the Lexical Units for the frame
    #
    print(
        f'\nNumber of Lexical Units in the "{m_frame.name}" frame:',
        len(m_frame.lexUnit),
    )
    print("  ", [x.name for x in m_frame.lexUnit.values()][:5], "...")

    #
    # get basic info on the second LU in the frame
    #
    tmp_id = m_frame.lexUnit["ailment.n"].ID  # grab the id of the specified LU
    luinfo = fn.lu_basic(tmp_id)  # get basic info on the LU
    print(f"\nInformation on the LU: {luinfo.name}")
    pprint(luinfo)

    #
    # Get a list of all of the corpora used for fulltext annotation
    #
    print("\nNames of all of the corpora used for fulltext annotation:")
    allcorpora = {x.corpname for x in fn.docs_metadata()}
    pprint(list(allcorpora))

    #
    # Get the names of the annotated documents in the first corpus
    #
    firstcorp = list(allcorpora)[0]
    firstcorp_docs = fn.docs(firstcorp)
    print(f'\nNames of the annotated documents in the "{firstcorp}" corpus:')
    pprint([x.filename for x in firstcorp_docs])

    #
    # Search for frames containing LUs whose name attribute matches a
    # regexp pattern.
    #
    # Note: if you were going to be doing a lot of this type of
    #       searching, you'd want to build an index that maps from
    #       lemmas to frames because each time frames_by_lemma() is
    #       called, it has to search through ALL of the frame XML files
    #       in the db.
    print(
        '\nSearching for all Frames that have a lemma that matches the regexp: "^run.v$":'
    )
    pprint(fn.frames_by_lemma(r"^run.v$"))


if __name__ == "__main__":
    demo()
