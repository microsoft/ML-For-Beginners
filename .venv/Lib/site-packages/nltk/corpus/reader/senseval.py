# Natural Language Toolkit: Senseval 2 Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Trevor Cohn <tacohn@cs.mu.oz.au>
#         Steven Bird <stevenbird1@gmail.com> (modifications)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Read from the Senseval 2 Corpus.

SENSEVAL [http://www.senseval.org/]
Evaluation exercises for Word Sense Disambiguation.
Organized by ACL-SIGLEX [https://www.siglex.org/]

Prepared by Ted Pedersen <tpederse@umn.edu>, University of Minnesota,
https://www.d.umn.edu/~tpederse/data.html
Distributed with permission.

The NLTK version of the Senseval 2 files uses well-formed XML.
Each instance of the ambiguous words "hard", "interest", "line", and "serve"
is tagged with a sense identifier, and supplied with context.
"""

import re
from xml.etree import ElementTree

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *


class SensevalInstance:
    def __init__(self, word, position, context, senses):
        self.word = word
        self.senses = tuple(senses)
        self.position = position
        self.context = context

    def __repr__(self):
        return "SensevalInstance(word=%r, position=%r, " "context=%r, senses=%r)" % (
            self.word,
            self.position,
            self.context,
            self.senses,
        )


class SensevalCorpusReader(CorpusReader):
    def instances(self, fileids=None):
        return concat(
            [
                SensevalCorpusView(fileid, enc)
                for (fileid, enc) in self.abspaths(fileids, True)
            ]
        )

    def _entry(self, tree):
        elts = []
        for lexelt in tree.findall("lexelt"):
            for inst in lexelt.findall("instance"):
                sense = inst[0].attrib["senseid"]
                context = [(w.text, w.attrib["pos"]) for w in inst[1]]
                elts.append((sense, context))
        return elts


class SensevalCorpusView(StreamBackedCorpusView):
    def __init__(self, fileid, encoding):
        StreamBackedCorpusView.__init__(self, fileid, encoding=encoding)

        self._word_tokenizer = WhitespaceTokenizer()
        self._lexelt_starts = [0]  # list of streampos
        self._lexelts = [None]  # list of lexelt names

    def read_block(self, stream):
        # Decide which lexical element we're in.
        lexelt_num = bisect.bisect_right(self._lexelt_starts, stream.tell()) - 1
        lexelt = self._lexelts[lexelt_num]

        instance_lines = []
        in_instance = False
        while True:
            line = stream.readline()
            if line == "":
                assert instance_lines == []
                return []

            # Start of a lexical element?
            if line.lstrip().startswith("<lexelt"):
                lexelt_num += 1
                m = re.search("item=(\"[^\"]+\"|'[^']+')", line)
                assert m is not None  # <lexelt> has no 'item=...'
                lexelt = m.group(1)[1:-1]
                if lexelt_num < len(self._lexelts):
                    assert lexelt == self._lexelts[lexelt_num]
                else:
                    self._lexelts.append(lexelt)
                    self._lexelt_starts.append(stream.tell())

            # Start of an instance?
            if line.lstrip().startswith("<instance"):
                assert instance_lines == []
                in_instance = True

            # Body of an instance?
            if in_instance:
                instance_lines.append(line)

            # End of an instance?
            if line.lstrip().startswith("</instance"):
                xml_block = "\n".join(instance_lines)
                xml_block = _fixXML(xml_block)
                inst = ElementTree.fromstring(xml_block)
                return [self._parse_instance(inst, lexelt)]

    def _parse_instance(self, instance, lexelt):
        senses = []
        context = []
        position = None
        for child in instance:
            if child.tag == "answer":
                senses.append(child.attrib["senseid"])
            elif child.tag == "context":
                context += self._word_tokenizer.tokenize(child.text)
                for cword in child:
                    if cword.tag == "compound":
                        cword = cword[0]  # is this ok to do?

                    if cword.tag == "head":
                        # Some santiy checks:
                        assert position is None, "head specified twice"
                        assert cword.text.strip() or len(cword) == 1
                        assert not (cword.text.strip() and len(cword) == 1)
                        # Record the position of the head:
                        position = len(context)
                        # Add on the head word itself:
                        if cword.text.strip():
                            context.append(cword.text.strip())
                        elif cword[0].tag == "wf":
                            context.append((cword[0].text, cword[0].attrib["pos"]))
                            if cword[0].tail:
                                context += self._word_tokenizer.tokenize(cword[0].tail)
                        else:
                            assert False, "expected CDATA or wf in <head>"
                    elif cword.tag == "wf":
                        context.append((cword.text, cword.attrib["pos"]))
                    elif cword.tag == "s":
                        pass  # Sentence boundary marker.

                    else:
                        print("ACK", cword.tag)
                        assert False, "expected CDATA or <wf> or <head>"
                    if cword.tail:
                        context += self._word_tokenizer.tokenize(cword.tail)
            else:
                assert False, "unexpected tag %s" % child.tag
        return SensevalInstance(lexelt, position, context, senses)


def _fixXML(text):
    """
    Fix the various issues with Senseval pseudo-XML.
    """
    # <~> or <^> => ~ or ^
    text = re.sub(r"<([~\^])>", r"\1", text)
    # fix lone &
    text = re.sub(r"(\s+)\&(\s+)", r"\1&amp;\2", text)
    # fix """
    text = re.sub(r'"""', "'\"'", text)
    # fix <s snum=dd> => <s snum="dd"/>
    text = re.sub(r'(<[^<]*snum=)([^">]+)>', r'\1"\2"/>', text)
    # fix foreign word tag
    text = re.sub(r"<\&frasl>\s*<p[^>]*>", "FRASL", text)
    # remove <&I .>
    text = re.sub(r"<\&I[^>]*>", "", text)
    # fix <{word}>
    text = re.sub(r"<{([^}]+)}>", r"\1", text)
    # remove <@>, <p>, </p>
    text = re.sub(r"<(@|/?p)>", r"", text)
    # remove <&M .> and <&T .> and <&Ms .>
    text = re.sub(r"<&\w+ \.>", r"", text)
    # remove <!DOCTYPE... > lines
    text = re.sub(r"<!DOCTYPE[^>]*>", r"", text)
    # remove <[hi]> and <[/p]> etc
    text = re.sub(r"<\[\/?[^>]+\]*>", r"", text)
    # take the thing out of the brackets: <&hellip;>
    text = re.sub(r"<(\&\w+;)>", r"\1", text)
    # and remove the & for those patterns that aren't regular XML
    text = re.sub(r"&(?!amp|gt|lt|apos|quot)", r"", text)
    # fix 'abc <p="foo"/>' style tags - now <wf pos="foo">abc</wf>
    text = re.sub(
        r'[ \t]*([^<>\s]+?)[ \t]*<p="([^"]*"?)"/>', r' <wf pos="\2">\1</wf>', text
    )
    text = re.sub(r'\s*"\s*<p=\'"\'/>', " <wf pos='\"'>\"</wf>", text)
    return text
