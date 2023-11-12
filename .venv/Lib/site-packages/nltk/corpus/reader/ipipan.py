# Natural Language Toolkit: IPI PAN Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Konrad Goluchowski <kodie@mimuw.edu.pl>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import functools

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, concat


def _parse_args(fun):
    @functools.wraps(fun)
    def decorator(self, fileids=None, **kwargs):
        kwargs.pop("tags", None)
        if not fileids:
            fileids = self.fileids()
        return fun(self, fileids, **kwargs)

    return decorator


class IPIPANCorpusReader(CorpusReader):
    """
    Corpus reader designed to work with corpus created by IPI PAN.
    See http://korpus.pl/en/ for more details about IPI PAN corpus.

    The corpus includes information about text domain, channel and categories.
    You can access possible values using ``domains()``, ``channels()`` and
    ``categories()``. You can use also this metadata to filter files, e.g.:
    ``fileids(channel='prasa')``, ``fileids(categories='publicystyczny')``.

    The reader supports methods: words, sents, paras and their tagged versions.
    You can get part of speech instead of full tag by giving "simplify_tags=True"
    parameter, e.g.: ``tagged_sents(simplify_tags=True)``.

    Also you can get all tags disambiguated tags specifying parameter
    "one_tag=False", e.g.: ``tagged_paras(one_tag=False)``.

    You can get all tags that were assigned by a morphological analyzer specifying
    parameter "disamb_only=False", e.g. ``tagged_words(disamb_only=False)``.

    The IPIPAN Corpus contains tags indicating if there is a space between two
    tokens. To add special "no space" markers, you should specify parameter
    "append_no_space=True", e.g. ``tagged_words(append_no_space=True)``.
    As a result in place where there should be no space between two tokens new
    pair ('', 'no-space') will be inserted (for tagged data) and just '' for
    methods without tags.

    The corpus reader can also try to append spaces between words. To enable this
    option, specify parameter "append_space=True", e.g. ``words(append_space=True)``.
    As a result either ' ' or (' ', 'space') will be inserted between tokens.

    By default, xml entities like &quot; and &amp; are replaced by corresponding
    characters. You can turn off this feature, specifying parameter
    "replace_xmlentities=False", e.g. ``words(replace_xmlentities=False)``.
    """

    def __init__(self, root, fileids):
        CorpusReader.__init__(self, root, fileids, None, None)

    def channels(self, fileids=None):
        if not fileids:
            fileids = self.fileids()
        return self._parse_header(fileids, "channel")

    def domains(self, fileids=None):
        if not fileids:
            fileids = self.fileids()
        return self._parse_header(fileids, "domain")

    def categories(self, fileids=None):
        if not fileids:
            fileids = self.fileids()
        return [
            self._map_category(cat) for cat in self._parse_header(fileids, "keyTerm")
        ]

    def fileids(self, channels=None, domains=None, categories=None):
        if channels is not None and domains is not None and categories is not None:
            raise ValueError(
                "You can specify only one of channels, domains "
                "and categories parameter at once"
            )
        if channels is None and domains is None and categories is None:
            return CorpusReader.fileids(self)
        if isinstance(channels, str):
            channels = [channels]
        if isinstance(domains, str):
            domains = [domains]
        if isinstance(categories, str):
            categories = [categories]
        if channels:
            return self._list_morph_files_by("channel", channels)
        elif domains:
            return self._list_morph_files_by("domain", domains)
        else:
            return self._list_morph_files_by(
                "keyTerm", categories, map=self._map_category
            )

    @_parse_args
    def sents(self, fileids=None, **kwargs):
        return concat(
            [
                self._view(
                    fileid, mode=IPIPANCorpusView.SENTS_MODE, tags=False, **kwargs
                )
                for fileid in self._list_morph_files(fileids)
            ]
        )

    @_parse_args
    def paras(self, fileids=None, **kwargs):
        return concat(
            [
                self._view(
                    fileid, mode=IPIPANCorpusView.PARAS_MODE, tags=False, **kwargs
                )
                for fileid in self._list_morph_files(fileids)
            ]
        )

    @_parse_args
    def words(self, fileids=None, **kwargs):
        return concat(
            [
                self._view(fileid, tags=False, **kwargs)
                for fileid in self._list_morph_files(fileids)
            ]
        )

    @_parse_args
    def tagged_sents(self, fileids=None, **kwargs):
        return concat(
            [
                self._view(fileid, mode=IPIPANCorpusView.SENTS_MODE, **kwargs)
                for fileid in self._list_morph_files(fileids)
            ]
        )

    @_parse_args
    def tagged_paras(self, fileids=None, **kwargs):
        return concat(
            [
                self._view(fileid, mode=IPIPANCorpusView.PARAS_MODE, **kwargs)
                for fileid in self._list_morph_files(fileids)
            ]
        )

    @_parse_args
    def tagged_words(self, fileids=None, **kwargs):
        return concat(
            [self._view(fileid, **kwargs) for fileid in self._list_morph_files(fileids)]
        )

    def _list_morph_files(self, fileids):
        return [f for f in self.abspaths(fileids)]

    def _list_header_files(self, fileids):
        return [
            f.replace("morph.xml", "header.xml")
            for f in self._list_morph_files(fileids)
        ]

    def _parse_header(self, fileids, tag):
        values = set()
        for f in self._list_header_files(fileids):
            values_list = self._get_tag(f, tag)
            for v in values_list:
                values.add(v)
        return list(values)

    def _list_morph_files_by(self, tag, values, map=None):
        fileids = self.fileids()
        ret_fileids = set()
        for f in fileids:
            fp = self.abspath(f).replace("morph.xml", "header.xml")
            values_list = self._get_tag(fp, tag)
            for value in values_list:
                if map is not None:
                    value = map(value)
                if value in values:
                    ret_fileids.add(f)
        return list(ret_fileids)

    def _get_tag(self, f, tag):
        tags = []
        with open(f) as infile:
            header = infile.read()
        tag_end = 0
        while True:
            tag_pos = header.find("<" + tag, tag_end)
            if tag_pos < 0:
                return tags
            tag_end = header.find("</" + tag + ">", tag_pos)
            tags.append(header[tag_pos + len(tag) + 2 : tag_end])

    def _map_category(self, cat):
        pos = cat.find(">")
        if pos == -1:
            return cat
        else:
            return cat[pos + 1 :]

    def _view(self, filename, **kwargs):
        tags = kwargs.pop("tags", True)
        mode = kwargs.pop("mode", 0)
        simplify_tags = kwargs.pop("simplify_tags", False)
        one_tag = kwargs.pop("one_tag", True)
        disamb_only = kwargs.pop("disamb_only", True)
        append_no_space = kwargs.pop("append_no_space", False)
        append_space = kwargs.pop("append_space", False)
        replace_xmlentities = kwargs.pop("replace_xmlentities", True)

        if len(kwargs) > 0:
            raise ValueError("Unexpected arguments: %s" % kwargs.keys())
        if not one_tag and not disamb_only:
            raise ValueError(
                "You cannot specify both one_tag=False and " "disamb_only=False"
            )
        if not tags and (simplify_tags or not one_tag or not disamb_only):
            raise ValueError(
                "You cannot specify simplify_tags, one_tag or "
                "disamb_only with functions other than tagged_*"
            )

        return IPIPANCorpusView(
            filename,
            tags=tags,
            mode=mode,
            simplify_tags=simplify_tags,
            one_tag=one_tag,
            disamb_only=disamb_only,
            append_no_space=append_no_space,
            append_space=append_space,
            replace_xmlentities=replace_xmlentities,
        )


class IPIPANCorpusView(StreamBackedCorpusView):

    WORDS_MODE = 0
    SENTS_MODE = 1
    PARAS_MODE = 2

    def __init__(self, filename, startpos=0, **kwargs):
        StreamBackedCorpusView.__init__(self, filename, None, startpos, None)
        self.in_sentence = False
        self.position = 0

        self.show_tags = kwargs.pop("tags", True)
        self.disamb_only = kwargs.pop("disamb_only", True)
        self.mode = kwargs.pop("mode", IPIPANCorpusView.WORDS_MODE)
        self.simplify_tags = kwargs.pop("simplify_tags", False)
        self.one_tag = kwargs.pop("one_tag", True)
        self.append_no_space = kwargs.pop("append_no_space", False)
        self.append_space = kwargs.pop("append_space", False)
        self.replace_xmlentities = kwargs.pop("replace_xmlentities", True)

    def read_block(self, stream):
        sentence = []
        sentences = []
        space = False
        no_space = False

        tags = set()

        lines = self._read_data(stream)

        while True:

            # we may have only part of last line
            if len(lines) <= 1:
                self._seek(stream)
                lines = self._read_data(stream)

            if lines == [""]:
                assert not sentences
                return []

            line = lines.pop()
            self.position += len(line) + 1

            if line.startswith('<chunk type="s"'):
                self.in_sentence = True
            elif line.startswith('<chunk type="p"'):
                pass
            elif line.startswith("<tok"):
                if self.append_space and space and not no_space:
                    self._append_space(sentence)
                space = True
                no_space = False
                orth = ""
                tags = set()
            elif line.startswith("</chunk"):
                if self.in_sentence:
                    self.in_sentence = False
                    self._seek(stream)
                    if self.mode == self.SENTS_MODE:
                        return [sentence]
                    elif self.mode == self.WORDS_MODE:
                        if self.append_space:
                            self._append_space(sentence)
                        return sentence
                    else:
                        sentences.append(sentence)
                elif self.mode == self.PARAS_MODE:
                    self._seek(stream)
                    return [sentences]
            elif line.startswith("<orth"):
                orth = line[6:-7]
                if self.replace_xmlentities:
                    orth = orth.replace("&quot;", '"').replace("&amp;", "&")
            elif line.startswith("<lex"):
                if not self.disamb_only or line.find("disamb=") != -1:
                    tag = line[line.index("<ctag") + 6 : line.index("</ctag")]
                    tags.add(tag)
            elif line.startswith("</tok"):
                if self.show_tags:
                    if self.simplify_tags:
                        tags = [t.split(":")[0] for t in tags]
                    if not self.one_tag or not self.disamb_only:
                        sentence.append((orth, tuple(tags)))
                    else:
                        sentence.append((orth, tags.pop()))
                else:
                    sentence.append(orth)
            elif line.startswith("<ns/>"):
                if self.append_space:
                    no_space = True
                if self.append_no_space:
                    if self.show_tags:
                        sentence.append(("", "no-space"))
                    else:
                        sentence.append("")
            elif line.startswith("</cesAna"):
                pass

    def _read_data(self, stream):
        self.position = stream.tell()
        buff = stream.read(4096)
        lines = buff.split("\n")
        lines.reverse()
        return lines

    def _seek(self, stream):
        stream.seek(self.position)

    def _append_space(self, sentence):
        if self.show_tags:
            sentence.append((" ", "space"))
        else:
            sentence.append(" ")
