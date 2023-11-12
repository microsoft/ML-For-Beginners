from collections import namedtuple
from functools import partial, wraps

from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize


def comma_separated_string_args(func):
    """
    A decorator that allows a function to be called with
    a single string of comma-separated values which become
    individual function arguments.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        _args = list()
        for arg in args:
            if isinstance(arg, str):
                _args.append({part.strip() for part in arg.split(",")})
            elif isinstance(arg, list):
                _args.append(set(arg))
            else:
                _args.append(arg)
        for name, value in kwargs.items():
            if isinstance(value, str):
                kwargs[name] = {part.strip() for part in value.split(",")}
        return func(*_args, **kwargs)

    return wrapper


def read_parse_blankline_block(stream, parser):
    block = read_blankline_block(stream)
    if block:
        return [parser.render(block[0])]
    return block


class MarkdownBlock:
    def __init__(self, content):
        self.content = content
        self.truncate_at = 16

    def __repr__(self):
        return f"{self.__class__.__name__}(content={repr(str(self))})"

    def __str__(self):
        return (
            f"{self.content[:self.truncate_at]}"
            f"{'...' if len(self.content) > self.truncate_at else ''}"
        )

    @property
    def raw(self):
        return self.content

    @property
    def words(self):
        return word_tokenize(self.content)

    @property
    def sents(self):
        return [word_tokenize(sent) for sent in sent_tokenize(self.content)]

    @property
    def paras(self):
        return [
            [word_tokenize(sent) for sent in sent_tokenize(para)]
            for para in blankline_tokenize(self.content)
        ]


class CodeBlock(MarkdownBlock):
    def __init__(self, language, *args):
        self.language = language
        super().__init__(*args)

    @property
    def sents(self):
        return [word_tokenize(line) for line in self.content.splitlines()]

    @property
    def lines(self):
        return self.content.splitlines()

    @property
    def paras(self):
        return [
            [word_tokenize(line) for line in para.splitlines()]
            for para in blankline_tokenize(self.content)
        ]


class MarkdownSection(MarkdownBlock):
    def __init__(self, heading, level, *args):
        self.heading = heading
        self.level = level
        super().__init__(*args)


Image = namedtuple("Image", "label, src, title")
Link = namedtuple("Link", "label, href, title")
List = namedtuple("List", "is_ordered, items")


class MarkdownCorpusReader(PlaintextCorpusReader):
    def __init__(self, *args, parser=None, **kwargs):
        from markdown_it import MarkdownIt
        from mdit_plain.renderer import RendererPlain
        from mdit_py_plugins.front_matter import front_matter_plugin

        self.parser = parser
        if self.parser is None:
            self.parser = MarkdownIt("commonmark", renderer_cls=RendererPlain)
            self.parser.use(front_matter_plugin)

        kwargs.setdefault(
            "para_block_reader", partial(read_parse_blankline_block, parser=self.parser)
        )
        super().__init__(*args, **kwargs)

    # This override takes care of removing markup.
    def _read_word_block(self, stream):
        words = list()
        for para in self._para_block_reader(stream):
            words.extend(self._word_tokenizer.tokenize(para))
        return words


class CategorizedMarkdownCorpusReader(CategorizedCorpusReader, MarkdownCorpusReader):
    """
    A reader for markdown corpora whose documents are divided into
    categories based on their file identifiers.

    Based on nltk.corpus.reader.plaintext.CategorizedPlaintextCorpusReader:
    https://www.nltk.org/_modules/nltk/corpus/reader/api.html#CategorizedCorpusReader
    """

    def __init__(self, *args, cat_field="tags", **kwargs):
        """
        Initialize the corpus reader. Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining arguments
        are passed to the ``MarkdownCorpusReader`` constructor.
        """
        cat_args = ["cat_pattern", "cat_map", "cat_file"]
        if not any(arg in kwargs for arg in cat_args):
            # Initialize with a blank map now,
            # and try to build categories from document metadata later.
            kwargs["cat_map"] = dict()
        CategorizedCorpusReader.__init__(self, kwargs)
        MarkdownCorpusReader.__init__(self, *args, **kwargs)

        # Map file IDs to categories if self._map exists but is still empty:
        if self._map is not None and not self._map:
            for file_id in self._fileids:
                metadata = self.metadata(file_id)
                if metadata:
                    self._map[file_id] = metadata[0].get(cat_field, [])

    ### Begin CategorizedCorpusReader Overrides
    @comma_separated_string_args
    def categories(self, fileids=None):
        return super().categories(fileids)

    @comma_separated_string_args
    def fileids(self, categories=None):
        if categories is None:
            return self._fileids
        return super().fileids(categories)

    ### End CategorizedCorpusReader Overrides

    ### Begin MarkdownCorpusReader Overrides
    @comma_separated_string_args
    def raw(self, fileids=None, categories=None):
        return super().raw(self._resolve(fileids, categories))

    @comma_separated_string_args
    def words(self, fileids=None, categories=None):
        return super().words(self._resolve(fileids, categories))

    @comma_separated_string_args
    def sents(self, fileids=None, categories=None):
        return super().sents(self._resolve(fileids, categories))

    @comma_separated_string_args
    def paras(self, fileids=None, categories=None):
        return super().paras(self._resolve(fileids, categories))

    ### End MarkdownCorpusReader Overrides

    def concatenated_view(self, reader, fileids, categories):
        return concat(
            [
                self.CorpusView(path, reader, encoding=enc)
                for (path, enc) in self.abspaths(
                    self._resolve(fileids, categories), include_encoding=True
                )
            ]
        )

    def metadata_reader(self, stream):
        from yaml import safe_load

        return [
            safe_load(t.content)
            for t in self.parser.parse(stream.read())
            if t.type == "front_matter"
        ]

    @comma_separated_string_args
    def metadata(self, fileids=None, categories=None):
        return self.concatenated_view(self.metadata_reader, fileids, categories)

    def blockquote_reader(self, stream):
        tokens = self.parser.parse(stream.read())
        opening_tokens = filter(
            lambda t: t.level == 0 and t.type == "blockquote_open", tokens
        )
        closing_tokens = filter(
            lambda t: t.level == 0 and t.type == "blockquote_close", tokens
        )
        blockquotes = list()
        for o, c in zip(opening_tokens, closing_tokens):
            opening_index = tokens.index(o)
            closing_index = tokens.index(c, opening_index)
            blockquotes.append(tokens[opening_index : closing_index + 1])
        return [
            MarkdownBlock(
                self.parser.renderer.render(block, self.parser.options, env=None)
            )
            for block in blockquotes
        ]

    @comma_separated_string_args
    def blockquotes(self, fileids=None, categories=None):
        return self.concatenated_view(self.blockquote_reader, fileids, categories)

    def code_block_reader(self, stream):
        return [
            CodeBlock(
                t.info,
                t.content,
            )
            for t in self.parser.parse(stream.read())
            if t.level == 0 and t.type in ("fence", "code_block")
        ]

    @comma_separated_string_args
    def code_blocks(self, fileids=None, categories=None):
        return self.concatenated_view(self.code_block_reader, fileids, categories)

    def image_reader(self, stream):
        return [
            Image(
                child_token.content,
                child_token.attrGet("src"),
                child_token.attrGet("title"),
            )
            for inline_token in filter(
                lambda t: t.type == "inline", self.parser.parse(stream.read())
            )
            for child_token in inline_token.children
            if child_token.type == "image"
        ]

    @comma_separated_string_args
    def images(self, fileids=None, categories=None):
        return self.concatenated_view(self.image_reader, fileids, categories)

    def link_reader(self, stream):
        return [
            Link(
                inline_token.children[i + 1].content,
                child_token.attrGet("href"),
                child_token.attrGet("title"),
            )
            for inline_token in filter(
                lambda t: t.type == "inline", self.parser.parse(stream.read())
            )
            for i, child_token in enumerate(inline_token.children)
            if child_token.type == "link_open"
        ]

    @comma_separated_string_args
    def links(self, fileids=None, categories=None):
        return self.concatenated_view(self.link_reader, fileids, categories)

    def list_reader(self, stream):
        tokens = self.parser.parse(stream.read())
        opening_types = ("bullet_list_open", "ordered_list_open")
        opening_tokens = filter(
            lambda t: t.level == 0 and t.type in opening_types, tokens
        )
        closing_types = ("bullet_list_close", "ordered_list_close")
        closing_tokens = filter(
            lambda t: t.level == 0 and t.type in closing_types, tokens
        )
        list_blocks = list()
        for o, c in zip(opening_tokens, closing_tokens):
            opening_index = tokens.index(o)
            closing_index = tokens.index(c, opening_index)
            list_blocks.append(tokens[opening_index : closing_index + 1])
        return [
            List(
                tokens[0].type == "ordered_list_open",
                [t.content for t in tokens if t.content],
            )
            for tokens in list_blocks
        ]

    @comma_separated_string_args
    def lists(self, fileids=None, categories=None):
        return self.concatenated_view(self.list_reader, fileids, categories)

    def section_reader(self, stream):
        section_blocks, block = list(), list()
        in_heading = False
        for t in self.parser.parse(stream.read()):
            if t.level == 0 and t.type == "heading_open":
                if block:
                    section_blocks.append(block)
                block = list()
                in_heading = True
            if in_heading:
                block.append(t)
        return [
            MarkdownSection(
                block[1].content,
                block[0].markup.count("#"),
                self.parser.renderer.render(block, self.parser.options, env=None),
            )
            for block in section_blocks
        ]

    @comma_separated_string_args
    def sections(self, fileids=None, categories=None):
        return self.concatenated_view(self.section_reader, fileids, categories)
