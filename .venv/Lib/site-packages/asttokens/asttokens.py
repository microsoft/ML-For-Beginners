# Copyright 2016 Grist Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import ast
import bisect
import sys
import token
from ast import Module
from typing import Iterable, Iterator, List, Optional, Tuple, Any, cast, TYPE_CHECKING

import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from .line_numbers import LineNumbers
from .util import (
  Token, match_token, is_non_coding_token, patched_generate_tokens, last_stmt,
  annotate_fstring_nodes, generate_tokens, is_module, is_stmt
)

if TYPE_CHECKING:  # pragma: no cover
  from .util import AstNode, TokenInfo


class ASTTextBase(six.with_metaclass(abc.ABCMeta, object)):
  def __init__(self, source_text, filename):
    # type: (Any, str) -> None
    # FIXME: Strictly, the type of source_text is one of the six string types, but hard to specify with mypy given
    # https://mypy.readthedocs.io/en/stable/common_issues.html#variables-vs-type-aliases

    self._filename = filename

    # Decode source after parsing to let Python 2 handle coding declarations.
    # (If the encoding was not utf-8 compatible, then even if it parses correctly,
    # we'll fail with a unicode error here.)
    source_text = six.ensure_text(source_text)

    self._text = source_text
    self._line_numbers = LineNumbers(source_text)

  @abc.abstractmethod
  def get_text_positions(self, node, padded):
    # type: (AstNode, bool) -> Tuple[Tuple[int, int], Tuple[int, int]]
    """
    Returns two ``(lineno, col_offset)`` tuples for the start and end of the given node.
    If the positions can't be determined, or the nodes don't correspond to any particular text,
    returns ``(1, 0)`` for both.

    ``padded`` corresponds to the ``padded`` argument to ``ast.get_source_segment()``.
    This means that if ``padded`` is True, the start position will be adjusted to include
    leading whitespace if ``node`` is a multiline statement.
    """
    raise NotImplementedError  # pragma: no cover

  def get_text_range(self, node, padded=True):
    # type: (AstNode, bool) -> Tuple[int, int]
    """
    Returns the (startpos, endpos) positions in source text corresponding to the given node.
    Returns (0, 0) for nodes (like `Load`) that don't correspond to any particular text.

    See ``get_text_positions()`` for details on the ``padded`` argument.
    """
    start, end = self.get_text_positions(node, padded)
    return (
      self._line_numbers.line_to_offset(*start),
      self._line_numbers.line_to_offset(*end),
    )

  def get_text(self, node, padded=True):
    # type: (AstNode, bool) -> str
    """
    Returns the text corresponding to the given node.
    Returns '' for nodes (like `Load`) that don't correspond to any particular text.

    See ``get_text_positions()`` for details on the ``padded`` argument.
    """
    start, end = self.get_text_range(node, padded)
    return self._text[start: end]


class ASTTokens(ASTTextBase, object):
  """
  ASTTokens maintains the text of Python code in several forms: as a string, as line numbers, and
  as tokens, and is used to mark and access token and position information.

  ``source_text`` must be a unicode or UTF8-encoded string. If you pass in UTF8 bytes, remember
  that all offsets you'll get are to the unicode text, which is available as the ``.text``
  property.

  If ``parse`` is set, the ``source_text`` will be parsed with ``ast.parse()``, and the resulting
  tree marked with token info and made available as the ``.tree`` property.

  If ``tree`` is given, it will be marked and made available as the ``.tree`` property. In
  addition to the trees produced by the ``ast`` module, ASTTokens will also mark trees produced
  using ``astroid`` library <https://www.astroid.org>.

  If only ``source_text`` is given, you may use ``.mark_tokens(tree)`` to mark the nodes of an AST
  tree created separately.
  """

  def __init__(self, source_text, parse=False, tree=None, filename='<unknown>', tokens=None):
    # type: (Any, bool, Optional[Module], str, Iterable[TokenInfo]) -> None
    # FIXME: Strictly, the type of source_text is one of the six string types, but hard to specify with mypy given
    # https://mypy.readthedocs.io/en/stable/common_issues.html#variables-vs-type-aliases

    super(ASTTokens, self).__init__(source_text, filename)

    self._tree = ast.parse(source_text, filename) if parse else tree

    # Tokenize the code.
    if tokens is None:
      tokens = generate_tokens(self._text)
    self._tokens = list(self._translate_tokens(tokens))

    # Extract the start positions of all tokens, so that we can quickly map positions to tokens.
    self._token_offsets = [tok.startpos for tok in self._tokens]

    if self._tree:
      self.mark_tokens(self._tree)

  def mark_tokens(self, root_node):
    # type: (Module) -> None
    """
    Given the root of the AST or Astroid tree produced from source_text, visits all nodes marking
    them with token and position information by adding ``.first_token`` and
    ``.last_token``attributes. This is done automatically in the constructor when ``parse`` or
    ``tree`` arguments are set, but may be used manually with a separate AST or Astroid tree.
    """
    # The hard work of this class is done by MarkTokens
    from .mark_tokens import MarkTokens  # to avoid import loops
    MarkTokens(self).visit_tree(root_node)

  def _translate_tokens(self, original_tokens):
    # type: (Iterable[TokenInfo]) -> Iterator[Token]
    """
    Translates the given standard library tokens into our own representation.
    """
    for index, tok in enumerate(patched_generate_tokens(original_tokens)):
      tok_type, tok_str, start, end, line = tok
      yield Token(tok_type, tok_str, start, end, line, index,
                  self._line_numbers.line_to_offset(start[0], start[1]),
                  self._line_numbers.line_to_offset(end[0], end[1]))

  @property
  def text(self):
    # type: () -> str
    """The source code passed into the constructor."""
    return self._text

  @property
  def tokens(self):
    # type: () -> List[Token]
    """The list of tokens corresponding to the source code from the constructor."""
    return self._tokens

  @property
  def tree(self):
    # type: () -> Optional[Module]
    """The root of the AST tree passed into the constructor or parsed from the source code."""
    return self._tree

  @property
  def filename(self):
    # type: () -> str
    """The filename that was parsed"""
    return self._filename

  def get_token_from_offset(self, offset):
    # type: (int) -> Token
    """
    Returns the token containing the given character offset (0-based position in source text),
    or the preceeding token if the position is between tokens.
    """
    return self._tokens[bisect.bisect(self._token_offsets, offset) - 1]

  def get_token(self, lineno, col_offset):
    # type: (int, int) -> Token
    """
    Returns the token containing the given (lineno, col_offset) position, or the preceeding token
    if the position is between tokens.
    """
    # TODO: add test for multibyte unicode. We need to translate offsets from ast module (which
    # are in utf8) to offsets into the unicode text. tokenize module seems to use unicode offsets
    # but isn't explicit.
    return self.get_token_from_offset(self._line_numbers.line_to_offset(lineno, col_offset))

  def get_token_from_utf8(self, lineno, col_offset):
    # type: (int, int) -> Token
    """
    Same as get_token(), but interprets col_offset as a UTF8 offset, which is what `ast` uses.
    """
    return self.get_token(lineno, self._line_numbers.from_utf8_col(lineno, col_offset))

  def next_token(self, tok, include_extra=False):
    # type: (Token, bool) -> Token
    """
    Returns the next token after the given one. If include_extra is True, includes non-coding
    tokens from the tokenize module, such as NL and COMMENT.
    """
    i = tok.index + 1
    if not include_extra:
      while is_non_coding_token(self._tokens[i].type):
        i += 1
    return self._tokens[i]

  def prev_token(self, tok, include_extra=False):
    # type: (Token, bool) -> Token
    """
    Returns the previous token before the given one. If include_extra is True, includes non-coding
    tokens from the tokenize module, such as NL and COMMENT.
    """
    i = tok.index - 1
    if not include_extra:
      while is_non_coding_token(self._tokens[i].type):
        i -= 1
    return self._tokens[i]

  def find_token(self, start_token, tok_type, tok_str=None, reverse=False):
    # type: (Token, int, Optional[str], bool) -> Token
    """
    Looks for the first token, starting at start_token, that matches tok_type and, if given, the
    token string. Searches backwards if reverse is True. Returns ENDMARKER token if not found (you
    can check it with `token.ISEOF(t.type)`).
    """
    t = start_token
    advance = self.prev_token if reverse else self.next_token
    while not match_token(t, tok_type, tok_str) and not token.ISEOF(t.type):
      t = advance(t, include_extra=True)
    return t

  def token_range(self,
                  first_token,  # type: Token
                  last_token,  # type: Token
                  include_extra=False,  # type: bool
                  ):
    # type: (...) -> Iterator[Token]
    """
    Yields all tokens in order from first_token through and including last_token. If
    include_extra is True, includes non-coding tokens such as tokenize.NL and .COMMENT.
    """
    for i in xrange(first_token.index, last_token.index + 1):
      if include_extra or not is_non_coding_token(self._tokens[i].type):
        yield self._tokens[i]

  def get_tokens(self, node, include_extra=False):
    # type: (AstNode, bool) -> Iterator[Token]
    """
    Yields all tokens making up the given node. If include_extra is True, includes non-coding
    tokens such as tokenize.NL and .COMMENT.
    """
    return self.token_range(node.first_token, node.last_token, include_extra=include_extra)

  def get_text_positions(self, node, padded):
    # type: (AstNode, bool) -> Tuple[Tuple[int, int], Tuple[int, int]]
    """
    Returns two ``(lineno, col_offset)`` tuples for the start and end of the given node.
    If the positions can't be determined, or the nodes don't correspond to any particular text,
    returns ``(1, 0)`` for both.

    ``padded`` corresponds to the ``padded`` argument to ``ast.get_source_segment()``.
    This means that if ``padded`` is True, the start position will be adjusted to include
    leading whitespace if ``node`` is a multiline statement.
    """
    if not hasattr(node, 'first_token'):
      return (1, 0), (1, 0)

    start = node.first_token.start
    end = node.last_token.end
    if padded and any(match_token(t, token.NEWLINE) for t in self.get_tokens(node)):
      # Set col_offset to 0 to include leading indentation for multiline statements.
      start = (start[0], 0)

    return start, end


class ASTText(ASTTextBase, object):
  """
  Supports the same ``get_text*`` methods as ``ASTTokens``,
  but uses the AST to determine the text positions instead of tokens.
  This is faster than ``ASTTokens`` as it requires less setup work.

  It also (sometimes) supports nodes inside f-strings, which ``ASTTokens`` doesn't.

  Some node types and/or Python versions are not supported.
  In these cases the ``get_text*`` methods will fall back to using ``ASTTokens``
  which incurs the usual setup cost the first time.
  If you want to avoid this, check ``supports_tokenless(node)`` before calling ``get_text*`` methods.
  """
  def __init__(self, source_text, tree=None, filename='<unknown>'):
    # type: (Any, Optional[Module], str) -> None
    # FIXME: Strictly, the type of source_text is one of the six string types, but hard to specify with mypy given
    # https://mypy.readthedocs.io/en/stable/common_issues.html#variables-vs-type-aliases

    super(ASTText, self).__init__(source_text, filename)

    self._tree = tree
    if self._tree is not None:
      annotate_fstring_nodes(self._tree)

    self._asttokens = None  # type: Optional[ASTTokens]

  @property
  def tree(self):
    # type: () -> Module
    if self._tree is None:
      self._tree = ast.parse(self._text, self._filename)
      annotate_fstring_nodes(self._tree)
    return self._tree

  @property
  def asttokens(self):
    # type: () -> ASTTokens
    if self._asttokens is None:
      self._asttokens = ASTTokens(
          self._text,
          tree=self.tree,
          filename=self._filename,
      )
    return self._asttokens

  def _get_text_positions_tokenless(self, node, padded):
    # type: (AstNode, bool) -> Tuple[Tuple[int, int], Tuple[int, int]]
    """
    Version of ``get_text_positions()`` that doesn't use tokens.
    """
    if sys.version_info[:2] < (3, 8):  # pragma: no cover
      # This is just for mpypy
      raise AssertionError("This method should only be called internally after checking supports_tokenless()")

    if is_module(node):
      # Modules don't have position info, so just return the range of the whole text.
      # The token-using method does something different, but its behavior seems weird and inconsistent.
      # For example, in a file with only comments, it only returns the first line.
      # It's hard to imagine a case when this matters.
      return (1, 0), self._line_numbers.offset_to_line(len(self._text))

    if getattr(node, 'lineno', None) is None:
      return (1, 0), (1, 0)

    assert node  # tell mypy that node is not None, which we allowed up to here for compatibility

    decorators = getattr(node, 'decorator_list', [])
    if not decorators:
      # Astroid uses node.decorators.nodes instead of node.decorator_list.
      decorators_node = getattr(node, 'decorators', None)
      decorators = getattr(decorators_node, 'nodes', [])
    if decorators:
      # Function/Class definition nodes are marked by AST as starting at def/class,
      # not the first decorator. This doesn't match the token-using behavior,
      # or inspect.getsource(), and just seems weird.
      start_node = decorators[0]
    else:
      start_node = node

    start_lineno = start_node.lineno
    end_node = last_stmt(node)

    # Include leading indentation for multiline statements.
    # This doesn't mean simple statements that happen to be on multiple lines,
    # but compound statements where inner indentation matters.
    # So we don't just compare node.lineno and node.end_lineno,
    # we check for a contained statement starting on a different line.
    if padded and (
        start_lineno != end_node.lineno
        or (
            # Astroid docstrings aren't treated as separate statements.
            # So to handle function/class definitions with a docstring but no other body,
            # we just check that the node is a statement with a docstring
            # and spanning multiple lines in the simple, literal sense.
            start_lineno != node.end_lineno
            and getattr(node, "doc_node", None)
            and is_stmt(node)
        )
    ):
      start_col_offset = 0
    else:
      start_col_offset = self._line_numbers.from_utf8_col(start_lineno, start_node.col_offset)

    start = (start_lineno, start_col_offset)

    # To match the token-using behaviour, we exclude trailing semicolons and comments.
    # This means that for blocks containing multiple statements, we have to use the last one
    # instead of the actual node for end_lineno and end_col_offset.
    end_lineno = cast(int, end_node.end_lineno)
    end_col_offset = cast(int, end_node.end_col_offset)
    end_col_offset = self._line_numbers.from_utf8_col(end_lineno, end_col_offset)
    end = (end_lineno, end_col_offset)

    return start, end

  def get_text_positions(self, node, padded):
    # type: (AstNode, bool) -> Tuple[Tuple[int, int], Tuple[int, int]]
    """
    Returns two ``(lineno, col_offset)`` tuples for the start and end of the given node.
    If the positions can't be determined, or the nodes don't correspond to any particular text,
    returns ``(1, 0)`` for both.

    ``padded`` corresponds to the ``padded`` argument to ``ast.get_source_segment()``.
    This means that if ``padded`` is True, the start position will be adjusted to include
    leading whitespace if ``node`` is a multiline statement.
    """
    if getattr(node, "_broken_positions", None):
      # This node was marked in util.annotate_fstring_nodes as having untrustworthy lineno/col_offset.
      return (1, 0), (1, 0)

    if supports_tokenless(node):
      return self._get_text_positions_tokenless(node, padded)

    return self.asttokens.get_text_positions(node, padded)


# Node types that _get_text_positions_tokenless doesn't support. Only relevant for Python 3.8+.
_unsupported_tokenless_types = ()  # type: Tuple[str, ...]
if sys.version_info[:2] >= (3, 8):
  # no lineno
  _unsupported_tokenless_types += ("arguments", "Arguments", "withitem")
  if sys.version_info[:2] == (3, 8):
    # _get_text_positions_tokenless works incorrectly for these types due to bugs in Python 3.8.
    _unsupported_tokenless_types += ("arg", "Starred")
    # no lineno in 3.8
    _unsupported_tokenless_types += ("Slice", "ExtSlice", "Index", "keyword")


def supports_tokenless(node=None):
  # type: (Any) -> bool
  """
  Returns True if the Python version and the node (if given) are supported by
  the ``get_text*`` methods of ``ASTText`` without falling back to ``ASTTokens``.
  See ``ASTText`` for why this matters.

  The following cases are not supported:

    - Python 3.7 and earlier
    - PyPy
    - ``ast.arguments`` / ``astroid.Arguments``
    - ``ast.withitem``
    - ``astroid.Comprehension``
    - ``astroid.AssignName`` inside ``astroid.Arguments`` or ``astroid.ExceptHandler``
    - The following nodes in Python 3.8 only:
      - ``ast.arg``
      - ``ast.Starred``
      - ``ast.Slice``
      - ``ast.ExtSlice``
      - ``ast.Index``
      - ``ast.keyword``
  """
  return (
      type(node).__name__ not in _unsupported_tokenless_types
      and not (
        # astroid nodes
        not isinstance(node, ast.AST) and node is not None and (
          (
            type(node).__name__ == "AssignName"
            and type(node.parent).__name__ in ("Arguments", "ExceptHandler")
          )
        )
      )
      and sys.version_info[:2] >= (3, 8)
      and 'pypy' not in sys.version.lower()
  )
