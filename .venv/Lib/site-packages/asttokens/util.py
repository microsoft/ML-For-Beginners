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

import ast
import collections
import io
import sys
import token
import tokenize
from abc import ABCMeta
from ast import Module, expr, AST
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast, Any, TYPE_CHECKING

from six import iteritems


if TYPE_CHECKING:  # pragma: no cover
  from .astroid_compat import NodeNG

  # Type class used to expand out the definition of AST to include fields added by this library
  # It's not actually used for anything other than type checking though!
  class EnhancedAST(AST):
    # Additional attributes set by mark_tokens
    first_token = None  # type: Token
    last_token = None  # type: Token
    lineno = 0  # type: int

  AstNode = Union[EnhancedAST, NodeNG]

  if sys.version_info[0] == 2:
    TokenInfo = Tuple[int, str, Tuple[int, int], Tuple[int, int], str]
  else:
    TokenInfo = tokenize.TokenInfo


def token_repr(tok_type, string):
  # type: (int, Optional[str]) -> str
  """Returns a human-friendly representation of a token with the given type and string."""
  # repr() prefixes unicode with 'u' on Python2 but not Python3; strip it out for consistency.
  return '%s:%s' % (token.tok_name[tok_type], repr(string).lstrip('u'))


class Token(collections.namedtuple('Token', 'type string start end line index startpos endpos')):
  """
  TokenInfo is an 8-tuple containing the same 5 fields as the tokens produced by the tokenize
  module, and 3 additional ones useful for this module:

  - [0] .type     Token type (see token.py)
  - [1] .string   Token (a string)
  - [2] .start    Starting (row, column) indices of the token (a 2-tuple of ints)
  - [3] .end      Ending (row, column) indices of the token (a 2-tuple of ints)
  - [4] .line     Original line (string)
  - [5] .index    Index of the token in the list of tokens that it belongs to.
  - [6] .startpos Starting character offset into the input text.
  - [7] .endpos   Ending character offset into the input text.
  """
  def __str__(self):
    # type: () -> str
    return token_repr(self.type, self.string)


if sys.version_info >= (3, 6):
  AstConstant = ast.Constant
else:
  class AstConstant:
    value = object()


def match_token(token, tok_type, tok_str=None):
  # type: (Token, int, Optional[str]) -> bool
  """Returns true if token is of the given type and, if a string is given, has that string."""
  return token.type == tok_type and (tok_str is None or token.string == tok_str)


def expect_token(token, tok_type, tok_str=None):
  # type: (Token, int, Optional[str]) -> None
  """
  Verifies that the given token is of the expected type. If tok_str is given, the token string
  is verified too. If the token doesn't match, raises an informative ValueError.
  """
  if not match_token(token, tok_type, tok_str):
    raise ValueError("Expected token %s, got %s on line %s col %s" % (
      token_repr(tok_type, tok_str), str(token),
      token.start[0], token.start[1] + 1))

# These were previously defined in tokenize.py and distinguishable by being greater than
# token.N_TOKEN. As of python3.7, they are in token.py, and we check for them explicitly.
if sys.version_info >= (3, 7):
  def is_non_coding_token(token_type):
    # type: (int) -> bool
    """
    These are considered non-coding tokens, as they don't affect the syntax tree.
    """
    return token_type in (token.NL, token.COMMENT, token.ENCODING)
else:
  def is_non_coding_token(token_type):
    # type: (int) -> bool
    """
    These are considered non-coding tokens, as they don't affect the syntax tree.
    """
    return token_type >= token.N_TOKENS


def generate_tokens(text):
  # type: (str) -> Iterator[TokenInfo]
  """
  Generates standard library tokens for the given code.
  """
  # tokenize.generate_tokens is technically an undocumented API for Python3, but allows us to use the same API as for
  # Python2. See http://stackoverflow.com/a/4952291/328565.
  # FIXME: Remove cast once https://github.com/python/typeshed/issues/7003 gets fixed
  return tokenize.generate_tokens(cast(Callable[[], str], io.StringIO(text).readline))


def iter_children_func(node):
  # type: (AST) -> Callable
  """
  Returns a function which yields all direct children of a AST node,
  skipping children that are singleton nodes.
  The function depends on whether ``node`` is from ``ast`` or from the ``astroid`` module.
  """
  return iter_children_astroid if hasattr(node, 'get_children') else iter_children_ast


def iter_children_astroid(node, include_joined_str=False):
  # type: (NodeNG, bool) -> Union[Iterator, List]
  if not include_joined_str and is_joined_str(node):
    return []

  return node.get_children()


SINGLETONS = {c for n, c in iteritems(ast.__dict__) if isinstance(c, type) and
              issubclass(c, (ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop))}


def iter_children_ast(node, include_joined_str=False):
  # type: (AST, bool) -> Iterator[Union[AST, expr]]
  if not include_joined_str and is_joined_str(node):
    return

  if isinstance(node, ast.Dict):
    # override the iteration order: instead of <all keys>, <all values>,
    # yield keys and values in source order (key1, value1, key2, value2, ...)
    for (key, value) in zip(node.keys, node.values):
      if key is not None:
        yield key
      yield value
    return

  for child in ast.iter_child_nodes(node):
    # Skip singleton children; they don't reflect particular positions in the code and break the
    # assumptions about the tree consisting of distinct nodes. Note that collecting classes
    # beforehand and checking them in a set is faster than using isinstance each time.
    if child.__class__ not in SINGLETONS:
      yield child


stmt_class_names = {n for n, c in iteritems(ast.__dict__)
                    if isinstance(c, type) and issubclass(c, ast.stmt)}
expr_class_names = ({n for n, c in iteritems(ast.__dict__)
                    if isinstance(c, type) and issubclass(c, ast.expr)} |
                    {'AssignName', 'DelName', 'Const', 'AssignAttr', 'DelAttr'})

# These feel hacky compared to isinstance() but allow us to work with both ast and astroid nodes
# in the same way, and without even importing astroid.
def is_expr(node):
  # type: (AstNode) -> bool
  """Returns whether node is an expression node."""
  return node.__class__.__name__ in expr_class_names

def is_stmt(node):
  # type: (AstNode) -> bool
  """Returns whether node is a statement node."""
  return node.__class__.__name__ in stmt_class_names

def is_module(node):
  # type: (AstNode) -> bool
  """Returns whether node is a module node."""
  return node.__class__.__name__ == 'Module'

def is_joined_str(node):
  # type: (AstNode) -> bool
  """Returns whether node is a JoinedStr node, used to represent f-strings."""
  # At the moment, nodes below JoinedStr have wrong line/col info, and trying to process them only
  # leads to errors.
  return node.__class__.__name__ == 'JoinedStr'


def is_starred(node):
  # type: (AstNode) -> bool
  """Returns whether node is a starred expression node."""
  return node.__class__.__name__ == 'Starred'


def is_slice(node):
  # type: (AstNode) -> bool
  """Returns whether node represents a slice, e.g. `1:2` in `x[1:2]`"""
  # Before 3.9, a tuple containing a slice is an ExtSlice,
  # but this was removed in https://bugs.python.org/issue34822
  return (
      node.__class__.__name__ in ('Slice', 'ExtSlice')
      or (
          node.__class__.__name__ == 'Tuple'
          and any(map(is_slice, cast(ast.Tuple, node).elts))
      )
  )


def is_empty_astroid_slice(node):
  # type: (AstNode) -> bool
  return (
      node.__class__.__name__ == "Slice"
      and not isinstance(node, ast.AST)
      and node.lower is node.upper is node.step is None
  )


# Sentinel value used by visit_tree().
_PREVISIT = object()

def visit_tree(node, previsit, postvisit):
  # type: (Module, Callable[[AstNode, Optional[Token]], Tuple[Optional[Token], Optional[Token]]], Optional[Callable[[AstNode, Optional[Token], Optional[Token]], None]])   -> None
  """
  Scans the tree under the node depth-first using an explicit stack. It avoids implicit recursion
  via the function call stack to avoid hitting 'maximum recursion depth exceeded' error.

  It calls ``previsit()`` and ``postvisit()`` as follows:

  * ``previsit(node, par_value)`` - should return ``(par_value, value)``
        ``par_value`` is as returned from ``previsit()`` of the parent.

  * ``postvisit(node, par_value, value)`` - should return ``value``
        ``par_value`` is as returned from ``previsit()`` of the parent, and ``value`` is as
        returned from ``previsit()`` of this node itself. The return ``value`` is ignored except
        the one for the root node, which is returned from the overall ``visit_tree()`` call.

  For the initial node, ``par_value`` is None. ``postvisit`` may be None.
  """
  if not postvisit:
    postvisit = lambda node, pvalue, value: None

  iter_children = iter_children_func(node)
  done = set()
  ret = None
  stack = [(node, None, _PREVISIT)] # type: List[Tuple[AstNode, Optional[Token], Union[Optional[Token], object]]]
  while stack:
    current, par_value, value = stack.pop()
    if value is _PREVISIT:
      assert current not in done    # protect againt infinite loop in case of a bad tree.
      done.add(current)

      pvalue, post_value = previsit(current, par_value)
      stack.append((current, par_value, post_value))

      # Insert all children in reverse order (so that first child ends up on top of the stack).
      ins = len(stack)
      for n in iter_children(current):
        stack.insert(ins, (n, pvalue, _PREVISIT))
    else:
      ret = postvisit(current, par_value, cast(Optional[Token], value))
  return ret


def walk(node, include_joined_str=False):
  # type: (AST, bool) -> Iterator[Union[Module, AstNode]]
  """
  Recursively yield all descendant nodes in the tree starting at ``node`` (including ``node``
  itself), using depth-first pre-order traversal (yieling parents before their children).

  This is similar to ``ast.walk()``, but with a different order, and it works for both ``ast`` and
  ``astroid`` trees. Also, as ``iter_children()``, it skips singleton nodes generated by ``ast``.

  By default, ``JoinedStr`` (f-string) nodes and their contents are skipped
  because they previously couldn't be handled. Set ``include_joined_str`` to True to include them.
  """
  iter_children = iter_children_func(node)
  done = set()
  stack = [node]
  while stack:
    current = stack.pop()
    assert current not in done    # protect againt infinite loop in case of a bad tree.
    done.add(current)

    yield current

    # Insert all children in reverse order (so that first child ends up on top of the stack).
    # This is faster than building a list and reversing it.
    ins = len(stack)
    for c in iter_children(current, include_joined_str):
      stack.insert(ins, c)


def replace(text, replacements):
  # type: (str, List[Tuple[int, int, str]]) -> str
  """
  Replaces multiple slices of text with new values. This is a convenience method for making code
  modifications of ranges e.g. as identified by ``ASTTokens.get_text_range(node)``. Replacements is
  an iterable of ``(start, end, new_text)`` tuples.

  For example, ``replace("this is a test", [(0, 4, "X"), (8, 9, "THE")])`` produces
  ``"X is THE test"``.
  """
  p = 0
  parts = []
  for (start, end, new_text) in sorted(replacements):
    parts.append(text[p:start])
    parts.append(new_text)
    p = end
  parts.append(text[p:])
  return ''.join(parts)


class NodeMethods(object):
  """
  Helper to get `visit_{node_type}` methods given a node's class and cache the results.
  """
  def __init__(self):
    # type: () -> None
    self._cache = {} # type: Dict[Union[ABCMeta, type], Callable[[AstNode, Token, Token], Tuple[Token, Token]]]

  def get(self, obj, cls):
    # type: (Any, Union[ABCMeta, type]) -> Callable
    """
    Using the lowercase name of the class as node_type, returns `obj.visit_{node_type}`,
    or `obj.visit_default` if the type-specific method is not found.
    """
    method = self._cache.get(cls)
    if not method:
      name = "visit_" + cls.__name__.lower()
      method = getattr(obj, name, obj.visit_default)
      self._cache[cls] = method
    return method


if sys.version_info[0] == 2:
  # Python 2 doesn't support non-ASCII identifiers, and making the real patched_generate_tokens support Python 2
  # means working with raw tuples instead of tokenize.TokenInfo namedtuples.
  def patched_generate_tokens(original_tokens):
    # type: (Iterable[TokenInfo]) -> Iterator[TokenInfo]
    return iter(original_tokens)
else:
  def patched_generate_tokens(original_tokens):
    # type: (Iterable[TokenInfo]) -> Iterator[TokenInfo]
    """
    Fixes tokens yielded by `tokenize.generate_tokens` to handle more non-ASCII characters in identifiers.
    Workaround for https://github.com/python/cpython/issues/68382.
    Should only be used when tokenizing a string that is known to be valid syntax,
    because it assumes that error tokens are not actually errors.
    Combines groups of consecutive NAME, NUMBER, and/or ERRORTOKEN tokens into a single NAME token.
    """
    group = []  # type: List[tokenize.TokenInfo]
    for tok in original_tokens:
      if (
          tok.type in (tokenize.NAME, tokenize.ERRORTOKEN, tokenize.NUMBER)
          # Only combine tokens if they have no whitespace in between
          and (not group or group[-1].end == tok.start)
      ):
        group.append(tok)
      else:
        for combined_token in combine_tokens(group):
          yield combined_token
        group = []
        yield tok
    for combined_token in combine_tokens(group):
      yield combined_token

  def combine_tokens(group):
    # type: (List[tokenize.TokenInfo]) -> List[tokenize.TokenInfo]
    if not any(tok.type == tokenize.ERRORTOKEN for tok in group) or len({tok.line for tok in group}) != 1:
      return group
    return [
      tokenize.TokenInfo(
        type=tokenize.NAME,
        string="".join(t.string for t in group),
        start=group[0].start,
        end=group[-1].end,
        line=group[0].line,
      )
    ]


def last_stmt(node):
  # type: (ast.AST) -> ast.AST
  """
  If the given AST node contains multiple statements, return the last one.
  Otherwise, just return the node.
  """
  child_stmts = [
    child for child in iter_children_func(node)(node)
    if is_stmt(child) or type(child).__name__ in (
      "excepthandler",
      "ExceptHandler",
      "match_case",
      "MatchCase",
      "TryExcept",
      "TryFinally",
    )
  ]
  if child_stmts:
    return last_stmt(child_stmts[-1])
  return node


if sys.version_info[:2] >= (3, 8):
  from functools import lru_cache

  @lru_cache(maxsize=None)
  def fstring_positions_work():
    # type: () -> bool
    """
    The positions attached to nodes inside f-string FormattedValues have some bugs
    that were fixed in Python 3.9.7 in https://github.com/python/cpython/pull/27729.
    This checks for those bugs more concretely without relying on the Python version.
    Specifically this checks:
     - Values with a format spec or conversion
     - Repeated (i.e. identical-looking) expressions
     - f-strings implicitly concatenated over multiple lines.
     - Multiline, triple-quoted f-strings.
    """
    source = """(
      f"a {b}{b} c {d!r} e {f:g} h {i:{j}} k {l:{m:n}}"
      f"a {b}{b} c {d!r} e {f:g} h {i:{j}} k {l:{m:n}}"
      f"{x + y + z} {x} {y} {z} {z} {z!a} {z:z}"
      f'''
      {s} {t}
      {u} {v}
      '''
    )"""
    tree = ast.parse(source)
    name_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
    name_positions = [(node.lineno, node.col_offset) for node in name_nodes]
    positions_are_unique = len(set(name_positions)) == len(name_positions)
    correct_source_segments = all(
      ast.get_source_segment(source, node) == node.id
      for node in name_nodes
    )
    return positions_are_unique and correct_source_segments

  def annotate_fstring_nodes(tree):
    # type: (ast.AST) -> None
    """
    Add a special attribute `_broken_positions` to nodes inside f-strings
    if the lineno/col_offset cannot be trusted.
    """
    if sys.version_info >= (3, 12):
      # f-strings were weirdly implemented until https://peps.python.org/pep-0701/
      # In Python 3.12, inner nodes have sensible positions.
      return
    for joinedstr in walk(tree, include_joined_str=True):
      if not isinstance(joinedstr, ast.JoinedStr):
        continue
      for part in joinedstr.values:
        # The ast positions of the FormattedValues/Constant nodes span the full f-string, which is weird.
        setattr(part, '_broken_positions', True)  # use setattr for mypy

        if isinstance(part, ast.FormattedValue):
          if not fstring_positions_work():
            for child in walk(part.value):
              setattr(child, '_broken_positions', True)

          if part.format_spec:  # this is another JoinedStr
            # Again, the standard positions span the full f-string.
            setattr(part.format_spec, '_broken_positions', True)

else:
  def fstring_positions_work():
    # type: () -> bool
    return False

  def annotate_fstring_nodes(_tree):
    # type: (ast.AST) -> None
    pass
