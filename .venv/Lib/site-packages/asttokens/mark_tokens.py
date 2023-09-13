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
import numbers
import sys
import token
from ast import Module
from typing import Callable, List, Union, cast, Optional, Tuple, TYPE_CHECKING

import six

from . import util
from .asttokens import ASTTokens
from .util import AstConstant
from .astroid_compat import astroid_node_classes as nc

if TYPE_CHECKING:
  from .util import AstNode


# Mapping of matching braces. To find a token here, look up token[:2].
_matching_pairs_left = {
  (token.OP, '('): (token.OP, ')'),
  (token.OP, '['): (token.OP, ']'),
  (token.OP, '{'): (token.OP, '}'),
}

_matching_pairs_right = {
  (token.OP, ')'): (token.OP, '('),
  (token.OP, ']'): (token.OP, '['),
  (token.OP, '}'): (token.OP, '{'),
}


class MarkTokens(object):
  """
  Helper that visits all nodes in the AST tree and assigns .first_token and .last_token attributes
  to each of them. This is the heart of the token-marking logic.
  """
  def __init__(self, code):
    # type: (ASTTokens) -> None
    self._code = code
    self._methods = util.NodeMethods()
    self._iter_children = None # type: Optional[Callable]

  def visit_tree(self, node):
    # type: (Module) -> None
    self._iter_children = util.iter_children_func(node)
    util.visit_tree(node, self._visit_before_children, self._visit_after_children)

  def _visit_before_children(self, node, parent_token):
    # type: (AstNode, Optional[util.Token]) -> Tuple[Optional[util.Token], Optional[util.Token]]
    col = getattr(node, 'col_offset', None)
    token = self._code.get_token_from_utf8(node.lineno, col) if col is not None else None

    if not token and util.is_module(node):
      # We'll assume that a Module node starts at the start of the source code.
      token = self._code.get_token(1, 0)

    # Use our own token, or our parent's if we don't have one, to pass to child calls as
    # parent_token argument. The second value becomes the token argument of _visit_after_children.
    return (token or parent_token, token)

  def _visit_after_children(self, node, parent_token, token):
    # type: (AstNode, Optional[util.Token], Optional[util.Token]) -> None
    # This processes the node generically first, after all children have been processed.

    # Get the first and last tokens that belong to children. Note how this doesn't assume that we
    # iterate through children in order that corresponds to occurrence in source code. This
    # assumption can fail (e.g. with return annotations).
    first = token
    last = None
    for child in cast(Callable, self._iter_children)(node):
      # astroid slices have especially wrong positions, we don't want them to corrupt their parents.
      if util.is_empty_astroid_slice(child):
        continue
      if not first or child.first_token.index < first.index:
        first = child.first_token
      if not last or child.last_token.index > last.index:
        last = child.last_token

    # If we don't have a first token from _visit_before_children, and there were no children, then
    # use the parent's token as the first token.
    first = first or parent_token

    # If no children, set last token to the first one.
    last = last or first

    # Statements continue to before NEWLINE. This helps cover a few different cases at once.
    if util.is_stmt(node):
      last = self._find_last_in_stmt(cast(util.Token, last))

    # Capture any unmatched brackets.
    first, last = self._expand_to_matching_pairs(cast(util.Token, first), cast(util.Token, last), node)

    # Give a chance to node-specific methods to adjust.
    nfirst, nlast = self._methods.get(self, node.__class__)(node, first, last)

    if (nfirst, nlast) != (first, last):
      # If anything changed, expand again to capture any unmatched brackets.
      nfirst, nlast = self._expand_to_matching_pairs(nfirst, nlast, node)

    node.first_token = nfirst
    node.last_token = nlast

  def _find_last_in_stmt(self, start_token):
    # type: (util.Token) -> util.Token
    t = start_token
    while (not util.match_token(t, token.NEWLINE) and
           not util.match_token(t, token.OP, ';') and
           not token.ISEOF(t.type)):
      t = self._code.next_token(t, include_extra=True)
    return self._code.prev_token(t)

  def _expand_to_matching_pairs(self, first_token, last_token, node):
    # type: (util.Token, util.Token, AstNode) -> Tuple[util.Token, util.Token]
    """
    Scan tokens in [first_token, last_token] range that are between node's children, and for any
    unmatched brackets, adjust first/last tokens to include the closing pair.
    """
    # We look for opening parens/braces among non-child tokens (i.e. tokens between our actual
    # child nodes). If we find any closing ones, we match them to the opens.
    to_match_right = [] # type: List[Tuple[int, str]]
    to_match_left = []
    for tok in self._code.token_range(first_token, last_token):
      tok_info = tok[:2]
      if to_match_right and tok_info == to_match_right[-1]:
        to_match_right.pop()
      elif tok_info in _matching_pairs_left:
        to_match_right.append(_matching_pairs_left[tok_info])
      elif tok_info in _matching_pairs_right:
        to_match_left.append(_matching_pairs_right[tok_info])

    # Once done, extend `last_token` to match any unclosed parens/braces.
    for match in reversed(to_match_right):
      last = self._code.next_token(last_token)
      # Allow for trailing commas or colons (allowed in subscripts) before the closing delimiter
      while any(util.match_token(last, token.OP, x) for x in (',', ':')):
        last = self._code.next_token(last)
      # Now check for the actual closing delimiter.
      if util.match_token(last, *match):
        last_token = last

    # And extend `first_token` to match any unclosed opening parens/braces.
    for match in to_match_left:
      first = self._code.prev_token(first_token)
      if util.match_token(first, *match):
        first_token = first

    return (first_token, last_token)

  #----------------------------------------------------------------------
  # Node visitors. Each takes a preliminary first and last tokens, and returns the adjusted pair
  # that will actually be assigned.

  def visit_default(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # pylint: disable=no-self-use
    # By default, we don't need to adjust the token we computed earlier.
    return (first_token, last_token)

  def handle_comp(self, open_brace, node, first_token, last_token):
    # type: (str, AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # For list/set/dict comprehensions, we only get the token of the first child, so adjust it to
    # include the opening brace (the closing brace will be matched automatically).
    before = self._code.prev_token(first_token)
    util.expect_token(before, token.OP, open_brace)
    return (before, last_token)

  # Python 3.8 fixed the starting position of list comprehensions:
  # https://bugs.python.org/issue31241
  if sys.version_info < (3, 8):
    def visit_listcomp(self, node, first_token, last_token):
      # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
      return self.handle_comp('[', node, first_token, last_token)

  if six.PY2:
    # We shouldn't do this on PY3 because its SetComp/DictComp already have a correct start.
    def visit_setcomp(self, node, first_token, last_token):
      # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
      return self.handle_comp('{', node, first_token, last_token)

    def visit_dictcomp(self, node, first_token, last_token):
      # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
      return self.handle_comp('{', node, first_token, last_token)

  def visit_comprehension(self,
                          node,  # type: AstNode
                          first_token,  # type: util.Token
                          last_token,  # type: util.Token
                          ):
    # type: (...) -> Tuple[util.Token, util.Token]
    # The 'comprehension' node starts with 'for' but we only get first child; we search backwards
    # to find the 'for' keyword.
    first = self._code.find_token(first_token, token.NAME, 'for', reverse=True)
    return (first, last_token)

  def visit_if(self, node, first_token, last_token):
    # type: (util.Token, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    while first_token.string not in ('if', 'elif'):
      first_token = self._code.prev_token(first_token)
    return first_token, last_token

  def handle_attr(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # Attribute node has ".attr" (2 tokens) after the last child.
    dot = self._code.find_token(last_token, token.OP, '.')
    name = self._code.next_token(dot)
    util.expect_token(name, token.NAME)
    return (first_token, name)

  visit_attribute = handle_attr
  visit_assignattr = handle_attr
  visit_delattr = handle_attr

  def handle_def(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # With astroid, nodes that start with a doc-string can have an empty body, in which case we
    # need to adjust the last token to include the doc string.
    if not node.body and getattr(node, 'doc', None): # type: ignore[union-attr]
      last_token = self._code.find_token(last_token, token.STRING)

    # Include @ from decorator
    if first_token.index > 0:
      prev = self._code.prev_token(first_token)
      if util.match_token(prev, token.OP, '@'):
        first_token = prev
    return (first_token, last_token)

  visit_classdef = handle_def
  visit_functiondef = handle_def

  def handle_following_brackets(self, node, last_token, opening_bracket):
    # type: (AstNode, util.Token, str) -> util.Token
    # This is for calls and subscripts, which have a pair of brackets
    # at the end which may contain no nodes, e.g. foo() or bar[:].
    # We look for the opening bracket and then let the matching pair be found automatically
    # Remember that last_token is at the end of all children,
    # so we are not worried about encountering a bracket that belongs to a child.
    first_child = next(cast(Callable, self._iter_children)(node))
    call_start = self._code.find_token(first_child.last_token, token.OP, opening_bracket)
    if call_start.index > last_token.index:
      last_token = call_start
    return last_token

  def visit_call(self, node, first_token, last_token):
    # type: (util.Token, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    last_token = self.handle_following_brackets(node, last_token, '(')

    # Handling a python bug with decorators with empty parens, e.g.
    # @deco()
    # def ...
    if util.match_token(first_token, token.OP, '@'):
      first_token = self._code.next_token(first_token)
    return (first_token, last_token)

  def visit_matchclass(self, node, first_token, last_token):
    # type: (util.Token, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    last_token = self.handle_following_brackets(node, last_token, '(')
    return (first_token, last_token)

  def visit_subscript(self,
                      node,  # type: AstNode
                      first_token,  # type: util.Token
                      last_token,  # type: util.Token
                      ):
    # type: (...) -> Tuple[util.Token, util.Token]
    last_token = self.handle_following_brackets(node, last_token, '[')
    return (first_token, last_token)

  def visit_slice(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # consume `:` tokens to the left and right. In Python 3.9, Slice nodes are
    # given a col_offset, (and end_col_offset), so this will always start inside
    # the slice, even if it is the empty slice. However, in 3.8 and below, this
    # will only expand to the full slice if the slice contains a node with a
    # col_offset. So x[:] will only get the correct tokens in 3.9, but x[1:] and
    # x[:1] will even on earlier versions of Python.
    while True:
      prev = self._code.prev_token(first_token)
      if prev.string != ':':
        break
      first_token = prev
    while True:
      next_ = self._code.next_token(last_token)
      if next_.string != ':':
        break
      last_token = next_
    return (first_token, last_token)

  def handle_bare_tuple(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # A bare tuple doesn't include parens; if there is a trailing comma, make it part of the tuple.
    maybe_comma = self._code.next_token(last_token)
    if util.match_token(maybe_comma, token.OP, ','):
      last_token = maybe_comma
    return (first_token, last_token)

  if sys.version_info >= (3, 8):
    # In Python3.8 parsed tuples include parentheses when present.
    def handle_tuple_nonempty(self, node, first_token, last_token):
      # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
      assert isinstance(node, ast.Tuple) or isinstance(node, nc._BaseContainer)
      # It's a bare tuple if the first token belongs to the first child. The first child may
      # include extraneous parentheses (which don't create new nodes), so account for those too.
      child = node.elts[0]
      if TYPE_CHECKING:
        child = cast(AstNode, child)
      child_first, child_last = self._gobble_parens(child.first_token, child.last_token, True)
      if first_token == child_first:
        return self.handle_bare_tuple(node, first_token, last_token)
      return (first_token, last_token)
  else:
    # Before python 3.8, parsed tuples do not include parens.
    def handle_tuple_nonempty(self, node, first_token, last_token):
      # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
      (first_token, last_token) = self.handle_bare_tuple(node, first_token, last_token)
      return self._gobble_parens(first_token, last_token, False)

  def visit_tuple(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    assert isinstance(node, ast.Tuple) or isinstance(node, nc._BaseContainer)
    if not node.elts:
      # An empty tuple is just "()", and we need no further info.
      return (first_token, last_token)
    return self.handle_tuple_nonempty(node, first_token, last_token)

  def _gobble_parens(self, first_token, last_token, include_all=False):
    # type: (util.Token, util.Token, bool) -> Tuple[util.Token, util.Token]
    # Expands a range of tokens to include one or all pairs of surrounding parentheses, and
    # returns (first, last) tokens that include these parens.
    while first_token.index > 0:
      prev = self._code.prev_token(first_token)
      next = self._code.next_token(last_token)
      if util.match_token(prev, token.OP, '(') and util.match_token(next, token.OP, ')'):
        first_token, last_token = prev, next
        if include_all:
          continue
      break
    return (first_token, last_token)

  def visit_str(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    return self.handle_str(first_token, last_token)

  def visit_joinedstr(self,
                      node,  # type: AstNode
                      first_token,  # type: util.Token
                      last_token,  # type: util.Token
                      ):
    # type: (...) -> Tuple[util.Token, util.Token]
    if sys.version_info < (3, 12):
      # Older versions don't tokenize the contents of f-strings
      return self.handle_str(first_token, last_token)

    last = first_token
    while True:
      if util.match_token(last, getattr(token, "FSTRING_START")):
        # Python 3.12+ has tokens for the start (e.g. `f"`) and end (`"`)
        # of the f-string. We can't just look for the next FSTRING_END
        # because f-strings can be nested, e.g. f"{f'{x}'}", so we need
        # to treat this like matching balanced parentheses.
        count = 1
        while count > 0:
          last = self._code.next_token(last)
          # mypy complains about token.FSTRING_START and token.FSTRING_END.
          if util.match_token(last, getattr(token, "FSTRING_START")):
            count += 1
          elif util.match_token(last, getattr(token, "FSTRING_END")):
            count -= 1
        last_token = last
        last = self._code.next_token(last_token)
      elif util.match_token(last, token.STRING):
        # Similar to handle_str, we also need to handle adjacent strings.
        last_token = last
        last = self._code.next_token(last_token)
      else:
        break
    return (first_token, last_token)

  def visit_bytes(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    return self.handle_str(first_token, last_token)

  def handle_str(self, first_token, last_token):
    # type: (util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # Multiple adjacent STRING tokens form a single string.
    last = self._code.next_token(last_token)
    while util.match_token(last, token.STRING):
      last_token = last
      last = self._code.next_token(last_token)
    return (first_token, last_token)

  def handle_num(self,
                 node,  # type: AstNode
                 value,  # type: Union[complex, int, numbers.Number]
                 first_token,  # type: util.Token
                 last_token,  # type: util.Token
                 ):
    # type: (...) -> Tuple[util.Token, util.Token]
    # A constant like '-1' gets turned into two tokens; this will skip the '-'.
    while util.match_token(last_token, token.OP):
      last_token = self._code.next_token(last_token)

    if isinstance(value, complex):
      # A complex number like -2j cannot be compared directly to 0
      # A complex number like 1-2j is expressed as a binary operation
      # so we don't need to worry about it
      value = value.imag

    # This makes sure that the - is included
    if value < 0 and first_token.type == token.NUMBER: # type: ignore[operator]
        first_token = self._code.prev_token(first_token)
    return (first_token, last_token)

  def visit_num(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    return self.handle_num(node, cast(ast.Num, node).n, first_token, last_token)

  # In Astroid, the Num and Str nodes are replaced by Const.
  def visit_const(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    assert isinstance(node, AstConstant) or isinstance(node, nc.Const)
    if isinstance(node.value, numbers.Number):
      return self.handle_num(node, node.value, first_token, last_token)
    elif isinstance(node.value, (six.text_type, six.binary_type)):
      return self.visit_str(node, first_token, last_token)
    return (first_token, last_token)

  # In Python >= 3.6, there is a similar class 'Constant' for literals
  # In 3.8 it became the type produced by ast.parse
  # https://bugs.python.org/issue32892
  visit_constant = visit_const

  def visit_keyword(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # Until python 3.9 (https://bugs.python.org/issue40141),
    # ast.keyword nodes didn't have line info. Astroid has lineno None.
    assert isinstance(node, ast.keyword) or isinstance(node, nc.Keyword)
    if node.arg is not None and getattr(node, 'lineno', None) is None:
      equals = self._code.find_token(first_token, token.OP, '=', reverse=True)
      name = self._code.prev_token(equals)
      util.expect_token(name, token.NAME, node.arg)
      first_token = name
    return (first_token, last_token)

  def visit_starred(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # Astroid has 'Starred' nodes (for "foo(*bar)" type args), but they need to be adjusted.
    if not util.match_token(first_token, token.OP, '*'):
      star = self._code.prev_token(first_token)
      if util.match_token(star, token.OP, '*'):
        first_token = star
    return (first_token, last_token)

  def visit_assignname(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    # Astroid may turn 'except' clause into AssignName, but we need to adjust it.
    if util.match_token(first_token, token.NAME, 'except'):
      colon = self._code.find_token(last_token, token.OP, ':')
      first_token = last_token = self._code.prev_token(colon)
    return (first_token, last_token)

  if six.PY2:
    # No need for this on Python3, which already handles 'with' nodes correctly.
    def visit_with(self, node, first_token, last_token):
      # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
      first = self._code.find_token(first_token, token.NAME, 'with', reverse=True)
      return (first, last_token)

  # Async nodes should typically start with the word 'async'
  # but Python < 3.7 doesn't put the col_offset there
  # AsyncFunctionDef is slightly different because it might have
  # decorators before that, which visit_functiondef handles
  def handle_async(self, node, first_token, last_token):
    # type: (AstNode, util.Token, util.Token) -> Tuple[util.Token, util.Token]
    if not first_token.string == 'async':
      first_token = self._code.prev_token(first_token)
    return (first_token, last_token)

  visit_asyncfor = handle_async
  visit_asyncwith = handle_async

  def visit_asyncfunctiondef(self,
                             node,  # type: AstNode
                             first_token,  # type: util.Token
                             last_token,  # type: util.Token
                             ):
    # type: (...) -> Tuple[util.Token, util.Token]
    if util.match_token(first_token, token.NAME, 'def'):
      # Include the 'async' token
      first_token = self._code.prev_token(first_token)
    return self.visit_functiondef(node, first_token, last_token)
