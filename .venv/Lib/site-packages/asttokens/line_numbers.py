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

import bisect
import re
from typing import Dict, List, Tuple

_line_start_re = re.compile(r'^', re.M)

class LineNumbers(object):
  """
  Class to convert between character offsets in a text string, and pairs (line, column) of 1-based
  line and 0-based column numbers, as used by tokens and AST nodes.

  This class expects unicode for input and stores positions in unicode. But it supports
  translating to and from utf8 offsets, which are used by ast parsing.
  """
  def __init__(self, text):
    # type: (str) -> None
    # A list of character offsets of each line's first character.
    self._line_offsets = [m.start(0) for m in _line_start_re.finditer(text)]
    self._text = text
    self._text_len = len(text)
    self._utf8_offset_cache = {} # type: Dict[int, List[int]] # maps line num to list of char offset for each byte in line

  def from_utf8_col(self, line, utf8_column):
    # type: (int, int) -> int
    """
    Given a 1-based line number and 0-based utf8 column, returns a 0-based unicode column.
    """
    offsets = self._utf8_offset_cache.get(line)
    if offsets is None:
      end_offset = self._line_offsets[line] if line < len(self._line_offsets) else self._text_len
      line_text = self._text[self._line_offsets[line - 1] : end_offset]

      offsets = [i for i,c in enumerate(line_text) for byte in c.encode('utf8')]
      offsets.append(len(line_text))
      self._utf8_offset_cache[line] = offsets

    return offsets[max(0, min(len(offsets)-1, utf8_column))]

  def line_to_offset(self, line, column):
    # type: (int, int) -> int
    """
    Converts 1-based line number and 0-based column to 0-based character offset into text.
    """
    line -= 1
    if line >= len(self._line_offsets):
      return self._text_len
    elif line < 0:
      return 0
    else:
      return min(self._line_offsets[line] + max(0, column), self._text_len)

  def offset_to_line(self, offset):
    # type: (int) -> Tuple[int, int]
    """
    Converts 0-based character offset to pair (line, col) of 1-based line and 0-based column
    numbers.
    """
    offset = max(0, min(self._text_len, offset))
    line_index = bisect.bisect_right(self._line_offsets, offset) - 1
    return (line_index + 1, offset - self._line_offsets[line_index])


