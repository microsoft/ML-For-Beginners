import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
    Iterator, List, Tuple, Iterable, Callable, Union,
    TypeVar, Mapping,
)

from asttokens import ASTText

T = TypeVar('T')
R = TypeVar('R')


def truncate(seq, max_length: int, middle):
    if len(seq) > max_length:
        right = (max_length - len(middle)) // 2
        left = max_length - len(middle) - right
        seq = seq[:left] + middle + seq[-right:]
    return seq


def unique_in_order(it: Iterable[T]) -> List[T]:
    return list(OrderedDict.fromkeys(it))


def line_range(atok: ASTText, node: ast.AST) -> Tuple[int, int]:
    """
    Returns a pair of numbers representing a half open range
    (i.e. suitable as arguments to the `range()` builtin)
    of line numbers of the given AST nodes.
    """
    if isinstance(node, getattr(ast, "match_case", ())):
        start, _end = line_range(atok, node.pattern)
        _start, end = line_range(atok, node.body[-1])
        return start, end
    else:
        (start, _), (end, _) = atok.get_text_positions(node, padded=False)
        return start, end + 1


def highlight_unique(lst: List[T]) -> Iterator[Tuple[T, bool]]:
    counts = Counter(lst)

    for is_common, group in itertools.groupby(lst, key=lambda x: counts[x] > 3):
        if is_common:
            group = list(group)
            highlighted = [False] * len(group)

            def highlight_index(f):
                try:
                    i = f()
                except ValueError:
                    return None
                highlighted[i] = True
                return i

            for item in set(group):
                first = highlight_index(lambda: group.index(item))
                if first is not None:
                    highlight_index(lambda: group.index(item, first + 1))
                highlight_index(lambda: -1 - group[::-1].index(item))
        else:
            highlighted = itertools.repeat(True)

        yield from zip(group, highlighted)


def identity(x: T) -> T:
    return x


def collapse_repeated(lst, *, collapser, mapper=identity, key=identity):
    keyed = list(map(key, lst))
    for is_highlighted, group in itertools.groupby(
            zip(lst, highlight_unique(keyed)),
            key=lambda t: t[1][1],
    ):
        original_group, highlighted_group = zip(*group)
        if is_highlighted:
            yield from map(mapper, original_group)
        else:
            keyed_group, _ = zip(*highlighted_group)
            yield collapser(list(original_group), list(keyed_group))


def is_frame(frame_or_tb: Union[FrameType, TracebackType]) -> bool:
    assert_(isinstance(frame_or_tb, (types.FrameType, types.TracebackType)))
    return isinstance(frame_or_tb, (types.FrameType,))


def iter_stack(frame_or_tb: Union[FrameType, TracebackType]) -> Iterator[Union[FrameType, TracebackType]]:
    while frame_or_tb:
        yield frame_or_tb
        if is_frame(frame_or_tb):
            frame_or_tb = frame_or_tb.f_back
        else:
            frame_or_tb = frame_or_tb.tb_next


def frame_and_lineno(frame_or_tb: Union[FrameType, TracebackType]) -> Tuple[FrameType, int]:
    if is_frame(frame_or_tb):
        return frame_or_tb, frame_or_tb.f_lineno
    else:
        return frame_or_tb.tb_frame, frame_or_tb.tb_lineno


def group_by_key_func(iterable: Iterable[T], key_func: Callable[[T], R]) -> Mapping[R, List[T]]:
    # noinspection PyUnresolvedReferences
    """
    Create a dictionary from an iterable such that the keys are the result of evaluating a key function on elements
    of the iterable and the values are lists of elements all of which correspond to the key.

    >>> def si(d): return sorted(d.items())
    >>> si(group_by_key_func("a bb ccc d ee fff".split(), len))
    [(1, ['a', 'd']), (2, ['bb', 'ee']), (3, ['ccc', 'fff'])]
    >>> si(group_by_key_func([-1, 0, 1, 3, 6, 8, 9, 2], lambda x: x % 2))
    [(0, [0, 6, 8, 2]), (1, [-1, 1, 3, 9])]
    """
    result = defaultdict(list)
    for item in iterable:
        result[key_func(item)].append(item)
    return result


class cached_property(object):
    """
    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Deleting the attribute resets the property.

    Based on https://github.com/pydanny/cached-property/blob/master/cached_property.py
    """

    def __init__(self, func):
        self.__doc__ = func.__doc__
        self.func = func

    def cached_property_wrapper(self, obj, _cls):
        if obj is None:
            return self

        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

    __get__ = cached_property_wrapper


def _pygmented_with_ranges(formatter, code, ranges):
    import pygments
    from pygments.lexers import get_lexer_by_name

    class MyLexer(type(get_lexer_by_name("python3"))):
        def get_tokens(self, text):
            length = 0
            for ttype, value in super().get_tokens(text):
                if any(start <= length < end for start, end in ranges):
                    ttype = ttype.ExecutingNode
                length += len(value)
                yield ttype, value

    lexer = MyLexer(stripnl=False)
    try:
        highlighted = pygments.highlight(code, lexer, formatter)
    except Exception:
        # When pygments fails, prefer code without highlighting over crashing
        highlighted = code
    return highlighted.splitlines()


def assert_(condition, error=""):
    if not condition:
        if isinstance(error, str):
            error = AssertionError(error)
        raise error


# Copied from the standard traceback module pre-3.11
def some_str(value):
    try:
        return str(value)
    except:
        return '<unprintable %s object>' % type(value).__name__
