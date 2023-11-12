# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import inspect
import locale
import os
import pydoc
import re
import textwrap
import warnings
from collections import defaultdict, deque
from itertools import chain, combinations, islice, tee
from pprint import pprint
from urllib.request import (
    HTTPPasswordMgrWithDefaultRealm,
    ProxyBasicAuthHandler,
    ProxyDigestAuthHandler,
    ProxyHandler,
    build_opener,
    getproxies,
    install_opener,
)

from nltk.collections import *
from nltk.internals import deprecated, raise_unorderable_types, slice_bounds

######################################################################
# Short usage message
######################################################################


@deprecated("Use help(obj) instead.")
def usage(obj):
    str(obj)  # In case it's lazy, this will load it.

    if not isinstance(obj, type):
        obj = obj.__class__

    print(f"{obj.__name__} supports the following operations:")
    for (name, method) in sorted(pydoc.allmethods(obj).items()):
        if name.startswith("_"):
            continue
        if getattr(method, "__deprecated__", False):
            continue

        try:
            sig = str(inspect.signature(method))
        except ValueError as e:
            # builtins sometimes don't support introspection
            if "builtin" in str(e):
                continue
            else:
                raise

        args = sig.lstrip("(").rstrip(")").split(", ")
        meth = inspect.getattr_static(obj, name)
        if isinstance(meth, (classmethod, staticmethod)):
            name = f"cls.{name}"
        elif args and args[0] == "self":
            name = f"self.{name}"
            args.pop(0)
        print(
            textwrap.fill(
                f"{name}({', '.join(args)})",
                initial_indent="  - ",
                subsequent_indent=" " * (len(name) + 5),
            )
        )


##########################################################################
# IDLE
##########################################################################


def in_idle():
    """
    Return True if this function is run within idle.  Tkinter
    programs that are run in idle should never call ``Tk.mainloop``; so
    this function should be used to gate all calls to ``Tk.mainloop``.

    :warning: This function works by checking ``sys.stdin``.  If the
        user has modified ``sys.stdin``, then it may return incorrect
        results.
    :rtype: bool
    """
    import sys

    return sys.stdin.__class__.__name__ in ("PyShell", "RPCProxy")


##########################################################################
# PRETTY PRINTING
##########################################################################


def pr(data, start=0, end=None):
    """
    Pretty print a sequence of data items

    :param data: the data stream to print
    :type data: sequence or iter
    :param start: the start position
    :type start: int
    :param end: the end position
    :type end: int
    """
    pprint(list(islice(data, start, end)))


def print_string(s, width=70):
    """
    Pretty print a string, breaking lines on whitespace

    :param s: the string to print, consisting of words and spaces
    :type s: str
    :param width: the display width
    :type width: int
    """
    print("\n".join(textwrap.wrap(s, width=width)))


def tokenwrap(tokens, separator=" ", width=70):
    """
    Pretty print a list of text tokens, breaking lines on whitespace

    :param tokens: the tokens to print
    :type tokens: list
    :param separator: the string to use to separate tokens
    :type separator: str
    :param width: the display width (default=70)
    :type width: int
    """
    return "\n".join(textwrap.wrap(separator.join(tokens), width=width))


##########################################################################
# Indexing
##########################################################################


class Index(defaultdict):
    def __init__(self, pairs):
        defaultdict.__init__(self, list)
        for key, value in pairs:
            self[key].append(value)


######################################################################
## Regexp display (thanks to David Mertz)
######################################################################


def re_show(regexp, string, left="{", right="}"):
    """
    Return a string with markers surrounding the matched substrings.
    Search str for substrings matching ``regexp`` and wrap the matches
    with braces.  This is convenient for learning about regular expressions.

    :param regexp: The regular expression.
    :type regexp: str
    :param string: The string being matched.
    :type string: str
    :param left: The left delimiter (printed before the matched substring)
    :type left: str
    :param right: The right delimiter (printed after the matched substring)
    :type right: str
    :rtype: str
    """
    print(re.compile(regexp, re.M).sub(left + r"\g<0>" + right, string.rstrip()))


##########################################################################
# READ FROM FILE OR STRING
##########################################################################

# recipe from David Mertz
def filestring(f):
    if hasattr(f, "read"):
        return f.read()
    elif isinstance(f, str):
        with open(f) as infile:
            return infile.read()
    else:
        raise ValueError("Must be called with a filename or file-like object")


##########################################################################
# Breadth-First Search
##########################################################################


def breadth_first(tree, children=iter, maxdepth=-1):
    """Traverse the nodes of a tree in breadth-first order.
    (No check for cycles.)
    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.
    """
    queue = deque([(tree, 0)])

    while queue:
        node, depth = queue.popleft()
        yield node

        if depth != maxdepth:
            try:
                queue.extend((c, depth + 1) for c in children(node))
            except TypeError:
                pass


##########################################################################
# Graph Drawing
##########################################################################


def edge_closure(tree, children=iter, maxdepth=-1, verbose=False):
    """Yield the edges of a graph in breadth-first order,
    discarding eventual cycles.
    The first argument should be the start node;
    children should be a function taking as argument a graph node
    and returning an iterator of the node's children.

    >>> from nltk.util import edge_closure
    >>> print(list(edge_closure('A', lambda node:{'A':['B','C'], 'B':'C', 'C':'B'}[node])))
    [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B')]
    """
    traversed = set()
    edges = set()
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        traversed.add(node)
        if depth != maxdepth:
            try:
                for child in children(node):
                    if child not in traversed:
                        queue.append((child, depth + 1))
                    else:
                        if verbose:
                            warnings.warn(
                                f"Discarded redundant search for {child} at depth {depth + 1}",
                                stacklevel=2,
                            )
                    edge = (node, child)
                    if edge not in edges:
                        yield edge
                        edges.add(edge)
            except TypeError:
                pass


def edges2dot(edges, shapes=None, attr=None):
    """
    :param edges: the set (or list) of edges of a directed graph.

    :return dot_string: a representation of 'edges' as a string in the DOT
        graph language, which can be converted to an image by the 'dot' program
        from the Graphviz package, or nltk.parse.dependencygraph.dot2img(dot_string).

    :param shapes: dictionary of strings that trigger a specified shape.
    :param attr: dictionary with global graph attributes

    >>> import nltk
    >>> from nltk.util import edges2dot
    >>> print(edges2dot([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B')]))
    digraph G {
    "A" -> "B";
    "A" -> "C";
    "B" -> "C";
    "C" -> "B";
    }
    <BLANKLINE>
    """
    if not shapes:
        shapes = dict()
    if not attr:
        attr = dict()

    dot_string = "digraph G {\n"

    for pair in attr.items():
        dot_string += f"{pair[0]} = {pair[1]};\n"

    for edge in edges:
        for shape in shapes.items():
            for node in range(2):
                if shape[0] in repr(edge[node]):
                    dot_string += f'"{edge[node]}" [shape = {shape[1]}];\n'
        dot_string += f'"{edge[0]}" -> "{edge[1]}";\n'

    dot_string += "}\n"
    return dot_string


def unweighted_minimum_spanning_digraph(tree, children=iter, shapes=None, attr=None):
    """

    Build a Minimum Spanning Tree (MST) of an unweighted graph,
    by traversing the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    Return a representation of this MST as a string in the DOT graph language,
    which can be converted to an image by the 'dot' program from the Graphviz
    package, or nltk.parse.dependencygraph.dot2img(dot_string).

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    >>> import nltk
    >>> wn=nltk.corpus.wordnet
    >>> from nltk.util import unweighted_minimum_spanning_digraph as umsd
    >>> print(umsd(wn.synset('bound.a.01'), lambda s:s.also_sees()))
    digraph G {
    "Synset('bound.a.01')" -> "Synset('unfree.a.02')";
    "Synset('unfree.a.02')" -> "Synset('confined.a.02')";
    "Synset('unfree.a.02')" -> "Synset('dependent.a.01')";
    "Synset('unfree.a.02')" -> "Synset('restricted.a.01')";
    "Synset('restricted.a.01')" -> "Synset('classified.a.02')";
    }
    <BLANKLINE>
    """
    return edges2dot(
        edge_closure(
            tree, lambda node: unweighted_minimum_spanning_dict(tree, children)[node]
        ),
        shapes,
        attr,
    )


##########################################################################
# Breadth-First / Depth-first Searches with Cycle Detection
##########################################################################


def acyclic_breadth_first(tree, children=iter, maxdepth=-1):
    """Traverse the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.
    """
    traversed = set()
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        yield node
        traversed.add(node)
        if depth != maxdepth:
            try:
                for child in children(node):
                    if child not in traversed:
                        queue.append((child, depth + 1))
                    else:
                        warnings.warn(
                            "Discarded redundant search for {} at depth {}".format(
                                child, depth + 1
                            ),
                            stacklevel=2,
                        )
            except TypeError:
                pass


def acyclic_depth_first(tree, children=iter, depth=-1, cut_mark=None, traversed=None):
    """Traverse the nodes of a tree in depth-first order,
    discarding eventual cycles within any branch,
    adding cut_mark (when specified) if cycles were truncated.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    Catches all cycles:

    >>> import nltk
    >>> from nltk.util import acyclic_depth_first as acyclic_tree
    >>> wn=nltk.corpus.wordnet
    >>> from pprint import pprint
    >>> pprint(acyclic_tree(wn.synset('dog.n.01'), lambda s:s.hypernyms(),cut_mark='...'))
    [Synset('dog.n.01'),
     [Synset('canine.n.02'),
      [Synset('carnivore.n.01'),
       [Synset('placental.n.01'),
        [Synset('mammal.n.01'),
         [Synset('vertebrate.n.01'),
          [Synset('chordate.n.01'),
           [Synset('animal.n.01'),
            [Synset('organism.n.01'),
             [Synset('living_thing.n.01'),
              [Synset('whole.n.02'),
               [Synset('object.n.01'),
                [Synset('physical_entity.n.01'),
                 [Synset('entity.n.01')]]]]]]]]]]]]],
     [Synset('domestic_animal.n.01'), "Cycle(Synset('animal.n.01'),-3,...)"]]
    """
    if traversed is None:
        traversed = {tree}
    out_tree = [tree]
    if depth != 0:
        try:
            for child in children(tree):
                if child not in traversed:
                    # Recurse with a common "traversed" set for all children:
                    traversed.add(child)
                    out_tree += [
                        acyclic_depth_first(
                            child, children, depth - 1, cut_mark, traversed
                        )
                    ]
                else:
                    warnings.warn(
                        "Discarded redundant search for {} at depth {}".format(
                            child, depth - 1
                        ),
                        stacklevel=3,
                    )
                    if cut_mark:
                        out_tree += [f"Cycle({child},{depth - 1},{cut_mark})"]
        except TypeError:
            pass
    elif cut_mark:
        out_tree += [cut_mark]
    return out_tree


def acyclic_branches_depth_first(
    tree, children=iter, depth=-1, cut_mark=None, traversed=None
):
    """Traverse the nodes of a tree in depth-first order,
    discarding eventual cycles within the same branch,
    but keep duplicate paths in different branches.
    Add cut_mark (when defined) if cycles were truncated.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    Catches only only cycles within the same branch,
    but keeping cycles from different branches:

    >>> import nltk
    >>> from nltk.util import acyclic_branches_depth_first as tree
    >>> wn=nltk.corpus.wordnet
    >>> from pprint import pprint
    >>> pprint(tree(wn.synset('certified.a.01'), lambda s:s.also_sees(), cut_mark='...', depth=4))
    [Synset('certified.a.01'),
     [Synset('authorized.a.01'),
      [Synset('lawful.a.01'),
       [Synset('legal.a.01'),
        "Cycle(Synset('lawful.a.01'),0,...)",
        [Synset('legitimate.a.01'), '...']],
       [Synset('straight.a.06'),
        [Synset('honest.a.01'), '...'],
        "Cycle(Synset('lawful.a.01'),0,...)"]],
      [Synset('legitimate.a.01'),
       "Cycle(Synset('authorized.a.01'),1,...)",
       [Synset('legal.a.01'),
        [Synset('lawful.a.01'), '...'],
        "Cycle(Synset('legitimate.a.01'),0,...)"],
       [Synset('valid.a.01'),
        "Cycle(Synset('legitimate.a.01'),0,...)",
        [Synset('reasonable.a.01'), '...']]],
      [Synset('official.a.01'), "Cycle(Synset('authorized.a.01'),1,...)"]],
     [Synset('documented.a.01')]]
    """
    if traversed is None:
        traversed = {tree}
    out_tree = [tree]
    if depth != 0:
        try:
            for child in children(tree):
                if child not in traversed:
                    # Recurse with a different "traversed" set for each child:
                    out_tree += [
                        acyclic_branches_depth_first(
                            child,
                            children,
                            depth - 1,
                            cut_mark,
                            traversed.union({child}),
                        )
                    ]
                else:
                    warnings.warn(
                        "Discarded redundant search for {} at depth {}".format(
                            child, depth - 1
                        ),
                        stacklevel=3,
                    )
                    if cut_mark:
                        out_tree += [f"Cycle({child},{depth - 1},{cut_mark})"]
        except TypeError:
            pass
    elif cut_mark:
        out_tree += [cut_mark]
    return out_tree


def acyclic_dic2tree(node, dic):
    """Convert acyclic dictionary 'dic', where the keys are nodes, and the
    values are lists of children, to output tree suitable for pprint(),
    starting at root 'node', with subtrees as nested lists."""
    return [node] + [acyclic_dic2tree(child, dic) for child in dic[node]]


def unweighted_minimum_spanning_dict(tree, children=iter):
    """
    Output a dictionary representing a Minimum Spanning Tree (MST)
    of an unweighted graph, by traversing the nodes of a tree in
    breadth-first order, discarding eventual cycles.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    >>> import nltk
    >>> from nltk.corpus import wordnet as wn
    >>> from nltk.util import unweighted_minimum_spanning_dict as umsd
    >>> from pprint import pprint
    >>> pprint(umsd(wn.synset('bound.a.01'), lambda s:s.also_sees()))
    {Synset('bound.a.01'): [Synset('unfree.a.02')],
     Synset('classified.a.02'): [],
     Synset('confined.a.02'): [],
     Synset('dependent.a.01'): [],
     Synset('restricted.a.01'): [Synset('classified.a.02')],
     Synset('unfree.a.02'): [Synset('confined.a.02'),
                             Synset('dependent.a.01'),
                             Synset('restricted.a.01')]}

    """
    traversed = set()  # Empty set of traversed nodes
    queue = deque([tree])  # Initialize queue
    agenda = {tree}  # Set of all nodes ever queued
    mstdic = {}  # Empty MST dictionary
    while queue:
        node = queue.popleft()  # Node is not yet in the MST dictionary,
        mstdic[node] = []  # so add it with an empty list of children
        if node not in traversed:  # Avoid cycles
            traversed.add(node)
            for child in children(node):
                if child not in agenda:  # Queue nodes only once
                    mstdic[node].append(child)  # Add child to the MST
                    queue.append(child)  # Add child to queue
                    agenda.add(child)
    return mstdic


def unweighted_minimum_spanning_tree(tree, children=iter):
    """
    Output a Minimum Spanning Tree (MST) of an unweighted graph,
    by traversing the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    >>> import nltk
    >>> from nltk.util import unweighted_minimum_spanning_tree as mst
    >>> wn=nltk.corpus.wordnet
    >>> from pprint import pprint
    >>> pprint(mst(wn.synset('bound.a.01'), lambda s:s.also_sees()))
    [Synset('bound.a.01'),
     [Synset('unfree.a.02'),
      [Synset('confined.a.02')],
      [Synset('dependent.a.01')],
      [Synset('restricted.a.01'), [Synset('classified.a.02')]]]]
    """
    return acyclic_dic2tree(tree, unweighted_minimum_spanning_dict(tree, children))


##########################################################################
# Guess Character Encoding
##########################################################################

# adapted from io.py in the docutils extension module (https://docutils.sourceforge.io/)
# http://www.pyzine.com/Issue008/Section_Articles/article_Encodings.html


def guess_encoding(data):
    """
    Given a byte string, attempt to decode it.
    Tries the standard 'UTF8' and 'latin-1' encodings,
    Plus several gathered from locale information.

    The calling program *must* first call::

        locale.setlocale(locale.LC_ALL, '')

    If successful it returns ``(decoded_unicode, successful_encoding)``.
    If unsuccessful it raises a ``UnicodeError``.
    """
    successful_encoding = None
    # we make 'utf-8' the first encoding
    encodings = ["utf-8"]
    #
    # next we add anything we can learn from the locale
    try:
        encodings.append(locale.nl_langinfo(locale.CODESET))
    except AttributeError:
        pass
    try:
        encodings.append(locale.getlocale()[1])
    except (AttributeError, IndexError):
        pass
    try:
        encodings.append(locale.getdefaultlocale()[1])
    except (AttributeError, IndexError):
        pass
    #
    # we try 'latin-1' last
    encodings.append("latin-1")
    for enc in encodings:
        # some of the locale calls
        # may have returned None
        if not enc:
            continue
        try:
            decoded = str(data, enc)
            successful_encoding = enc

        except (UnicodeError, LookupError):
            pass
        else:
            break
    if not successful_encoding:
        raise UnicodeError(
            "Unable to decode input data. "
            "Tried the following encodings: %s."
            % ", ".join([repr(enc) for enc in encodings if enc])
        )
    else:
        return (decoded, successful_encoding)


##########################################################################
# Remove repeated elements from a list deterministcally
##########################################################################


def unique_list(xs):
    seen = set()
    # not seen.add(x) here acts to make the code shorter without using if statements, seen.add(x) always returns None.
    return [x for x in xs if x not in seen and not seen.add(x)]


##########################################################################
# Invert a dictionary
##########################################################################


def invert_dict(d):
    inverted_dict = defaultdict(list)
    for key in d:
        if hasattr(d[key], "__iter__"):
            for term in d[key]:
                inverted_dict[term].append(key)
        else:
            inverted_dict[d[key]] = key
    return inverted_dict


##########################################################################
# Utilities for directed graphs: transitive closure, and inversion
# The graph is represented as a dictionary of sets
##########################################################################


def transitive_closure(graph, reflexive=False):
    """
    Calculate the transitive closure of a directed graph,
    optionally the reflexive transitive closure.

    The algorithm is a slight modification of the "Marking Algorithm" of
    Ioannidis & Ramakrishnan (1998) "Efficient Transitive Closure Algorithms".

    :param graph: the initial graph, represented as a dictionary of sets
    :type graph: dict(set)
    :param reflexive: if set, also make the closure reflexive
    :type reflexive: bool
    :rtype: dict(set)
    """
    if reflexive:
        base_set = lambda k: {k}
    else:
        base_set = lambda k: set()
    # The graph U_i in the article:
    agenda_graph = {k: graph[k].copy() for k in graph}
    # The graph M_i in the article:
    closure_graph = {k: base_set(k) for k in graph}
    for i in graph:
        agenda = agenda_graph[i]
        closure = closure_graph[i]
        while agenda:
            j = agenda.pop()
            closure.add(j)
            closure |= closure_graph.setdefault(j, base_set(j))
            agenda |= agenda_graph.get(j, base_set(j))
            agenda -= closure
    return closure_graph


def invert_graph(graph):
    """
    Inverts a directed graph.

    :param graph: the graph, represented as a dictionary of sets
    :type graph: dict(set)
    :return: the inverted graph
    :rtype: dict(set)
    """
    inverted = {}
    for key in graph:
        for value in graph[key]:
            inverted.setdefault(value, set()).add(key)
    return inverted


##########################################################################
# HTML Cleaning
##########################################################################


def clean_html(html):
    raise NotImplementedError(
        "To remove HTML markup, use BeautifulSoup's get_text() function"
    )


def clean_url(url):
    raise NotImplementedError(
        "To remove HTML markup, use BeautifulSoup's get_text() function"
    )


##########################################################################
# FLATTEN LISTS
##########################################################################


def flatten(*args):
    """
    Flatten a list.

        >>> from nltk.util import flatten
        >>> flatten(1, 2, ['b', 'a' , ['c', 'd']], 3)
        [1, 2, 'b', 'a', 'c', 'd', 3]

    :param args: items and lists to be combined into a single list
    :rtype: list
    """

    x = []
    for l in args:
        if not isinstance(l, (list, tuple)):
            l = [l]
        for item in l:
            if isinstance(item, (list, tuple)):
                x.extend(flatten(item))
            else:
                x.append(item)
    return x


##########################################################################
# Ngram iteration
##########################################################################


def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


# add a flag to pad the sequence so we get peripheral ngrams?


def ngrams(sequence, n, **kwargs):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(sequence, n, **kwargs)

    # Creates the sliding window, of n no. of items.
    # `iterables` is a tuple of iterables where each iterable is a window of n items.
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


def bigrams(sequence, **kwargs):
    """
    Return the bigrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import bigrams
        >>> list(bigrams([1,2,3,4,5]))
        [(1, 2), (2, 3), (3, 4), (4, 5)]

    Use bigrams for a list version of this function.

    :param sequence: the source data to be converted into bigrams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """

    yield from ngrams(sequence, 2, **kwargs)


def trigrams(sequence, **kwargs):
    """
    Return the trigrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import trigrams
        >>> list(trigrams([1,2,3,4,5]))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Use trigrams for a list version of this function.

    :param sequence: the source data to be converted into trigrams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """

    yield from ngrams(sequence, 3, **kwargs)


def everygrams(
    sequence, min_len=1, max_len=-1, pad_left=False, pad_right=False, **kwargs
):
    """
    Returns all possible ngrams generated from a sequence of items, as an iterator.

        >>> sent = 'a b c'.split()

    New version outputs for everygrams.
        >>> list(everygrams(sent))
        [('a',), ('a', 'b'), ('a', 'b', 'c'), ('b',), ('b', 'c'), ('c',)]

    Old version outputs for everygrams.
        >>> sorted(everygrams(sent), key=len)
        [('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c'), ('a', 'b', 'c')]

        >>> list(everygrams(sent, max_len=2))
        [('a',), ('a', 'b'), ('b',), ('b', 'c'), ('c',)]

    :param sequence: the source data to be converted into ngrams. If max_len is
        not provided, this sequence will be loaded into memory
    :type sequence: sequence or iter
    :param min_len: minimum length of the ngrams, aka. n-gram order/degree of ngram
    :type  min_len: int
    :param max_len: maximum length of the ngrams (set to length of sequence by default)
    :type  max_len: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :rtype: iter(tuple)
    """

    # Get max_len for padding.
    if max_len == -1:
        try:
            max_len = len(sequence)
        except TypeError:
            sequence = list(sequence)
            max_len = len(sequence)

    # Pad if indicated using max_len.
    sequence = pad_sequence(sequence, max_len, pad_left, pad_right, **kwargs)

    # Sliding window to store grams.
    history = list(islice(sequence, max_len))

    # Yield ngrams from sequence.
    while history:
        for ngram_len in range(min_len, len(history) + 1):
            yield tuple(history[:ngram_len])

        # Append element to history if sequence has more items.
        try:
            history.append(next(sequence))
        except StopIteration:
            pass

        del history[0]


def skipgrams(sequence, n, k, **kwargs):
    """
    Returns all possible skipgrams generated from a sequence of items, as an iterator.
    Skipgrams are ngrams that allows tokens to be skipped.
    Refer to http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf

        >>> sent = "Insurgents killed in ongoing fighting".split()
        >>> list(skipgrams(sent, 2, 2))
        [('Insurgents', 'killed'), ('Insurgents', 'in'), ('Insurgents', 'ongoing'), ('killed', 'in'), ('killed', 'ongoing'), ('killed', 'fighting'), ('in', 'ongoing'), ('in', 'fighting'), ('ongoing', 'fighting')]
        >>> list(skipgrams(sent, 3, 2))
        [('Insurgents', 'killed', 'in'), ('Insurgents', 'killed', 'ongoing'), ('Insurgents', 'killed', 'fighting'), ('Insurgents', 'in', 'ongoing'), ('Insurgents', 'in', 'fighting'), ('Insurgents', 'ongoing', 'fighting'), ('killed', 'in', 'ongoing'), ('killed', 'in', 'fighting'), ('killed', 'ongoing', 'fighting'), ('in', 'ongoing', 'fighting')]

    :param sequence: the source data to be converted into trigrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param k: the skip distance
    :type  k: int
    :rtype: iter(tuple)
    """

    # Pads the sequence as desired by **kwargs.
    if "pad_left" in kwargs or "pad_right" in kwargs:
        sequence = pad_sequence(sequence, n, **kwargs)

    # Note when iterating through the ngrams, the pad_right here is not
    # the **kwargs padding, it's for the algorithm to detect the SENTINEL
    # object on the right pad to stop inner loop.
    SENTINEL = object()
    for ngram in ngrams(sequence, n + k, pad_right=True, right_pad_symbol=SENTINEL):
        head = ngram[:1]
        tail = ngram[1:]
        for skip_tail in combinations(tail, n - 1):
            if skip_tail[-1] is SENTINEL:
                continue
            yield head + skip_tail


######################################################################
# Binary Search in a File
######################################################################

# inherited from pywordnet, by Oliver Steele
def binary_search_file(file, key, cache=None, cacheDepth=-1):
    """
    Return the line from the file with first word key.
    Searches through a sorted file using the binary search algorithm.

    :type file: file
    :param file: the file to be searched through.
    :type key: str
    :param key: the identifier we are searching for.
    """

    key = key + " "
    keylen = len(key)
    start = 0
    currentDepth = 0

    if hasattr(file, "name"):
        end = os.stat(file.name).st_size - 1
    else:
        file.seek(0, 2)
        end = file.tell() - 1
        file.seek(0)

    if cache is None:
        cache = {}

    while start < end:
        lastState = start, end
        middle = (start + end) // 2

        if cache.get(middle):
            offset, line = cache[middle]

        else:
            line = ""
            while True:
                file.seek(max(0, middle - 1))
                if middle > 0:
                    file.discard_line()
                offset = file.tell()
                line = file.readline()
                if line != "":
                    break
                # at EOF; try to find start of the last line
                middle = (start + middle) // 2
                if middle == end - 1:
                    return None
            if currentDepth < cacheDepth:
                cache[middle] = (offset, line)

        if offset > end:
            assert end != middle - 1, "infinite loop"
            end = middle - 1
        elif line[:keylen] == key:
            return line
        elif line > key:
            assert end != middle - 1, "infinite loop"
            end = middle - 1
        elif line < key:
            start = offset + len(line) - 1

        currentDepth += 1
        thisState = start, end

        if lastState == thisState:
            # Detects the condition where we're searching past the end
            # of the file, which is otherwise difficult to detect
            return None

    return None


######################################################################
# Proxy configuration
######################################################################


def set_proxy(proxy, user=None, password=""):
    """
    Set the HTTP proxy for Python to download through.

    If ``proxy`` is None then tries to set proxy from environment or system
    settings.

    :param proxy: The HTTP proxy server to use. For example:
        'http://proxy.example.com:3128/'
    :param user: The username to authenticate with. Use None to disable
        authentication.
    :param password: The password to authenticate with.
    """
    if proxy is None:
        # Try and find the system proxy settings
        try:
            proxy = getproxies()["http"]
        except KeyError as e:
            raise ValueError("Could not detect default proxy settings") from e

    # Set up the proxy handler
    proxy_handler = ProxyHandler({"https": proxy, "http": proxy})
    opener = build_opener(proxy_handler)

    if user is not None:
        # Set up basic proxy authentication if provided
        password_manager = HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(realm=None, uri=proxy, user=user, passwd=password)
        opener.add_handler(ProxyBasicAuthHandler(password_manager))
        opener.add_handler(ProxyDigestAuthHandler(password_manager))

    # Override the existing url opener
    install_opener(opener)


######################################################################
# ElementTree pretty printing from https://www.effbot.org/zone/element-lib.htm
######################################################################


def elementtree_indent(elem, level=0):
    """
    Recursive function to indent an ElementTree._ElementInterface
    used for pretty printing. Run indent on elem and then output
    in the normal way.

    :param elem: element to be indented. will be modified.
    :type elem: ElementTree._ElementInterface
    :param level: level of indentation for this element
    :type level: nonnegative integer
    :rtype:   ElementTree._ElementInterface
    :return:  Contents of elem indented to reflect its structure
    """

    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for elem in elem:
            elementtree_indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


######################################################################
# Mathematical approximations
######################################################################


def choose(n, k):
    """
    This function is a fast way to calculate binomial coefficients, commonly
    known as nCk, i.e. the number of combinations of n things taken k at a time.
    (https://en.wikipedia.org/wiki/Binomial_coefficient).

    This is the *scipy.special.comb()* with long integer computation but this
    approximation is faster, see https://github.com/nltk/nltk/issues/1181

        >>> choose(4, 2)
        6
        >>> choose(6, 2)
        15

    :param n: The number of things.
    :type n: int
    :param r: The number of times a thing is taken.
    :type r: int
    """
    if 0 <= k <= n:
        ntok, ktok = 1, 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


######################################################################
# Iteration utilities
######################################################################


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


######################################################################
# Parallelization.
######################################################################


def parallelize_preprocess(func, iterator, processes, progress_bar=False):
    from joblib import Parallel, delayed
    from tqdm import tqdm

    iterator = tqdm(iterator) if progress_bar else iterator
    if processes <= 1:
        return map(func, iterator)
    return Parallel(n_jobs=processes)(delayed(func)(line) for line in iterator)
