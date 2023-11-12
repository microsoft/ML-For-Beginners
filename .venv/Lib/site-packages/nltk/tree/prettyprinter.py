# Natural Language Toolkit: ASCII visualization of NLTK trees
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Andreas van Cranenburgh <A.W.vanCranenburgh@uva.nl>
#         Peter Ljunglöf <peter.ljunglof@gu.se>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Pretty-printing of discontinuous trees.
Adapted from the disco-dop project, by Andreas van Cranenburgh.
https://github.com/andreasvc/disco-dop

Interesting reference (not used for this code):
T. Eschbach et al., Orth. Hypergraph Drawing, Journal of
Graph Algorithms and Applications, 10(2) 141--157 (2006)149.
https://jgaa.info/accepted/2006/EschbachGuentherBecker2006.10.2.pdf
"""

import re

try:
    from html import escape
except ImportError:
    from cgi import escape

from collections import defaultdict
from operator import itemgetter

from nltk.tree.tree import Tree
from nltk.util import OrderedDict

ANSICOLOR = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}


class TreePrettyPrinter:
    """
    Pretty-print a tree in text format, either as ASCII or Unicode.
    The tree can be a normal tree, or discontinuous.

    ``TreePrettyPrinter(tree, sentence=None, highlight=())``
    creates an object from which different visualizations can be created.

    :param tree: a Tree object.
    :param sentence: a list of words (strings). If `sentence` is given,
        `tree` must contain integers as leaves, which are taken as indices
        in `sentence`. Using this you can display a discontinuous tree.
    :param highlight: Optionally, a sequence of Tree objects in `tree` which
        should be highlighted. Has the effect of only applying colors to nodes
        in this sequence (nodes should be given as Tree objects, terminals as
        indices).

    >>> from nltk.tree import Tree
    >>> tree = Tree.fromstring('(S (NP Mary) (VP walks))')
    >>> print(TreePrettyPrinter(tree).text())
    ... # doctest: +NORMALIZE_WHITESPACE
          S
      ____|____
     NP        VP
     |         |
    Mary     walks
    """

    def __init__(self, tree, sentence=None, highlight=()):
        if sentence is None:
            leaves = tree.leaves()
            if (
                leaves
                and all(len(a) > 0 for a in tree.subtrees())
                and all(isinstance(a, int) for a in leaves)
            ):
                sentence = [str(a) for a in leaves]
            else:
                # this deals with empty nodes (frontier non-terminals)
                # and multiple/mixed terminals under non-terminals.
                tree = tree.copy(True)
                sentence = []
                for a in tree.subtrees():
                    if len(a) == 0:
                        a.append(len(sentence))
                        sentence.append(None)
                    elif any(not isinstance(b, Tree) for b in a):
                        for n, b in enumerate(a):
                            if not isinstance(b, Tree):
                                a[n] = len(sentence)
                                if type(b) == tuple:
                                    b = "/".join(b)
                                sentence.append("%s" % b)
        self.nodes, self.coords, self.edges, self.highlight = self.nodecoords(
            tree, sentence, highlight
        )

    def __str__(self):
        return self.text()

    def __repr__(self):
        return "<TreePrettyPrinter with %d nodes>" % len(self.nodes)

    @staticmethod
    def nodecoords(tree, sentence, highlight):
        """
        Produce coordinates of nodes on a grid.

        Objective:

        - Produce coordinates for a non-overlapping placement of nodes and
            horizontal lines.
        - Order edges so that crossing edges cross a minimal number of previous
            horizontal lines (never vertical lines).

        Approach:

        - bottom up level order traversal (start at terminals)
        - at each level, identify nodes which cannot be on the same row
        - identify nodes which cannot be in the same column
        - place nodes into a grid at (row, column)
        - order child-parent edges with crossing edges last

        Coordinates are (row, column); the origin (0, 0) is at the top left;
        the root node is on row 0. Coordinates do not consider the size of a
        node (which depends on font, &c), so the width of a column of the grid
        should be automatically determined by the element with the greatest
        width in that column. Alternatively, the integer coordinates could be
        converted to coordinates in which the distances between adjacent nodes
        are non-uniform.

        Produces tuple (nodes, coords, edges, highlighted) where:

        - nodes[id]: Tree object for the node with this integer id
        - coords[id]: (n, m) coordinate where to draw node with id in the grid
        - edges[id]: parent id of node with this id (ordered dictionary)
        - highlighted: set of ids that should be highlighted
        """

        def findcell(m, matrix, startoflevel, children):
            """
            Find vacant row, column index for node ``m``.
            Iterate over current rows for this level (try lowest first)
            and look for cell between first and last child of this node,
            add new row to level if no free row available.
            """
            candidates = [a for _, a in children[m]]
            minidx, maxidx = min(candidates), max(candidates)
            leaves = tree[m].leaves()
            center = scale * sum(leaves) // len(leaves)  # center of gravity
            if minidx < maxidx and not minidx < center < maxidx:
                center = sum(candidates) // len(candidates)
            if max(candidates) - min(candidates) > 2 * scale:
                center -= center % scale  # round to unscaled coordinate
                if minidx < maxidx and not minidx < center < maxidx:
                    center += scale
            if ids[m] == 0:
                startoflevel = len(matrix)
            for rowidx in range(startoflevel, len(matrix) + 1):
                if rowidx == len(matrix):  # need to add a new row
                    matrix.append(
                        [
                            vertline if a not in (corner, None) else None
                            for a in matrix[-1]
                        ]
                    )
                row = matrix[rowidx]
                if len(children[m]) == 1:  # place unaries directly above child
                    return rowidx, next(iter(children[m]))[1]
                elif all(
                    a is None or a == vertline
                    for a in row[min(candidates) : max(candidates) + 1]
                ):
                    # find free column
                    for n in range(scale):
                        i = j = center + n
                        while j > minidx or i < maxidx:
                            if i < maxidx and (
                                matrix[rowidx][i] is None or i in candidates
                            ):
                                return rowidx, i
                            elif j > minidx and (
                                matrix[rowidx][j] is None or j in candidates
                            ):
                                return rowidx, j
                            i += scale
                            j -= scale
            raise ValueError(
                "could not find a free cell for:\n%s\n%s"
                "min=%d; max=%d" % (tree[m], minidx, maxidx, dumpmatrix())
            )

        def dumpmatrix():
            """Dump matrix contents for debugging purposes."""
            return "\n".join(
                "%2d: %s" % (n, " ".join(("%2r" % i)[:2] for i in row))
                for n, row in enumerate(matrix)
            )

        leaves = tree.leaves()
        if not all(isinstance(n, int) for n in leaves):
            raise ValueError("All leaves must be integer indices.")
        if len(leaves) != len(set(leaves)):
            raise ValueError("Indices must occur at most once.")
        if not all(0 <= n < len(sentence) for n in leaves):
            raise ValueError(
                "All leaves must be in the interval 0..n "
                "with n=len(sentence)\ntokens: %d indices: "
                "%r\nsentence: %s" % (len(sentence), tree.leaves(), sentence)
            )
        vertline, corner = -1, -2  # constants
        tree = tree.copy(True)
        for a in tree.subtrees():
            a.sort(key=lambda n: min(n.leaves()) if isinstance(n, Tree) else n)
        scale = 2
        crossed = set()
        # internal nodes and lexical nodes (no frontiers)
        positions = tree.treepositions()
        maxdepth = max(map(len, positions)) + 1
        childcols = defaultdict(set)
        matrix = [[None] * (len(sentence) * scale)]
        nodes = {}
        ids = {a: n for n, a in enumerate(positions)}
        highlighted_nodes = {
            n for a, n in ids.items() if not highlight or tree[a] in highlight
        }
        levels = {n: [] for n in range(maxdepth - 1)}
        terminals = []
        for a in positions:
            node = tree[a]
            if isinstance(node, Tree):
                levels[maxdepth - node.height()].append(a)
            else:
                terminals.append(a)

        for n in levels:
            levels[n].sort(key=lambda n: max(tree[n].leaves()) - min(tree[n].leaves()))
        terminals.sort()
        positions = set(positions)

        for m in terminals:
            i = int(tree[m]) * scale
            assert matrix[0][i] is None, (matrix[0][i], m, i)
            matrix[0][i] = ids[m]
            nodes[ids[m]] = sentence[tree[m]]
            if nodes[ids[m]] is None:
                nodes[ids[m]] = "..."
                highlighted_nodes.discard(ids[m])
            positions.remove(m)
            childcols[m[:-1]].add((0, i))

        # add other nodes centered on their children,
        # if the center is already taken, back off
        # to the left and right alternately, until an empty cell is found.
        for n in sorted(levels, reverse=True):
            nodesatdepth = levels[n]
            startoflevel = len(matrix)
            matrix.append(
                [vertline if a not in (corner, None) else None for a in matrix[-1]]
            )
            for m in nodesatdepth:  # [::-1]:
                if n < maxdepth - 1 and childcols[m]:
                    _, pivot = min(childcols[m], key=itemgetter(1))
                    if {
                        a[:-1]
                        for row in matrix[:-1]
                        for a in row[:pivot]
                        if isinstance(a, tuple)
                    } & {
                        a[:-1]
                        for row in matrix[:-1]
                        for a in row[pivot:]
                        if isinstance(a, tuple)
                    }:
                        crossed.add(m)

                rowidx, i = findcell(m, matrix, startoflevel, childcols)
                positions.remove(m)

                # block positions where children of this node branch out
                for _, x in childcols[m]:
                    matrix[rowidx][x] = corner
                # assert m == () or matrix[rowidx][i] in (None, corner), (
                #         matrix[rowidx][i], m, str(tree), ' '.join(sentence))
                # node itself
                matrix[rowidx][i] = ids[m]
                nodes[ids[m]] = tree[m]
                # add column to the set of children for its parent
                if len(m) > 0:
                    childcols[m[:-1]].add((rowidx, i))
        assert len(positions) == 0

        # remove unused columns, right to left
        for m in range(scale * len(sentence) - 1, -1, -1):
            if not any(isinstance(row[m], (Tree, int)) for row in matrix):
                for row in matrix:
                    del row[m]

        # remove unused rows, reverse
        matrix = [
            row
            for row in reversed(matrix)
            if not all(a is None or a == vertline for a in row)
        ]

        # collect coordinates of nodes
        coords = {}
        for n, _ in enumerate(matrix):
            for m, i in enumerate(matrix[n]):
                if isinstance(i, int) and i >= 0:
                    coords[i] = n, m

        # move crossed edges last
        positions = sorted(
            (a for level in levels.values() for a in level),
            key=lambda a: a[:-1] in crossed,
        )

        # collect edges from node to node
        edges = OrderedDict()
        for i in reversed(positions):
            for j, _ in enumerate(tree[i]):
                edges[ids[i + (j,)]] = ids[i]

        return nodes, coords, edges, highlighted_nodes

    def text(
        self,
        nodedist=1,
        unicodelines=False,
        html=False,
        ansi=False,
        nodecolor="blue",
        leafcolor="red",
        funccolor="green",
        abbreviate=None,
        maxwidth=16,
    ):
        """
        :return: ASCII art for a discontinuous tree.

        :param unicodelines: whether to use Unicode line drawing characters
            instead of plain (7-bit) ASCII.
        :param html: whether to wrap output in html code (default plain text).
        :param ansi: whether to produce colors with ANSI escape sequences
            (only effective when html==False).
        :param leafcolor, nodecolor: specify colors of leaves and phrasal
            nodes; effective when either html or ansi is True.
        :param abbreviate: if True, abbreviate labels longer than 5 characters.
            If integer, abbreviate labels longer than `abbr` characters.
        :param maxwidth: maximum number of characters before a label starts to
            wrap; pass None to disable.
        """
        if abbreviate == True:
            abbreviate = 5
        if unicodelines:
            horzline = "\u2500"
            leftcorner = "\u250c"
            rightcorner = "\u2510"
            vertline = " \u2502 "
            tee = horzline + "\u252C" + horzline
            bottom = horzline + "\u2534" + horzline
            cross = horzline + "\u253c" + horzline
            ellipsis = "\u2026"
        else:
            horzline = "_"
            leftcorner = rightcorner = " "
            vertline = " | "
            tee = 3 * horzline
            cross = bottom = "_|_"
            ellipsis = "."

        def crosscell(cur, x=vertline):
            """Overwrite center of this cell with a vertical branch."""
            splitl = len(cur) - len(cur) // 2 - len(x) // 2 - 1
            lst = list(cur)
            lst[splitl : splitl + len(x)] = list(x)
            return "".join(lst)

        result = []
        matrix = defaultdict(dict)
        maxnodewith = defaultdict(lambda: 3)
        maxnodeheight = defaultdict(lambda: 1)
        maxcol = 0
        minchildcol = {}
        maxchildcol = {}
        childcols = defaultdict(set)
        labels = {}
        wrapre = re.compile(
            "(.{%d,%d}\\b\\W*|.{%d})" % (maxwidth - 4, maxwidth, maxwidth)
        )
        # collect labels and coordinates
        for a in self.nodes:
            row, column = self.coords[a]
            matrix[row][column] = a
            maxcol = max(maxcol, column)
            label = (
                self.nodes[a].label()
                if isinstance(self.nodes[a], Tree)
                else self.nodes[a]
            )
            if abbreviate and len(label) > abbreviate:
                label = label[:abbreviate] + ellipsis
            if maxwidth and len(label) > maxwidth:
                label = wrapre.sub(r"\1\n", label).strip()
            label = label.split("\n")
            maxnodeheight[row] = max(maxnodeheight[row], len(label))
            maxnodewith[column] = max(maxnodewith[column], max(map(len, label)))
            labels[a] = label
            if a not in self.edges:
                continue  # e.g., root
            parent = self.edges[a]
            childcols[parent].add((row, column))
            minchildcol[parent] = min(minchildcol.get(parent, column), column)
            maxchildcol[parent] = max(maxchildcol.get(parent, column), column)
        # bottom up level order traversal
        for row in sorted(matrix, reverse=True):
            noderows = [
                ["".center(maxnodewith[col]) for col in range(maxcol + 1)]
                for _ in range(maxnodeheight[row])
            ]
            branchrow = ["".center(maxnodewith[col]) for col in range(maxcol + 1)]
            for col in matrix[row]:
                n = matrix[row][col]
                node = self.nodes[n]
                text = labels[n]
                if isinstance(node, Tree):
                    # draw horizontal branch towards children for this node
                    if n in minchildcol and minchildcol[n] < maxchildcol[n]:
                        i, j = minchildcol[n], maxchildcol[n]
                        a, b = (maxnodewith[i] + 1) // 2 - 1, maxnodewith[j] // 2
                        branchrow[i] = ((" " * a) + leftcorner).ljust(
                            maxnodewith[i], horzline
                        )
                        branchrow[j] = (rightcorner + (" " * b)).rjust(
                            maxnodewith[j], horzline
                        )
                        for i in range(minchildcol[n] + 1, maxchildcol[n]):
                            if i == col and any(a == i for _, a in childcols[n]):
                                line = cross
                            elif i == col:
                                line = bottom
                            elif any(a == i for _, a in childcols[n]):
                                line = tee
                            else:
                                line = horzline
                            branchrow[i] = line.center(maxnodewith[i], horzline)
                    else:  # if n and n in minchildcol:
                        branchrow[col] = crosscell(branchrow[col])
                text = [a.center(maxnodewith[col]) for a in text]
                color = nodecolor if isinstance(node, Tree) else leafcolor
                if isinstance(node, Tree) and node.label().startswith("-"):
                    color = funccolor
                if html:
                    text = [escape(a, quote=False) for a in text]
                    if n in self.highlight:
                        text = [f"<font color={color}>{a}</font>" for a in text]
                elif ansi and n in self.highlight:
                    text = ["\x1b[%d;1m%s\x1b[0m" % (ANSICOLOR[color], a) for a in text]
                for x in range(maxnodeheight[row]):
                    # draw vertical lines in partially filled multiline node
                    # labels, but only if it's not a frontier node.
                    noderows[x][col] = (
                        text[x]
                        if x < len(text)
                        else (vertline if childcols[n] else " ").center(
                            maxnodewith[col], " "
                        )
                    )
            # for each column, if there is a node below us which has a parent
            # above us, draw a vertical branch in that column.
            if row != max(matrix):
                for n, (childrow, col) in self.coords.items():
                    if n > 0 and self.coords[self.edges[n]][0] < row < childrow:
                        branchrow[col] = crosscell(branchrow[col])
                        if col not in matrix[row]:
                            for noderow in noderows:
                                noderow[col] = crosscell(noderow[col])
                branchrow = [
                    a + ((a[-1] if a[-1] != " " else b[0]) * nodedist)
                    for a, b in zip(branchrow, branchrow[1:] + [" "])
                ]
                result.append("".join(branchrow))
            result.extend(
                (" " * nodedist).join(noderow) for noderow in reversed(noderows)
            )
        return "\n".join(reversed(result)) + "\n"

    def svg(self, nodecolor="blue", leafcolor="red", funccolor="green"):
        """
        :return: SVG representation of a tree.
        """
        fontsize = 12
        hscale = 40
        vscale = 25
        hstart = vstart = 20
        width = max(col for _, col in self.coords.values())
        height = max(row for row, _ in self.coords.values())
        result = [
            '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
            'width="%dem" height="%dem" viewBox="%d %d %d %d">'
            % (
                width * 3,
                height * 2.5,
                -hstart,
                -vstart,
                width * hscale + 3 * hstart,
                height * vscale + 3 * vstart,
            )
        ]

        children = defaultdict(set)
        for n in self.nodes:
            if n:
                children[self.edges[n]].add(n)

        # horizontal branches from nodes to children
        for node in self.nodes:
            if not children[node]:
                continue
            y, x = self.coords[node]
            x *= hscale
            y *= vscale
            x += hstart
            y += vstart + fontsize // 2
            childx = [self.coords[c][1] for c in children[node]]
            xmin = hstart + hscale * min(childx)
            xmax = hstart + hscale * max(childx)
            result.append(
                '\t<polyline style="stroke:black; stroke-width:1; fill:none;" '
                'points="%g,%g %g,%g" />' % (xmin, y, xmax, y)
            )
            result.append(
                '\t<polyline style="stroke:black; stroke-width:1; fill:none;" '
                'points="%g,%g %g,%g" />' % (x, y, x, y - fontsize // 3)
            )

        # vertical branches from children to parents
        for child, parent in self.edges.items():
            y, _ = self.coords[parent]
            y *= vscale
            y += vstart + fontsize // 2
            childy, childx = self.coords[child]
            childx *= hscale
            childy *= vscale
            childx += hstart
            childy += vstart - fontsize
            result += [
                '\t<polyline style="stroke:white; stroke-width:10; fill:none;"'
                ' points="%g,%g %g,%g" />' % (childx, childy, childx, y + 5),
                '\t<polyline style="stroke:black; stroke-width:1; fill:none;"'
                ' points="%g,%g %g,%g" />' % (childx, childy, childx, y),
            ]

        # write nodes with coordinates
        for n, (row, column) in self.coords.items():
            node = self.nodes[n]
            x = column * hscale + hstart
            y = row * vscale + vstart
            if n in self.highlight:
                color = nodecolor if isinstance(node, Tree) else leafcolor
                if isinstance(node, Tree) and node.label().startswith("-"):
                    color = funccolor
            else:
                color = "black"
            result += [
                '\t<text style="text-anchor: middle; fill: %s; '
                'font-size: %dpx;" x="%g" y="%g">%s</text>'
                % (
                    color,
                    fontsize,
                    x,
                    y,
                    escape(
                        node.label() if isinstance(node, Tree) else node, quote=False
                    ),
                )
            ]

        result += ["</svg>"]
        return "\n".join(result)


def test():
    """Do some tree drawing tests."""

    def print_tree(n, tree, sentence=None, ansi=True, **xargs):
        print()
        print('{}: "{}"'.format(n, " ".join(sentence or tree.leaves())))
        print(tree)
        print()
        drawtree = TreePrettyPrinter(tree, sentence)
        try:
            print(drawtree.text(unicodelines=ansi, ansi=ansi, **xargs))
        except (UnicodeDecodeError, UnicodeEncodeError):
            print(drawtree.text(unicodelines=False, ansi=False, **xargs))

    from nltk.corpus import treebank

    for n in [0, 1440, 1591, 2771, 2170]:
        tree = treebank.parsed_sents()[n]
        print_tree(n, tree, nodedist=2, maxwidth=8)
    print()
    print("ASCII version:")
    print(TreePrettyPrinter(tree).text(nodedist=2))

    tree = Tree.fromstring(
        "(top (punct 8) (smain (noun 0) (verb 1) (inf (verb 5) (inf (verb 6) "
        "(conj (inf (pp (prep 2) (np (det 3) (noun 4))) (verb 7)) (inf (verb 9)) "
        "(vg 10) (inf (verb 11)))))) (punct 12))",
        read_leaf=int,
    )
    sentence = (
        "Ze had met haar moeder kunnen gaan winkelen ,"
        " zwemmen of terrassen .".split()
    )
    print_tree("Discontinuous tree", tree, sentence, nodedist=2)


__all__ = ["TreePrettyPrinter"]

if __name__ == "__main__":
    test()
