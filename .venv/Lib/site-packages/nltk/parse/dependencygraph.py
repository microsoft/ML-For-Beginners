# Natural Language Toolkit: Dependency Grammars
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Jason Narad <jason.narad@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (modifications)
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Tools for reading and writing dependency trees.
The input is assumed to be in Malt-TAB format
(https://stp.lingfil.uu.se/~nivre/research/MaltXML.html).
"""

import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat

from nltk.internals import find_binary
from nltk.tree import Tree

#################################################################
# DependencyGraph Class
#################################################################


class DependencyGraph:
    """
    A container for the nodes and labelled edges of a dependency structure.
    """

    def __init__(
        self,
        tree_str=None,
        cell_extractor=None,
        zero_based=False,
        cell_separator=None,
        top_relation_label="ROOT",
    ):
        """Dependency graph.

        We place a dummy `TOP` node with the index 0, since the root node is
        often assigned 0 as its head. This also means that the indexing of the
        nodes corresponds directly to the Malt-TAB format, which starts at 1.

        If zero-based is True, then Malt-TAB-like input with node numbers
        starting at 0 and the root node assigned -1 (as produced by, e.g.,
        zpar).

        :param str cell_separator: the cell separator. If not provided, cells
            are split by whitespace.

        :param str top_relation_label: the label by which the top relation is
            identified, for examlple, `ROOT`, `null` or `TOP`.
        """
        self.nodes = defaultdict(
            lambda: {
                "address": None,
                "word": None,
                "lemma": None,
                "ctag": None,
                "tag": None,
                "feats": None,
                "head": None,
                "deps": defaultdict(list),
                "rel": None,
            }
        )

        self.nodes[0].update({"ctag": "TOP", "tag": "TOP", "address": 0})

        self.root = None

        if tree_str:
            self._parse(
                tree_str,
                cell_extractor=cell_extractor,
                zero_based=zero_based,
                cell_separator=cell_separator,
                top_relation_label=top_relation_label,
            )

    def remove_by_address(self, address):
        """
        Removes the node with the given address.  References
        to this node in others will still exist.
        """
        del self.nodes[address]

    def redirect_arcs(self, originals, redirect):
        """
        Redirects arcs to any of the nodes in the originals list
        to the redirect node address.
        """
        for node in self.nodes.values():
            new_deps = []
            for dep in node["deps"]:
                if dep in originals:
                    new_deps.append(redirect)
                else:
                    new_deps.append(dep)
            node["deps"] = new_deps

    def add_arc(self, head_address, mod_address):
        """
        Adds an arc from the node specified by head_address to the
        node specified by the mod address.
        """
        relation = self.nodes[mod_address]["rel"]
        self.nodes[head_address]["deps"].setdefault(relation, [])
        self.nodes[head_address]["deps"][relation].append(mod_address)
        # self.nodes[head_address]['deps'].append(mod_address)

    def connect_graph(self):
        """
        Fully connects all non-root nodes.  All nodes are set to be dependents
        of the root node.
        """
        for node1 in self.nodes.values():
            for node2 in self.nodes.values():
                if node1["address"] != node2["address"] and node2["rel"] != "TOP":
                    relation = node2["rel"]
                    node1["deps"].setdefault(relation, [])
                    node1["deps"][relation].append(node2["address"])
                    # node1['deps'].append(node2['address'])

    def get_by_address(self, node_address):
        """Return the node with the given address."""
        return self.nodes[node_address]

    def contains_address(self, node_address):
        """
        Returns true if the graph contains a node with the given node
        address, false otherwise.
        """
        return node_address in self.nodes

    def to_dot(self):
        """Return a dot representation suitable for using with Graphviz.

        >>> dg = DependencyGraph(
        ...     'John N 2\\n'
        ...     'loves V 0\\n'
        ...     'Mary N 2'
        ... )
        >>> print(dg.to_dot())
        digraph G{
        edge [dir=forward]
        node [shape=plaintext]
        <BLANKLINE>
        0 [label="0 (None)"]
        0 -> 2 [label="ROOT"]
        1 [label="1 (John)"]
        2 [label="2 (loves)"]
        2 -> 1 [label=""]
        2 -> 3 [label=""]
        3 [label="3 (Mary)"]
        }

        """
        # Start the digraph specification
        s = "digraph G{\n"
        s += "edge [dir=forward]\n"
        s += "node [shape=plaintext]\n"

        # Draw the remaining nodes
        for node in sorted(self.nodes.values(), key=lambda v: v["address"]):
            s += '\n{} [label="{} ({})"]'.format(
                node["address"],
                node["address"],
                node["word"],
            )
            for rel, deps in node["deps"].items():
                for dep in deps:
                    if rel is not None:
                        s += '\n{} -> {} [label="{}"]'.format(node["address"], dep, rel)
                    else:
                        s += "\n{} -> {} ".format(node["address"], dep)
        s += "\n}"

        return s

    def _repr_svg_(self):
        """Show SVG representation of the transducer (IPython magic).
        >>> from nltk.test.setup_fixt import check_binary
        >>> check_binary('dot')
        >>> dg = DependencyGraph(
        ...     'John N 2\\n'
        ...     'loves V 0\\n'
        ...     'Mary N 2'
        ... )
        >>> dg._repr_svg_().split('\\n')[0]
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'

        """
        dot_string = self.to_dot()
        return dot2img(dot_string)

    def __str__(self):
        return pformat(self.nodes)

    def __repr__(self):
        return f"<DependencyGraph with {len(self.nodes)} nodes>"

    @staticmethod
    def load(
        filename, zero_based=False, cell_separator=None, top_relation_label="ROOT"
    ):
        """
        :param filename: a name of a file in Malt-TAB format
        :param zero_based: nodes in the input file are numbered starting from 0
            rather than 1 (as produced by, e.g., zpar)
        :param str cell_separator: the cell separator. If not provided, cells
            are split by whitespace.
        :param str top_relation_label: the label by which the top relation is
            identified, for examlple, `ROOT`, `null` or `TOP`.

        :return: a list of DependencyGraphs

        """
        with open(filename) as infile:
            return [
                DependencyGraph(
                    tree_str,
                    zero_based=zero_based,
                    cell_separator=cell_separator,
                    top_relation_label=top_relation_label,
                )
                for tree_str in infile.read().split("\n\n")
            ]

    def left_children(self, node_index):
        """
        Returns the number of left children under the node specified
        by the given address.
        """
        children = chain.from_iterable(self.nodes[node_index]["deps"].values())
        index = self.nodes[node_index]["address"]
        return sum(1 for c in children if c < index)

    def right_children(self, node_index):
        """
        Returns the number of right children under the node specified
        by the given address.
        """
        children = chain.from_iterable(self.nodes[node_index]["deps"].values())
        index = self.nodes[node_index]["address"]
        return sum(1 for c in children if c > index)

    def add_node(self, node):
        if not self.contains_address(node["address"]):
            self.nodes[node["address"]].update(node)

    def _parse(
        self,
        input_,
        cell_extractor=None,
        zero_based=False,
        cell_separator=None,
        top_relation_label="ROOT",
    ):
        """Parse a sentence.

        :param extractor: a function that given a tuple of cells returns a
        7-tuple, where the values are ``word, lemma, ctag, tag, feats, head,
        rel``.

        :param str cell_separator: the cell separator. If not provided, cells
        are split by whitespace.

        :param str top_relation_label: the label by which the top relation is
        identified, for examlple, `ROOT`, `null` or `TOP`.

        """

        def extract_3_cells(cells, index):
            word, tag, head = cells
            return index, word, word, tag, tag, "", head, ""

        def extract_4_cells(cells, index):
            word, tag, head, rel = cells
            return index, word, word, tag, tag, "", head, rel

        def extract_7_cells(cells, index):
            line_index, word, lemma, tag, _, head, rel = cells
            try:
                index = int(line_index)
            except ValueError:
                # index can't be parsed as an integer, use default
                pass
            return index, word, lemma, tag, tag, "", head, rel

        def extract_10_cells(cells, index):
            line_index, word, lemma, ctag, tag, feats, head, rel, _, _ = cells
            try:
                index = int(line_index)
            except ValueError:
                # index can't be parsed as an integer, use default
                pass
            return index, word, lemma, ctag, tag, feats, head, rel

        extractors = {
            3: extract_3_cells,
            4: extract_4_cells,
            7: extract_7_cells,
            10: extract_10_cells,
        }

        if isinstance(input_, str):
            input_ = (line for line in input_.split("\n"))

        lines = (l.rstrip() for l in input_)
        lines = (l for l in lines if l)

        cell_number = None
        for index, line in enumerate(lines, start=1):
            cells = line.split(cell_separator)
            if cell_number is None:
                cell_number = len(cells)
            else:
                assert cell_number == len(cells)

            if cell_extractor is None:
                try:
                    cell_extractor = extractors[cell_number]
                except KeyError as e:
                    raise ValueError(
                        "Number of tab-delimited fields ({}) not supported by "
                        "CoNLL(10) or Malt-Tab(4) format".format(cell_number)
                    ) from e

            try:
                index, word, lemma, ctag, tag, feats, head, rel = cell_extractor(
                    cells, index
                )
            except (TypeError, ValueError):
                # cell_extractor doesn't take 2 arguments or doesn't return 8
                # values; assume the cell_extractor is an older external
                # extractor and doesn't accept or return an index.
                word, lemma, ctag, tag, feats, head, rel = cell_extractor(cells)

            if head == "_":
                continue

            head = int(head)
            if zero_based:
                head += 1

            self.nodes[index].update(
                {
                    "address": index,
                    "word": word,
                    "lemma": lemma,
                    "ctag": ctag,
                    "tag": tag,
                    "feats": feats,
                    "head": head,
                    "rel": rel,
                }
            )

            # Make sure that the fake root node has labeled dependencies.
            if (cell_number == 3) and (head == 0):
                rel = top_relation_label
            self.nodes[head]["deps"][rel].append(index)

        if self.nodes[0]["deps"][top_relation_label]:
            root_address = self.nodes[0]["deps"][top_relation_label][0]
            self.root = self.nodes[root_address]
            self.top_relation_label = top_relation_label
        else:
            warnings.warn(
                "The graph doesn't contain a node " "that depends on the root element."
            )

    def _word(self, node, filter=True):
        w = node["word"]
        if filter:
            if w != ",":
                return w
        return w

    def _tree(self, i):
        """Turn dependency graphs into NLTK trees.

        :param int i: index of a node
        :return: either a word (if the indexed node is a leaf) or a ``Tree``.
        """
        node = self.get_by_address(i)
        word = node["word"]
        deps = sorted(chain.from_iterable(node["deps"].values()))

        if deps:
            return Tree(word, [self._tree(dep) for dep in deps])
        else:
            return word

    def tree(self):
        """
        Starting with the ``root`` node, build a dependency tree using the NLTK
        ``Tree`` constructor. Dependency labels are omitted.
        """
        node = self.root

        word = node["word"]
        deps = sorted(chain.from_iterable(node["deps"].values()))
        return Tree(word, [self._tree(dep) for dep in deps])

    def triples(self, node=None):
        """
        Extract dependency triples of the form:
        ((head word, head tag), rel, (dep word, dep tag))
        """

        if not node:
            node = self.root

        head = (node["word"], node["ctag"])
        for i in sorted(chain.from_iterable(node["deps"].values())):
            dep = self.get_by_address(i)
            yield (head, dep["rel"], (dep["word"], dep["ctag"]))
            yield from self.triples(node=dep)

    def _hd(self, i):
        try:
            return self.nodes[i]["head"]
        except IndexError:
            return None

    def _rel(self, i):
        try:
            return self.nodes[i]["rel"]
        except IndexError:
            return None

    # what's the return type?  Boolean or list?
    def contains_cycle(self):
        """Check whether there are cycles.

        >>> dg = DependencyGraph(treebank_data)
        >>> dg.contains_cycle()
        False

        >>> cyclic_dg = DependencyGraph()
        >>> top = {'word': None, 'deps': [1], 'rel': 'TOP', 'address': 0}
        >>> child1 = {'word': None, 'deps': [2], 'rel': 'NTOP', 'address': 1}
        >>> child2 = {'word': None, 'deps': [4], 'rel': 'NTOP', 'address': 2}
        >>> child3 = {'word': None, 'deps': [1], 'rel': 'NTOP', 'address': 3}
        >>> child4 = {'word': None, 'deps': [3], 'rel': 'NTOP', 'address': 4}
        >>> cyclic_dg.nodes = {
        ...     0: top,
        ...     1: child1,
        ...     2: child2,
        ...     3: child3,
        ...     4: child4,
        ... }
        >>> cyclic_dg.root = top

        >>> cyclic_dg.contains_cycle()
        [1, 2, 4, 3]

        """
        distances = {}

        for node in self.nodes.values():
            for dep in node["deps"]:
                key = tuple([node["address"], dep])
                distances[key] = 1

        for _ in self.nodes:
            new_entries = {}

            for pair1 in distances:
                for pair2 in distances:
                    if pair1[1] == pair2[0]:
                        key = tuple([pair1[0], pair2[1]])
                        new_entries[key] = distances[pair1] + distances[pair2]

            for pair in new_entries:
                distances[pair] = new_entries[pair]
                if pair[0] == pair[1]:
                    path = self.get_cycle_path(self.get_by_address(pair[0]), pair[0])
                    return path

        return False  # return []?

    def get_cycle_path(self, curr_node, goal_node_index):
        for dep in curr_node["deps"]:
            if dep == goal_node_index:
                return [curr_node["address"]]
        for dep in curr_node["deps"]:
            path = self.get_cycle_path(self.get_by_address(dep), goal_node_index)
            if len(path) > 0:
                path.insert(0, curr_node["address"])
                return path
        return []

    def to_conll(self, style):
        """
        The dependency graph in CoNLL format.

        :param style: the style to use for the format (3, 4, 10 columns)
        :type style: int
        :rtype: str
        """

        if style == 3:
            template = "{word}\t{tag}\t{head}\n"
        elif style == 4:
            template = "{word}\t{tag}\t{head}\t{rel}\n"
        elif style == 10:
            template = (
                "{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t_\n"
            )
        else:
            raise ValueError(
                "Number of tab-delimited fields ({}) not supported by "
                "CoNLL(10) or Malt-Tab(4) format".format(style)
            )

        return "".join(
            template.format(i=i, **node)
            for i, node in sorted(self.nodes.items())
            if node["tag"] != "TOP"
        )

    def nx_graph(self):
        """Convert the data in a ``nodelist`` into a networkx labeled directed graph."""
        import networkx

        nx_nodelist = list(range(1, len(self.nodes)))
        nx_edgelist = [
            (n, self._hd(n), self._rel(n)) for n in nx_nodelist if self._hd(n)
        ]
        self.nx_labels = {}
        for n in nx_nodelist:
            self.nx_labels[n] = self.nodes[n]["word"]

        g = networkx.MultiDiGraph()
        g.add_nodes_from(nx_nodelist)
        g.add_edges_from(nx_edgelist)

        return g


def dot2img(dot_string, t="svg"):
    """
    Create image representation fom dot_string, using the 'dot' program
    from the Graphviz package.

    Use the 't' argument to specify the image file format, for ex. 'jpeg', 'eps',
    'json', 'png' or 'webp' (Running 'dot -T:' lists all available formats).

    Note that the "capture_output" option of subprocess.run() is only available
    with text formats (like svg), but not with binary image formats (like png).
    """

    try:
        find_binary("dot")
        try:
            if t in ["dot", "dot_json", "json", "svg"]:
                proc = subprocess.run(
                    ["dot", "-T%s" % t],
                    capture_output=True,
                    input=dot_string,
                    text=True,
                )
            else:
                proc = subprocess.run(
                    ["dot", "-T%s" % t],
                    input=bytes(dot_string, encoding="utf8"),
                )
            return proc.stdout
        except:
            raise Exception(
                "Cannot create image representation by running dot from string: {}"
                "".format(dot_string)
            )
    except OSError as e:
        raise Exception("Cannot find the dot binary from Graphviz package") from e


class DependencyGraphError(Exception):
    """Dependency graph exception."""


def demo():
    malt_demo()
    conll_demo()
    conll_file_demo()
    cycle_finding_demo()


def malt_demo(nx=False):
    """
    A demonstration of the result of reading a dependency
    version of the first sentence of the Penn Treebank.
    """
    dg = DependencyGraph(
        """Pierre  NNP     2       NMOD
Vinken  NNP     8       SUB
,       ,       2       P
61      CD      5       NMOD
years   NNS     6       AMOD
old     JJ      2       NMOD
,       ,       2       P
will    MD      0       ROOT
join    VB      8       VC
the     DT      11      NMOD
board   NN      9       OBJ
as      IN      9       VMOD
a       DT      15      NMOD
nonexecutive    JJ      15      NMOD
director        NN      12      PMOD
Nov.    NNP     9       VMOD
29      CD      16      NMOD
.       .       9       VMOD
"""
    )
    tree = dg.tree()
    tree.pprint()
    if nx:
        # currently doesn't work
        import networkx
        from matplotlib import pylab

        g = dg.nx_graph()
        g.info()
        pos = networkx.spring_layout(g, dim=1)
        networkx.draw_networkx_nodes(g, pos, node_size=50)
        # networkx.draw_networkx_edges(g, pos, edge_color='k', width=8)
        networkx.draw_networkx_labels(g, pos, dg.nx_labels)
        pylab.xticks([])
        pylab.yticks([])
        pylab.savefig("tree.png")
        pylab.show()


def conll_demo():
    """
    A demonstration of how to read a string representation of
    a CoNLL format dependency tree.
    """
    dg = DependencyGraph(conll_data1)
    tree = dg.tree()
    tree.pprint()
    print(dg)
    print(dg.to_conll(4))


def conll_file_demo():
    print("Mass conll_read demo...")
    graphs = [DependencyGraph(entry) for entry in conll_data2.split("\n\n") if entry]
    for graph in graphs:
        tree = graph.tree()
        print("\n")
        tree.pprint()


def cycle_finding_demo():
    dg = DependencyGraph(treebank_data)
    print(dg.contains_cycle())
    cyclic_dg = DependencyGraph()
    cyclic_dg.add_node({"word": None, "deps": [1], "rel": "TOP", "address": 0})
    cyclic_dg.add_node({"word": None, "deps": [2], "rel": "NTOP", "address": 1})
    cyclic_dg.add_node({"word": None, "deps": [4], "rel": "NTOP", "address": 2})
    cyclic_dg.add_node({"word": None, "deps": [1], "rel": "NTOP", "address": 3})
    cyclic_dg.add_node({"word": None, "deps": [3], "rel": "NTOP", "address": 4})
    print(cyclic_dg.contains_cycle())


treebank_data = """Pierre  NNP     2       NMOD
Vinken  NNP     8       SUB
,       ,       2       P
61      CD      5       NMOD
years   NNS     6       AMOD
old     JJ      2       NMOD
,       ,       2       P
will    MD      0       ROOT
join    VB      8       VC
the     DT      11      NMOD
board   NN      9       OBJ
as      IN      9       VMOD
a       DT      15      NMOD
nonexecutive    JJ      15      NMOD
director        NN      12      PMOD
Nov.    NNP     9       VMOD
29      CD      16      NMOD
.       .       9       VMOD
"""

conll_data1 = """
1   Ze                ze                Pron  Pron  per|3|evofmv|nom                 2   su      _  _
2   had               heb               V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _
3   met               met               Prep  Prep  voor                             8   mod     _  _
4   haar              haar              Pron  Pron  bez|3|ev|neut|attr               5   det     _  _
5   moeder            moeder            N     N     soort|ev|neut                    3   obj1    _  _
6   kunnen            kan               V     V     hulp|ott|1of2of3|mv              2   vc      _  _
7   gaan              ga                V     V     hulp|inf                         6   vc      _  _
8   winkelen          winkel            V     V     intrans|inf                      11  cnj     _  _
9   ,                 ,                 Punc  Punc  komma                            8   punct   _  _
10  zwemmen           zwem              V     V     intrans|inf                      11  cnj     _  _
11  of                of                Conj  Conj  neven                            7   vc      _  _
12  terrassen         terras            N     N     soort|mv|neut                    11  cnj     _  _
13  .                 .                 Punc  Punc  punt                             12  punct   _  _
"""

conll_data2 = """1   Cathy             Cathy             N     N     eigen|ev|neut                    2   su      _  _
2   zag               zie               V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _
3   hen               hen               Pron  Pron  per|3|mv|datofacc                2   obj1    _  _
4   wild              wild              Adj   Adj   attr|stell|onverv                5   mod     _  _
5   zwaaien           zwaai             N     N     soort|mv|neut                    2   vc      _  _
6   .                 .                 Punc  Punc  punt                             5   punct   _  _

1   Ze                ze                Pron  Pron  per|3|evofmv|nom                 2   su      _  _
2   had               heb               V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _
3   met               met               Prep  Prep  voor                             8   mod     _  _
4   haar              haar              Pron  Pron  bez|3|ev|neut|attr               5   det     _  _
5   moeder            moeder            N     N     soort|ev|neut                    3   obj1    _  _
6   kunnen            kan               V     V     hulp|ott|1of2of3|mv              2   vc      _  _
7   gaan              ga                V     V     hulp|inf                         6   vc      _  _
8   winkelen          winkel            V     V     intrans|inf                      11  cnj     _  _
9   ,                 ,                 Punc  Punc  komma                            8   punct   _  _
10  zwemmen           zwem              V     V     intrans|inf                      11  cnj     _  _
11  of                of                Conj  Conj  neven                            7   vc      _  _
12  terrassen         terras            N     N     soort|mv|neut                    11  cnj     _  _
13  .                 .                 Punc  Punc  punt                             12  punct   _  _

1   Dat               dat               Pron  Pron  aanw|neut|attr                   2   det     _  _
2   werkwoord         werkwoord         N     N     soort|ev|neut                    6   obj1    _  _
3   had               heb               V     V     hulp|ovt|1of2of3|ev              0   ROOT    _  _
4   ze                ze                Pron  Pron  per|3|evofmv|nom                 6   su      _  _
5   zelf              zelf              Pron  Pron  aanw|neut|attr|wzelf             3   predm   _  _
6   uitgevonden       vind              V     V     trans|verldw|onverv              3   vc      _  _
7   .                 .                 Punc  Punc  punt                             6   punct   _  _

1   Het               het               Pron  Pron  onbep|neut|zelfst                2   su      _  _
2   hoorde            hoor              V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _
3   bij               bij               Prep  Prep  voor                             2   ld      _  _
4   de                de                Art   Art   bep|zijdofmv|neut                6   det     _  _
5   warme             warm              Adj   Adj   attr|stell|vervneut              6   mod     _  _
6   zomerdag          zomerdag          N     N     soort|ev|neut                    3   obj1    _  _
7   die               die               Pron  Pron  betr|neut|zelfst                 6   mod     _  _
8   ze                ze                Pron  Pron  per|3|evofmv|nom                 12  su      _  _
9   ginds             ginds             Adv   Adv   gew|aanw                         12  mod     _  _
10  achter            achter            Adv   Adv   gew|geenfunc|stell|onverv        12  svp     _  _
11  had               heb               V     V     hulp|ovt|1of2of3|ev              7   body    _  _
12  gelaten           laat              V     V     trans|verldw|onverv              11  vc      _  _
13  .                 .                 Punc  Punc  punt                             12  punct   _  _

1   Ze                ze                Pron  Pron  per|3|evofmv|nom                 2   su      _  _
2   hadden            heb               V     V     trans|ovt|1of2of3|mv             0   ROOT    _  _
3   languit           languit           Adv   Adv   gew|geenfunc|stell|onverv        11  mod     _  _
4   naast             naast             Prep  Prep  voor                             11  mod     _  _
5   elkaar            elkaar            Pron  Pron  rec|neut                         4   obj1    _  _
6   op                op                Prep  Prep  voor                             11  ld      _  _
7   de                de                Art   Art   bep|zijdofmv|neut                8   det     _  _
8   strandstoelen     strandstoel       N     N     soort|mv|neut                    6   obj1    _  _
9   kunnen            kan               V     V     hulp|inf                         2   vc      _  _
10  gaan              ga                V     V     hulp|inf                         9   vc      _  _
11  liggen            lig               V     V     intrans|inf                      10  vc      _  _
12  .                 .                 Punc  Punc  punt                             11  punct   _  _

1   Zij               zij               Pron  Pron  per|3|evofmv|nom                 2   su      _  _
2   zou               zal               V     V     hulp|ovt|1of2of3|ev              7   cnj     _  _
3   mams              mams              N     N     soort|ev|neut                    4   det     _  _
4   rug               rug               N     N     soort|ev|neut                    5   obj1    _  _
5   ingewreven        wrijf             V     V     trans|verldw|onverv              6   vc      _  _
6   hebben            heb               V     V     hulp|inf                         2   vc      _  _
7   en                en                Conj  Conj  neven                            0   ROOT    _  _
8   mam               mam               V     V     trans|ovt|1of2of3|ev             7   cnj     _  _
9   de                de                Art   Art   bep|zijdofmv|neut                10  det     _  _
10  hare              hare              Pron  Pron  bez|3|ev|neut|attr               8   obj1    _  _
11  .                 .                 Punc  Punc  punt                             10  punct   _  _

1   Of                of                Conj  Conj  onder|metfin                     0   ROOT    _  _
2   ze                ze                Pron  Pron  per|3|evofmv|nom                 3   su      _  _
3   had               heb               V     V     hulp|ovt|1of2of3|ev              0   ROOT    _  _
4   gewoon            gewoon            Adj   Adj   adv|stell|onverv                 10  mod     _  _
5   met               met               Prep  Prep  voor                             10  mod     _  _
6   haar              haar              Pron  Pron  bez|3|ev|neut|attr               7   det     _  _
7   vriendinnen       vriendin          N     N     soort|mv|neut                    5   obj1    _  _
8   rond              rond              Adv   Adv   deelv                            10  svp     _  _
9   kunnen            kan               V     V     hulp|inf                         3   vc      _  _
10  slenteren         slenter           V     V     intrans|inf                      9   vc      _  _
11  in                in                Prep  Prep  voor                             10  mod     _  _
12  de                de                Art   Art   bep|zijdofmv|neut                13  det     _  _
13  buurt             buurt             N     N     soort|ev|neut                    11  obj1    _  _
14  van               van               Prep  Prep  voor                             13  mod     _  _
15  Trafalgar_Square  Trafalgar_Square  MWU   N_N   eigen|ev|neut_eigen|ev|neut      14  obj1    _  _
16  .                 .                 Punc  Punc  punt                             15  punct   _  _
"""

if __name__ == "__main__":
    demo()
