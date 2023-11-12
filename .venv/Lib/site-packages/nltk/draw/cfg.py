# Natural Language Toolkit: CFG visualization
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Visualization tools for CFGs.
"""

# Idea for a nice demo:
#   - 3 panes: grammar, treelet, working area
#     - grammar is a list of productions
#     - when you select a production, the treelet that it licenses appears
#       in the treelet area
#     - the working area has the text on the bottom, and S at top.  When
#       you select a production, it shows (ghosted) the locations where
#       that production's treelet could be attached to either the text
#       or the tree rooted at S.
#     - the user can drag the treelet onto one of those (or click on them?)
#     - the user can delete pieces of the tree from the working area
#       (right click?)
#     - connecting top to bottom? drag one NP onto another?
#
# +-------------------------------------------------------------+
# | S -> NP VP   |                 S                            |
# |[NP -> Det N ]|                / \                           |
# |     ...      |              NP  VP                          |
# | N -> 'dog'   |                                              |
# | N -> 'cat'   |                                              |
# |     ...      |                                              |
# +--------------+                                              |
# |      NP      |                      Det     N               |
# |     /  \     |                       |      |               |
# |   Det   N    |  the    cat    saw   the    dog              |
# |              |                                              |
# +--------------+----------------------------------------------+
#
# Operations:
#   - connect a new treelet -- drag or click shadow
#   - delete a treelet -- right click
#     - if only connected to top, delete everything below
#     - if only connected to bottom, delete everything above
#   - connect top & bottom -- drag a leaf to a root or a root to a leaf
#   - disconnect top & bottom -- right click
#     - if connected to top & bottom, then disconnect

import re
from tkinter import (
    Button,
    Canvas,
    Entry,
    Frame,
    IntVar,
    Label,
    Scrollbar,
    Text,
    Tk,
    Toplevel,
)

from nltk.draw.tree import TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
    CanvasFrame,
    ColorizedList,
    ShowText,
    SymbolWidget,
    TextWidget,
)
from nltk.grammar import CFG, Nonterminal, _read_cfg_production, nonterminals
from nltk.tree import Tree

######################################################################
# Production List
######################################################################


class ProductionList(ColorizedList):
    ARROW = SymbolWidget.SYMBOLS["rightarrow"]

    def _init_colortags(self, textwidget, options):
        textwidget.tag_config("terminal", foreground="#006000")
        textwidget.tag_config("arrow", font="symbol", underline="0")
        textwidget.tag_config(
            "nonterminal", foreground="blue", font=("helvetica", -12, "bold")
        )

    def _item_repr(self, item):
        contents = []
        contents.append(("%s\t" % item.lhs(), "nonterminal"))
        contents.append((self.ARROW, "arrow"))
        for elt in item.rhs():
            if isinstance(elt, Nonterminal):
                contents.append((" %s" % elt.symbol(), "nonterminal"))
            else:
                contents.append((" %r" % elt, "terminal"))
        return contents


######################################################################
# CFG Editor
######################################################################

_CFGEditor_HELP = """

The CFG Editor can be used to create or modify context free grammars.
A context free grammar consists of a start symbol and a list of
productions.  The start symbol is specified by the text entry field in
the upper right hand corner of the editor; and the list of productions
are specified in the main text editing box.

Every non-blank line specifies a single production.  Each production
has the form "LHS -> RHS," where LHS is a single nonterminal, and RHS
is a list of nonterminals and terminals.

Nonterminals must be a single word, such as S or NP or NP_subj.
Currently, nonterminals must consists of alphanumeric characters and
underscores (_).  Nonterminals are colored blue.  If you place the
mouse over any nonterminal, then all occurrences of that nonterminal
will be highlighted.

Terminals must be surrounded by single quotes (') or double
quotes(\").  For example, "dog" and "New York" are terminals.
Currently, the string within the quotes must consist of alphanumeric
characters, underscores, and spaces.

To enter a new production, go to a blank line, and type a nonterminal,
followed by an arrow (->), followed by a sequence of terminals and
nonterminals.  Note that "->" (dash + greater-than) is automatically
converted to an arrow symbol.  When you move your cursor to a
different line, your production will automatically be colorized.  If
there are any errors, they will be highlighted in red.

Note that the order of the productions is significant for some
algorithms.  To re-order the productions, use cut and paste to move
them.

Use the buttons at the bottom of the window when you are done editing
the CFG:
  - Ok: apply the new CFG, and exit the editor.
  - Apply: apply the new CFG, and do not exit the editor.
  - Reset: revert to the original CFG, and do not exit the editor.
  - Cancel: revert to the original CFG, and exit the editor.

"""


class CFGEditor:
    """
    A dialog window for creating and editing context free grammars.
    ``CFGEditor`` imposes the following restrictions:

    - All nonterminals must be strings consisting of word
      characters.
    - All terminals must be strings consisting of word characters
      and space characters.
    """

    # Regular expressions used by _analyze_line.  Precompile them, so
    # we can process the text faster.
    ARROW = SymbolWidget.SYMBOLS["rightarrow"]
    _LHS_RE = re.compile(r"(^\s*\w+\s*)(->|(" + ARROW + "))")
    _ARROW_RE = re.compile(r"\s*(->|(" + ARROW + r"))\s*")
    _PRODUCTION_RE = re.compile(
        r"(^\s*\w+\s*)"
        + "(->|("  # LHS
        + ARROW
        + r"))\s*"
        + r"((\w+|'[\w ]*'|\"[\w ]*\"|\|)\s*)*$"  # arrow
    )  # RHS
    _TOKEN_RE = re.compile("\\w+|->|'[\\w ]+'|\"[\\w ]+\"|(" + ARROW + ")")
    _BOLD = ("helvetica", -12, "bold")

    def __init__(self, parent, cfg=None, set_cfg_callback=None):
        self._parent = parent
        if cfg is not None:
            self._cfg = cfg
        else:
            self._cfg = CFG(Nonterminal("S"), [])
        self._set_cfg_callback = set_cfg_callback

        self._highlight_matching_nonterminals = 1

        # Create the top-level window.
        self._top = Toplevel(parent)
        self._init_bindings()

        self._init_startframe()
        self._startframe.pack(side="top", fill="x", expand=0)
        self._init_prodframe()
        self._prodframe.pack(side="top", fill="both", expand=1)
        self._init_buttons()
        self._buttonframe.pack(side="bottom", fill="x", expand=0)

        self._textwidget.focus()

    def _init_startframe(self):
        frame = self._startframe = Frame(self._top)
        self._start = Entry(frame)
        self._start.pack(side="right")
        Label(frame, text="Start Symbol:").pack(side="right")
        Label(frame, text="Productions:").pack(side="left")
        self._start.insert(0, self._cfg.start().symbol())

    def _init_buttons(self):
        frame = self._buttonframe = Frame(self._top)
        Button(frame, text="Ok", command=self._ok, underline=0, takefocus=0).pack(
            side="left"
        )
        Button(frame, text="Apply", command=self._apply, underline=0, takefocus=0).pack(
            side="left"
        )
        Button(frame, text="Reset", command=self._reset, underline=0, takefocus=0).pack(
            side="left"
        )
        Button(
            frame, text="Cancel", command=self._cancel, underline=0, takefocus=0
        ).pack(side="left")
        Button(frame, text="Help", command=self._help, underline=0, takefocus=0).pack(
            side="right"
        )

    def _init_bindings(self):
        self._top.title("CFG Editor")
        self._top.bind("<Control-q>", self._cancel)
        self._top.bind("<Alt-q>", self._cancel)
        self._top.bind("<Control-d>", self._cancel)
        # self._top.bind('<Control-x>', self._cancel)
        self._top.bind("<Alt-x>", self._cancel)
        self._top.bind("<Escape>", self._cancel)
        # self._top.bind('<Control-c>', self._cancel)
        self._top.bind("<Alt-c>", self._cancel)

        self._top.bind("<Control-o>", self._ok)
        self._top.bind("<Alt-o>", self._ok)
        self._top.bind("<Control-a>", self._apply)
        self._top.bind("<Alt-a>", self._apply)
        self._top.bind("<Control-r>", self._reset)
        self._top.bind("<Alt-r>", self._reset)
        self._top.bind("<Control-h>", self._help)
        self._top.bind("<Alt-h>", self._help)
        self._top.bind("<F1>", self._help)

    def _init_prodframe(self):
        self._prodframe = Frame(self._top)

        # Create the basic Text widget & scrollbar.
        self._textwidget = Text(
            self._prodframe, background="#e0e0e0", exportselection=1
        )
        self._textscroll = Scrollbar(self._prodframe, takefocus=0, orient="vertical")
        self._textwidget.config(yscrollcommand=self._textscroll.set)
        self._textscroll.config(command=self._textwidget.yview)
        self._textscroll.pack(side="right", fill="y")
        self._textwidget.pack(expand=1, fill="both", side="left")

        # Initialize the colorization tags.  Each nonterminal gets its
        # own tag, so they aren't listed here.
        self._textwidget.tag_config("terminal", foreground="#006000")
        self._textwidget.tag_config("arrow", font="symbol")
        self._textwidget.tag_config("error", background="red")

        # Keep track of what line they're on.  We use that to remember
        # to re-analyze a line whenever they leave it.
        self._linenum = 0

        # Expand "->" to an arrow.
        self._top.bind(">", self._replace_arrows)

        # Re-colorize lines when appropriate.
        self._top.bind("<<Paste>>", self._analyze)
        self._top.bind("<KeyPress>", self._check_analyze)
        self._top.bind("<ButtonPress>", self._check_analyze)

        # Tab cycles focus. (why doesn't this work??)
        def cycle(e, textwidget=self._textwidget):
            textwidget.tk_focusNext().focus()

        self._textwidget.bind("<Tab>", cycle)

        prod_tuples = [(p.lhs(), [p.rhs()]) for p in self._cfg.productions()]
        for i in range(len(prod_tuples) - 1, 0, -1):
            if prod_tuples[i][0] == prod_tuples[i - 1][0]:
                if () in prod_tuples[i][1]:
                    continue
                if () in prod_tuples[i - 1][1]:
                    continue
                print(prod_tuples[i - 1][1])
                print(prod_tuples[i][1])
                prod_tuples[i - 1][1].extend(prod_tuples[i][1])
                del prod_tuples[i]

        for lhs, rhss in prod_tuples:
            print(lhs, rhss)
            s = "%s ->" % lhs
            for rhs in rhss:
                for elt in rhs:
                    if isinstance(elt, Nonterminal):
                        s += " %s" % elt
                    else:
                        s += " %r" % elt
                s += " |"
            s = s[:-2] + "\n"
            self._textwidget.insert("end", s)

        self._analyze()

    #         # Add the producitons to the text widget, and colorize them.
    #         prod_by_lhs = {}
    #         for prod in self._cfg.productions():
    #             if len(prod.rhs()) > 0:
    #                 prod_by_lhs.setdefault(prod.lhs(),[]).append(prod)
    #         for (lhs, prods) in prod_by_lhs.items():
    #             self._textwidget.insert('end', '%s ->' % lhs)
    #             self._textwidget.insert('end', self._rhs(prods[0]))
    #             for prod in prods[1:]:
    #                 print '\t|'+self._rhs(prod),
    #                 self._textwidget.insert('end', '\t|'+self._rhs(prod))
    #             print
    #             self._textwidget.insert('end', '\n')
    #         for prod in self._cfg.productions():
    #             if len(prod.rhs()) == 0:
    #                 self._textwidget.insert('end', '%s' % prod)
    #         self._analyze()

    #     def _rhs(self, prod):
    #         s = ''
    #         for elt in prod.rhs():
    #             if isinstance(elt, Nonterminal): s += ' %s' % elt.symbol()
    #             else: s += ' %r' % elt
    #         return s

    def _clear_tags(self, linenum):
        """
        Remove all tags (except ``arrow`` and ``sel``) from the given
        line of the text widget used for editing the productions.
        """
        start = "%d.0" % linenum
        end = "%d.end" % linenum
        for tag in self._textwidget.tag_names():
            if tag not in ("arrow", "sel"):
                self._textwidget.tag_remove(tag, start, end)

    def _check_analyze(self, *e):
        """
        Check if we've moved to a new line.  If we have, then remove
        all colorization from the line we moved to, and re-colorize
        the line that we moved from.
        """
        linenum = int(self._textwidget.index("insert").split(".")[0])
        if linenum != self._linenum:
            self._clear_tags(linenum)
            self._analyze_line(self._linenum)
            self._linenum = linenum

    def _replace_arrows(self, *e):
        """
        Replace any ``'->'`` text strings with arrows (char \\256, in
        symbol font).  This searches the whole buffer, but is fast
        enough to be done anytime they press '>'.
        """
        arrow = "1.0"
        while True:
            arrow = self._textwidget.search("->", arrow, "end+1char")
            if arrow == "":
                break
            self._textwidget.delete(arrow, arrow + "+2char")
            self._textwidget.insert(arrow, self.ARROW, "arrow")
            self._textwidget.insert(arrow, "\t")

        arrow = "1.0"
        while True:
            arrow = self._textwidget.search(self.ARROW, arrow + "+1char", "end+1char")
            if arrow == "":
                break
            self._textwidget.tag_add("arrow", arrow, arrow + "+1char")

    def _analyze_token(self, match, linenum):
        """
        Given a line number and a regexp match for a token on that
        line, colorize the token.  Note that the regexp match gives us
        the token's text, start index (on the line), and end index (on
        the line).
        """
        # What type of token is it?
        if match.group()[0] in "'\"":
            tag = "terminal"
        elif match.group() in ("->", self.ARROW):
            tag = "arrow"
        else:
            # If it's a nonterminal, then set up new bindings, so we
            # can highlight all instances of that nonterminal when we
            # put the mouse over it.
            tag = "nonterminal_" + match.group()
            if tag not in self._textwidget.tag_names():
                self._init_nonterminal_tag(tag)

        start = "%d.%d" % (linenum, match.start())
        end = "%d.%d" % (linenum, match.end())
        self._textwidget.tag_add(tag, start, end)

    def _init_nonterminal_tag(self, tag, foreground="blue"):
        self._textwidget.tag_config(tag, foreground=foreground, font=CFGEditor._BOLD)
        if not self._highlight_matching_nonterminals:
            return

        def enter(e, textwidget=self._textwidget, tag=tag):
            textwidget.tag_config(tag, background="#80ff80")

        def leave(e, textwidget=self._textwidget, tag=tag):
            textwidget.tag_config(tag, background="")

        self._textwidget.tag_bind(tag, "<Enter>", enter)
        self._textwidget.tag_bind(tag, "<Leave>", leave)

    def _analyze_line(self, linenum):
        """
        Colorize a given line.
        """
        # Get rid of any tags that were previously on the line.
        self._clear_tags(linenum)

        # Get the line line's text string.
        line = self._textwidget.get(repr(linenum) + ".0", repr(linenum) + ".end")

        # If it's a valid production, then colorize each token.
        if CFGEditor._PRODUCTION_RE.match(line):
            # It's valid; Use _TOKEN_RE to tokenize the production,
            # and call analyze_token on each token.
            def analyze_token(match, self=self, linenum=linenum):
                self._analyze_token(match, linenum)
                return ""

            CFGEditor._TOKEN_RE.sub(analyze_token, line)
        elif line.strip() != "":
            # It's invalid; show the user where the error is.
            self._mark_error(linenum, line)

    def _mark_error(self, linenum, line):
        """
        Mark the location of an error in a line.
        """
        arrowmatch = CFGEditor._ARROW_RE.search(line)
        if not arrowmatch:
            # If there's no arrow at all, highlight the whole line.
            start = "%d.0" % linenum
            end = "%d.end" % linenum
        elif not CFGEditor._LHS_RE.match(line):
            # Otherwise, if the LHS is bad, highlight it.
            start = "%d.0" % linenum
            end = "%d.%d" % (linenum, arrowmatch.start())
        else:
            # Otherwise, highlight the RHS.
            start = "%d.%d" % (linenum, arrowmatch.end())
            end = "%d.end" % linenum

        # If we're highlighting 0 chars, highlight the whole line.
        if self._textwidget.compare(start, "==", end):
            start = "%d.0" % linenum
            end = "%d.end" % linenum
        self._textwidget.tag_add("error", start, end)

    def _analyze(self, *e):
        """
        Replace ``->`` with arrows, and colorize the entire buffer.
        """
        self._replace_arrows()
        numlines = int(self._textwidget.index("end").split(".")[0])
        for linenum in range(1, numlines + 1):  # line numbers start at 1.
            self._analyze_line(linenum)

    def _parse_productions(self):
        """
        Parse the current contents of the textwidget buffer, to create
        a list of productions.
        """
        productions = []

        # Get the text, normalize it, and split it into lines.
        text = self._textwidget.get("1.0", "end")
        text = re.sub(self.ARROW, "->", text)
        text = re.sub("\t", " ", text)
        lines = text.split("\n")

        # Convert each line to a CFG production
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            productions += _read_cfg_production(line)
            # if line.strip() == '': continue
            # if not CFGEditor._PRODUCTION_RE.match(line):
            #    raise ValueError('Bad production string %r' % line)
            #
            # (lhs_str, rhs_str) = line.split('->')
            # lhs = Nonterminal(lhs_str.strip())
            # rhs = []
            # def parse_token(match, rhs=rhs):
            #    token = match.group()
            #    if token[0] in "'\"": rhs.append(token[1:-1])
            #    else: rhs.append(Nonterminal(token))
            #    return ''
            # CFGEditor._TOKEN_RE.sub(parse_token, rhs_str)
            #
            # productions.append(Production(lhs, *rhs))

        return productions

    def _destroy(self, *e):
        if self._top is None:
            return
        self._top.destroy()
        self._top = None

    def _ok(self, *e):
        self._apply()
        self._destroy()

    def _apply(self, *e):
        productions = self._parse_productions()
        start = Nonterminal(self._start.get())
        cfg = CFG(start, productions)
        if self._set_cfg_callback is not None:
            self._set_cfg_callback(cfg)

    def _reset(self, *e):
        self._textwidget.delete("1.0", "end")
        for production in self._cfg.productions():
            self._textwidget.insert("end", "%s\n" % production)
        self._analyze()
        if self._set_cfg_callback is not None:
            self._set_cfg_callback(self._cfg)

    def _cancel(self, *e):
        try:
            self._reset()
        except:
            pass
        self._destroy()

    def _help(self, *e):
        # The default font's not very legible; try using 'fixed' instead.
        try:
            ShowText(
                self._parent,
                "Help: Chart Parser Demo",
                (_CFGEditor_HELP).strip(),
                width=75,
                font="fixed",
            )
        except:
            ShowText(
                self._parent,
                "Help: Chart Parser Demo",
                (_CFGEditor_HELP).strip(),
                width=75,
            )


######################################################################
# New Demo (built tree based on cfg)
######################################################################


class CFGDemo:
    def __init__(self, grammar, text):
        self._grammar = grammar
        self._text = text

        # Set up the main window.
        self._top = Tk()
        self._top.title("Context Free Grammar Demo")

        # Base font size
        self._size = IntVar(self._top)
        self._size.set(12)  # = medium

        # Set up the key bindings
        self._init_bindings(self._top)

        # Create the basic frames
        frame1 = Frame(self._top)
        frame1.pack(side="left", fill="y", expand=0)
        self._init_menubar(self._top)
        self._init_buttons(self._top)
        self._init_grammar(frame1)
        self._init_treelet(frame1)
        self._init_workspace(self._top)

    # //////////////////////////////////////////////////
    # Initialization
    # //////////////////////////////////////////////////

    def _init_bindings(self, top):
        top.bind("<Control-q>", self.destroy)

    def _init_menubar(self, parent):
        pass

    def _init_buttons(self, parent):
        pass

    def _init_grammar(self, parent):
        self._prodlist = ProductionList(parent, self._grammar, width=20)
        self._prodlist.pack(side="top", fill="both", expand=1)
        self._prodlist.focus()
        self._prodlist.add_callback("select", self._selectprod_cb)
        self._prodlist.add_callback("move", self._selectprod_cb)

    def _init_treelet(self, parent):
        self._treelet_canvas = Canvas(parent, background="white")
        self._treelet_canvas.pack(side="bottom", fill="x")
        self._treelet = None

    def _init_workspace(self, parent):
        self._workspace = CanvasFrame(parent, background="white")
        self._workspace.pack(side="right", fill="both", expand=1)
        self._tree = None
        self.reset_workspace()

    # //////////////////////////////////////////////////
    # Workspace
    # //////////////////////////////////////////////////

    def reset_workspace(self):
        c = self._workspace.canvas()
        fontsize = int(self._size.get())
        node_font = ("helvetica", -(fontsize + 4), "bold")
        leaf_font = ("helvetica", -(fontsize + 2))

        # Remove the old tree
        if self._tree is not None:
            self._workspace.remove_widget(self._tree)

        # The root of the tree.
        start = self._grammar.start().symbol()
        rootnode = TextWidget(c, start, font=node_font, draggable=1)

        # The leaves of the tree.
        leaves = []
        for word in self._text:
            leaves.append(TextWidget(c, word, font=leaf_font, draggable=1))

        # Put it all together into one tree
        self._tree = TreeSegmentWidget(c, rootnode, leaves, color="white")

        # Add it to the workspace.
        self._workspace.add_widget(self._tree)

        # Move the leaves to the bottom of the workspace.
        for leaf in leaves:
            leaf.move(0, 100)

        # self._nodes = {start:1}
        # self._leaves = dict([(l,1) for l in leaves])

    def workspace_markprod(self, production):
        pass

    def _markproduction(self, prod, tree=None):
        if tree is None:
            tree = self._tree
        for i in range(len(tree.subtrees()) - len(prod.rhs())):
            if tree["color", i] == "white":
                self._markproduction  # FIXME: Is this necessary at all?

            for j, node in enumerate(prod.rhs()):
                widget = tree.subtrees()[i + j]
                if (
                    isinstance(node, Nonterminal)
                    and isinstance(widget, TreeSegmentWidget)
                    and node.symbol == widget.label().text()
                ):
                    pass  # matching nonterminal
                elif (
                    isinstance(node, str)
                    and isinstance(widget, TextWidget)
                    and node == widget.text()
                ):
                    pass  # matching nonterminal
                else:
                    break
            else:
                # Everything matched!
                print("MATCH AT", i)

    # //////////////////////////////////////////////////
    # Grammar
    # //////////////////////////////////////////////////

    def _selectprod_cb(self, production):
        canvas = self._treelet_canvas

        self._prodlist.highlight(production)
        if self._treelet is not None:
            self._treelet.destroy()

        # Convert the production to a tree.
        rhs = production.rhs()
        for (i, elt) in enumerate(rhs):
            if isinstance(elt, Nonterminal):
                elt = Tree(elt)
        tree = Tree(production.lhs().symbol(), *rhs)

        # Draw the tree in the treelet area.
        fontsize = int(self._size.get())
        node_font = ("helvetica", -(fontsize + 4), "bold")
        leaf_font = ("helvetica", -(fontsize + 2))
        self._treelet = tree_to_treesegment(
            canvas, tree, node_font=node_font, leaf_font=leaf_font
        )
        self._treelet["draggable"] = 1

        # Center the treelet.
        (x1, y1, x2, y2) = self._treelet.bbox()
        w, h = int(canvas["width"]), int(canvas["height"])
        self._treelet.move((w - x1 - x2) / 2, (h - y1 - y2) / 2)

        # Mark the places where we can add it to the workspace.
        self._markproduction(production)

    def destroy(self, *args):
        self._top.destroy()

    def mainloop(self, *args, **kwargs):
        self._top.mainloop(*args, **kwargs)


def demo2():
    from nltk import CFG, Nonterminal, Production

    nonterminals = "S VP NP PP P N Name V Det"
    (S, VP, NP, PP, P, N, Name, V, Det) = (Nonterminal(s) for s in nonterminals.split())
    productions = (
        # Syntactic Productions
        Production(S, [NP, VP]),
        Production(NP, [Det, N]),
        Production(NP, [NP, PP]),
        Production(VP, [VP, PP]),
        Production(VP, [V, NP, PP]),
        Production(VP, [V, NP]),
        Production(PP, [P, NP]),
        Production(PP, []),
        Production(PP, ["up", "over", NP]),
        # Lexical Productions
        Production(NP, ["I"]),
        Production(Det, ["the"]),
        Production(Det, ["a"]),
        Production(N, ["man"]),
        Production(V, ["saw"]),
        Production(P, ["in"]),
        Production(P, ["with"]),
        Production(N, ["park"]),
        Production(N, ["dog"]),
        Production(N, ["statue"]),
        Production(Det, ["my"]),
    )
    grammar = CFG(S, productions)

    text = "I saw a man in the park".split()
    d = CFGDemo(grammar, text)
    d.mainloop()


######################################################################
# Old Demo
######################################################################


def demo():
    from nltk import CFG, Nonterminal

    nonterminals = "S VP NP PP P N Name V Det"
    (S, VP, NP, PP, P, N, Name, V, Det) = (Nonterminal(s) for s in nonterminals.split())

    grammar = CFG.fromstring(
        """
    S -> NP VP
    PP -> P NP
    NP -> Det N
    NP -> NP PP
    VP -> V NP
    VP -> VP PP
    Det -> 'a'
    Det -> 'the'
    Det -> 'my'
    NP -> 'I'
    N -> 'dog'
    N -> 'man'
    N -> 'park'
    N -> 'statue'
    V -> 'saw'
    P -> 'in'
    P -> 'up'
    P -> 'over'
    P -> 'with'
    """
    )

    def cb(grammar):
        print(grammar)

    top = Tk()
    editor = CFGEditor(top, grammar, cb)
    Label(top, text="\nTesting CFG Editor\n").pack()
    Button(top, text="Quit", command=top.destroy).pack()
    top.mainloop()


def demo3():
    from nltk import Production

    (S, VP, NP, PP, P, N, Name, V, Det) = nonterminals(
        "S, VP, NP, PP, P, N, Name, V, Det"
    )

    productions = (
        # Syntactic Productions
        Production(S, [NP, VP]),
        Production(NP, [Det, N]),
        Production(NP, [NP, PP]),
        Production(VP, [VP, PP]),
        Production(VP, [V, NP, PP]),
        Production(VP, [V, NP]),
        Production(PP, [P, NP]),
        Production(PP, []),
        Production(PP, ["up", "over", NP]),
        # Lexical Productions
        Production(NP, ["I"]),
        Production(Det, ["the"]),
        Production(Det, ["a"]),
        Production(N, ["man"]),
        Production(V, ["saw"]),
        Production(P, ["in"]),
        Production(P, ["with"]),
        Production(N, ["park"]),
        Production(N, ["dog"]),
        Production(N, ["statue"]),
        Production(Det, ["my"]),
    )

    t = Tk()

    def destroy(e, t=t):
        t.destroy()

    t.bind("q", destroy)
    p = ProductionList(t, productions)
    p.pack(expand=1, fill="both")
    p.add_callback("select", p.markonly)
    p.add_callback("move", p.markonly)
    p.focus()
    p.mark(productions[2])
    p.mark(productions[8])


if __name__ == "__main__":
    demo()
