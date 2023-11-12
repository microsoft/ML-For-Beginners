# Natural Language Toolkit: Recursive Descent Parser Application
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A graphical tool for exploring the recursive descent parser.

The recursive descent parser maintains a tree, which records the
structure of the portion of the text that has been parsed.  It uses
CFG productions to expand the fringe of the tree, and matches its
leaves against the text.  Initially, the tree contains the start
symbol ("S").  It is shown in the main canvas, to the right of the
list of available expansions.

The parser builds up a tree structure for the text using three
operations:

  - "expand" uses a CFG production to add children to a node on the
    fringe of the tree.
  - "match" compares a leaf in the tree to a text token.
  - "backtrack" returns the tree to its state before the most recent
    expand or match operation.

The parser maintains a list of tree locations called a "frontier" to
remember which nodes have not yet been expanded and which leaves have
not yet been matched against the text.  The leftmost frontier node is
shown in green, and the other frontier nodes are shown in blue.  The
parser always performs expand and match operations on the leftmost
element of the frontier.

You can control the parser's operation by using the "expand," "match,"
and "backtrack" buttons; or you can use the "step" button to let the
parser automatically decide which operation to apply.  The parser uses
the following rules to decide which operation to apply:

  - If the leftmost frontier element is a token, try matching it.
  - If the leftmost frontier element is a node, try expanding it with
    the first untried expansion.
  - Otherwise, backtrack.

The "expand" button applies the untried expansion whose CFG production
is listed earliest in the grammar.  To manually choose which expansion
to apply, click on a CFG production from the list of available
expansions, on the left side of the main window.

The "autostep" button will let the parser continue applying
applications to the tree until it reaches a complete parse.  You can
cancel an autostep in progress at any time by clicking on the
"autostep" button again.

Keyboard Shortcuts::
      [Space]\t Perform the next expand, match, or backtrack operation
      [a]\t Step through operations until the next complete parse
      [e]\t Perform an expand operation
      [m]\t Perform a match operation
      [b]\t Perform a backtrack operation
      [Delete]\t Reset the parser
      [g]\t Show/hide available expansions list
      [h]\t Help
      [Ctrl-p]\t Print
      [q]\t Quit
"""

from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font

from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingRecursiveDescentParser
from nltk.tree import Tree
from nltk.util import in_idle


class RecursiveDescentApp:
    """
    A graphical tool for exploring the recursive descent parser.  The tool
    displays the parser's tree and the remaining text, and allows the
    user to control the parser's operation.  In particular, the user
    can expand subtrees on the frontier, match tokens on the frontier
    against the text, and backtrack.  A "step" button simply steps
    through the parsing process, performing the operations that
    ``RecursiveDescentParser`` would use.
    """

    def __init__(self, grammar, sent, trace=0):
        self._sent = sent
        self._parser = SteppingRecursiveDescentParser(grammar, trace)

        # Set up the main window.
        self._top = Tk()
        self._top.title("Recursive Descent Parser Application")

        # Set up key bindings.
        self._init_bindings()

        # Initialize the fonts.
        self._init_fonts(self._top)

        # Animations.  animating_lock is a lock to prevent the demo
        # from performing new operations while it's animating.
        self._animation_frames = IntVar(self._top)
        self._animation_frames.set(5)
        self._animating_lock = 0
        self._autostep = 0

        # The user can hide the grammar.
        self._show_grammar = IntVar(self._top)
        self._show_grammar.set(1)

        # Create the basic frames.
        self._init_menubar(self._top)
        self._init_buttons(self._top)
        self._init_feedback(self._top)
        self._init_grammar(self._top)
        self._init_canvas(self._top)

        # Initialize the parser.
        self._parser.initialize(self._sent)

        # Resize callback
        self._canvas.bind("<Configure>", self._configure)

    #########################################
    ##  Initialization Helpers
    #########################################

    def _init_fonts(self, root):
        # See: <http://www.astro.washington.edu/owen/ROTKFolklore.html>
        self._sysfont = Font(font=Button()["font"])
        root.option_add("*Font", self._sysfont)

        # TWhat's our font size (default=same as sysfont)
        self._size = IntVar(root)
        self._size.set(self._sysfont.cget("size"))

        self._boldfont = Font(family="helvetica", weight="bold", size=self._size.get())
        self._font = Font(family="helvetica", size=self._size.get())
        if self._size.get() < 0:
            big = self._size.get() - 2
        else:
            big = self._size.get() + 2
        self._bigfont = Font(family="helvetica", weight="bold", size=big)

    def _init_grammar(self, parent):
        # Grammar view.
        self._prodframe = listframe = Frame(parent)
        self._prodframe.pack(fill="both", side="left", padx=2)
        self._prodlist_label = Label(
            self._prodframe, font=self._boldfont, text="Available Expansions"
        )
        self._prodlist_label.pack()
        self._prodlist = Listbox(
            self._prodframe,
            selectmode="single",
            relief="groove",
            background="white",
            foreground="#909090",
            font=self._font,
            selectforeground="#004040",
            selectbackground="#c0f0c0",
        )

        self._prodlist.pack(side="right", fill="both", expand=1)

        self._productions = list(self._parser.grammar().productions())
        for production in self._productions:
            self._prodlist.insert("end", ("  %s" % production))
        self._prodlist.config(height=min(len(self._productions), 25))

        # Add a scrollbar if there are more than 25 productions.
        if len(self._productions) > 25:
            listscroll = Scrollbar(self._prodframe, orient="vertical")
            self._prodlist.config(yscrollcommand=listscroll.set)
            listscroll.config(command=self._prodlist.yview)
            listscroll.pack(side="left", fill="y")

        # If they select a production, apply it.
        self._prodlist.bind("<<ListboxSelect>>", self._prodlist_select)

    def _init_bindings(self):
        # Key bindings are a good thing.
        self._top.bind("<Control-q>", self.destroy)
        self._top.bind("<Control-x>", self.destroy)
        self._top.bind("<Escape>", self.destroy)
        self._top.bind("e", self.expand)
        # self._top.bind('<Alt-e>', self.expand)
        # self._top.bind('<Control-e>', self.expand)
        self._top.bind("m", self.match)
        self._top.bind("<Alt-m>", self.match)
        self._top.bind("<Control-m>", self.match)
        self._top.bind("b", self.backtrack)
        self._top.bind("<Alt-b>", self.backtrack)
        self._top.bind("<Control-b>", self.backtrack)
        self._top.bind("<Control-z>", self.backtrack)
        self._top.bind("<BackSpace>", self.backtrack)
        self._top.bind("a", self.autostep)
        # self._top.bind('<Control-a>', self.autostep)
        self._top.bind("<Control-space>", self.autostep)
        self._top.bind("<Control-c>", self.cancel_autostep)
        self._top.bind("<space>", self.step)
        self._top.bind("<Delete>", self.reset)
        self._top.bind("<Control-p>", self.postscript)
        # self._top.bind('<h>', self.help)
        # self._top.bind('<Alt-h>', self.help)
        self._top.bind("<Control-h>", self.help)
        self._top.bind("<F1>", self.help)
        # self._top.bind('<g>', self.toggle_grammar)
        # self._top.bind('<Alt-g>', self.toggle_grammar)
        # self._top.bind('<Control-g>', self.toggle_grammar)
        self._top.bind("<Control-g>", self.edit_grammar)
        self._top.bind("<Control-t>", self.edit_sentence)

    def _init_buttons(self, parent):
        # Set up the frames.
        self._buttonframe = buttonframe = Frame(parent)
        buttonframe.pack(fill="none", side="bottom", padx=3, pady=2)
        Button(
            buttonframe,
            text="Step",
            background="#90c0d0",
            foreground="black",
            command=self.step,
        ).pack(side="left")
        Button(
            buttonframe,
            text="Autostep",
            background="#90c0d0",
            foreground="black",
            command=self.autostep,
        ).pack(side="left")
        Button(
            buttonframe,
            text="Expand",
            underline=0,
            background="#90f090",
            foreground="black",
            command=self.expand,
        ).pack(side="left")
        Button(
            buttonframe,
            text="Match",
            underline=0,
            background="#90f090",
            foreground="black",
            command=self.match,
        ).pack(side="left")
        Button(
            buttonframe,
            text="Backtrack",
            underline=0,
            background="#f0a0a0",
            foreground="black",
            command=self.backtrack,
        ).pack(side="left")
        # Replace autostep...

    #         self._autostep_button = Button(buttonframe, text='Autostep',
    #                                        underline=0, command=self.autostep)
    #         self._autostep_button.pack(side='left')

    def _configure(self, event):
        self._autostep = 0
        (x1, y1, x2, y2) = self._cframe.scrollregion()
        y2 = event.height - 6
        self._canvas["scrollregion"] = "%d %d %d %d" % (x1, y1, x2, y2)
        self._redraw()

    def _init_feedback(self, parent):
        self._feedbackframe = feedbackframe = Frame(parent)
        feedbackframe.pack(fill="x", side="bottom", padx=3, pady=3)
        self._lastoper_label = Label(
            feedbackframe, text="Last Operation:", font=self._font
        )
        self._lastoper_label.pack(side="left")
        lastoperframe = Frame(feedbackframe, relief="sunken", border=1)
        lastoperframe.pack(fill="x", side="right", expand=1, padx=5)
        self._lastoper1 = Label(
            lastoperframe, foreground="#007070", background="#f0f0f0", font=self._font
        )
        self._lastoper2 = Label(
            lastoperframe,
            anchor="w",
            width=30,
            foreground="#004040",
            background="#f0f0f0",
            font=self._font,
        )
        self._lastoper1.pack(side="left")
        self._lastoper2.pack(side="left", fill="x", expand=1)

    def _init_canvas(self, parent):
        self._cframe = CanvasFrame(
            parent,
            background="white",
            # width=525, height=250,
            closeenough=10,
            border=2,
            relief="sunken",
        )
        self._cframe.pack(expand=1, fill="both", side="top", pady=2)
        canvas = self._canvas = self._cframe.canvas()

        # Initially, there's no tree or text
        self._tree = None
        self._textwidgets = []
        self._textline = None

    def _init_menubar(self, parent):
        menubar = Menu(parent)

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(
            label="Reset Parser", underline=0, command=self.reset, accelerator="Del"
        )
        filemenu.add_command(
            label="Print to Postscript",
            underline=0,
            command=self.postscript,
            accelerator="Ctrl-p",
        )
        filemenu.add_command(
            label="Exit", underline=1, command=self.destroy, accelerator="Ctrl-x"
        )
        menubar.add_cascade(label="File", underline=0, menu=filemenu)

        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(
            label="Edit Grammar",
            underline=5,
            command=self.edit_grammar,
            accelerator="Ctrl-g",
        )
        editmenu.add_command(
            label="Edit Text",
            underline=5,
            command=self.edit_sentence,
            accelerator="Ctrl-t",
        )
        menubar.add_cascade(label="Edit", underline=0, menu=editmenu)

        rulemenu = Menu(menubar, tearoff=0)
        rulemenu.add_command(
            label="Step", underline=1, command=self.step, accelerator="Space"
        )
        rulemenu.add_separator()
        rulemenu.add_command(
            label="Match", underline=0, command=self.match, accelerator="Ctrl-m"
        )
        rulemenu.add_command(
            label="Expand", underline=0, command=self.expand, accelerator="Ctrl-e"
        )
        rulemenu.add_separator()
        rulemenu.add_command(
            label="Backtrack", underline=0, command=self.backtrack, accelerator="Ctrl-b"
        )
        menubar.add_cascade(label="Apply", underline=0, menu=rulemenu)

        viewmenu = Menu(menubar, tearoff=0)
        viewmenu.add_checkbutton(
            label="Show Grammar",
            underline=0,
            variable=self._show_grammar,
            command=self._toggle_grammar,
        )
        viewmenu.add_separator()
        viewmenu.add_radiobutton(
            label="Tiny",
            variable=self._size,
            underline=0,
            value=10,
            command=self.resize,
        )
        viewmenu.add_radiobutton(
            label="Small",
            variable=self._size,
            underline=0,
            value=12,
            command=self.resize,
        )
        viewmenu.add_radiobutton(
            label="Medium",
            variable=self._size,
            underline=0,
            value=14,
            command=self.resize,
        )
        viewmenu.add_radiobutton(
            label="Large",
            variable=self._size,
            underline=0,
            value=18,
            command=self.resize,
        )
        viewmenu.add_radiobutton(
            label="Huge",
            variable=self._size,
            underline=0,
            value=24,
            command=self.resize,
        )
        menubar.add_cascade(label="View", underline=0, menu=viewmenu)

        animatemenu = Menu(menubar, tearoff=0)
        animatemenu.add_radiobutton(
            label="No Animation", underline=0, variable=self._animation_frames, value=0
        )
        animatemenu.add_radiobutton(
            label="Slow Animation",
            underline=0,
            variable=self._animation_frames,
            value=10,
            accelerator="-",
        )
        animatemenu.add_radiobutton(
            label="Normal Animation",
            underline=0,
            variable=self._animation_frames,
            value=5,
            accelerator="=",
        )
        animatemenu.add_radiobutton(
            label="Fast Animation",
            underline=0,
            variable=self._animation_frames,
            value=2,
            accelerator="+",
        )
        menubar.add_cascade(label="Animate", underline=1, menu=animatemenu)

        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", underline=0, command=self.about)
        helpmenu.add_command(
            label="Instructions", underline=0, command=self.help, accelerator="F1"
        )
        menubar.add_cascade(label="Help", underline=0, menu=helpmenu)

        parent.config(menu=menubar)

    #########################################
    ##  Helper
    #########################################

    def _get(self, widget, treeloc):
        for i in treeloc:
            widget = widget.subtrees()[i]
        if isinstance(widget, TreeSegmentWidget):
            widget = widget.label()
        return widget

    #########################################
    ##  Main draw procedure
    #########################################

    def _redraw(self):
        canvas = self._canvas

        # Delete the old tree, widgets, etc.
        if self._tree is not None:
            self._cframe.destroy_widget(self._tree)
        for twidget in self._textwidgets:
            self._cframe.destroy_widget(twidget)
        if self._textline is not None:
            self._canvas.delete(self._textline)

        # Draw the tree.
        helv = ("helvetica", -self._size.get())
        bold = ("helvetica", -self._size.get(), "bold")
        attribs = {
            "tree_color": "#000000",
            "tree_width": 2,
            "node_font": bold,
            "leaf_font": helv,
        }
        tree = self._parser.tree()
        self._tree = tree_to_treesegment(canvas, tree, **attribs)
        self._cframe.add_widget(self._tree, 30, 5)

        # Draw the text.
        helv = ("helvetica", -self._size.get())
        bottom = y = self._cframe.scrollregion()[3]
        self._textwidgets = [
            TextWidget(canvas, word, font=self._font) for word in self._sent
        ]
        for twidget in self._textwidgets:
            self._cframe.add_widget(twidget, 0, 0)
            twidget.move(0, bottom - twidget.bbox()[3] - 5)
            y = min(y, twidget.bbox()[1])

        # Draw a line over the text, to separate it from the tree.
        self._textline = canvas.create_line(-5000, y - 5, 5000, y - 5, dash=".")

        # Highlight appropriate nodes.
        self._highlight_nodes()
        self._highlight_prodlist()

        # Make sure the text lines up.
        self._position_text()

    def _redraw_quick(self):
        # This should be more-or-less sufficient after an animation.
        self._highlight_nodes()
        self._highlight_prodlist()
        self._position_text()

    def _highlight_nodes(self):
        # Highlight the list of nodes to be checked.
        bold = ("helvetica", -self._size.get(), "bold")
        for treeloc in self._parser.frontier()[:1]:
            self._get(self._tree, treeloc)["color"] = "#20a050"
            self._get(self._tree, treeloc)["font"] = bold
        for treeloc in self._parser.frontier()[1:]:
            self._get(self._tree, treeloc)["color"] = "#008080"

    def _highlight_prodlist(self):
        # Highlight the productions that can be expanded.
        # Boy, too bad tkinter doesn't implement Listbox.itemconfig;
        # that would be pretty useful here.
        self._prodlist.delete(0, "end")
        expandable = self._parser.expandable_productions()
        untried = self._parser.untried_expandable_productions()
        productions = self._productions
        for index in range(len(productions)):
            if productions[index] in expandable:
                if productions[index] in untried:
                    self._prodlist.insert(index, " %s" % productions[index])
                else:
                    self._prodlist.insert(index, " %s (TRIED)" % productions[index])
                self._prodlist.selection_set(index)
            else:
                self._prodlist.insert(index, " %s" % productions[index])

    def _position_text(self):
        # Line up the text widgets that are matched against the tree
        numwords = len(self._sent)
        num_matched = numwords - len(self._parser.remaining_text())
        leaves = self._tree_leaves()[:num_matched]
        xmax = self._tree.bbox()[0]
        for i in range(0, len(leaves)):
            widget = self._textwidgets[i]
            leaf = leaves[i]
            widget["color"] = "#006040"
            leaf["color"] = "#006040"
            widget.move(leaf.bbox()[0] - widget.bbox()[0], 0)
            xmax = widget.bbox()[2] + 10

        # Line up the text widgets that are not matched against the tree.
        for i in range(len(leaves), numwords):
            widget = self._textwidgets[i]
            widget["color"] = "#a0a0a0"
            widget.move(xmax - widget.bbox()[0], 0)
            xmax = widget.bbox()[2] + 10

        # If we have a complete parse, make everything green :)
        if self._parser.currently_complete():
            for twidget in self._textwidgets:
                twidget["color"] = "#00a000"

        # Move the matched leaves down to the text.
        for i in range(0, len(leaves)):
            widget = self._textwidgets[i]
            leaf = leaves[i]
            dy = widget.bbox()[1] - leaf.bbox()[3] - 10.0
            dy = max(dy, leaf.parent().label().bbox()[3] - leaf.bbox()[3] + 10)
            leaf.move(0, dy)

    def _tree_leaves(self, tree=None):
        if tree is None:
            tree = self._tree
        if isinstance(tree, TreeSegmentWidget):
            leaves = []
            for child in tree.subtrees():
                leaves += self._tree_leaves(child)
            return leaves
        else:
            return [tree]

    #########################################
    ##  Button Callbacks
    #########################################

    def destroy(self, *e):
        self._autostep = 0
        if self._top is None:
            return
        self._top.destroy()
        self._top = None

    def reset(self, *e):
        self._autostep = 0
        self._parser.initialize(self._sent)
        self._lastoper1["text"] = "Reset Application"
        self._lastoper2["text"] = ""
        self._redraw()

    def autostep(self, *e):
        if self._animation_frames.get() == 0:
            self._animation_frames.set(2)
        if self._autostep:
            self._autostep = 0
        else:
            self._autostep = 1
            self._step()

    def cancel_autostep(self, *e):
        # self._autostep_button['text'] = 'Autostep'
        self._autostep = 0

    # Make sure to stop auto-stepping if we get any user input.
    def step(self, *e):
        self._autostep = 0
        self._step()

    def match(self, *e):
        self._autostep = 0
        self._match()

    def expand(self, *e):
        self._autostep = 0
        self._expand()

    def backtrack(self, *e):
        self._autostep = 0
        self._backtrack()

    def _step(self):
        if self._animating_lock:
            return

        # Try expanding, matching, and backtracking (in that order)
        if self._expand():
            pass
        elif self._parser.untried_match() and self._match():
            pass
        elif self._backtrack():
            pass
        else:
            self._lastoper1["text"] = "Finished"
            self._lastoper2["text"] = ""
            self._autostep = 0

        # Check if we just completed a parse.
        if self._parser.currently_complete():
            self._autostep = 0
            self._lastoper2["text"] += "    [COMPLETE PARSE]"

    def _expand(self, *e):
        if self._animating_lock:
            return
        old_frontier = self._parser.frontier()
        rv = self._parser.expand()
        if rv is not None:
            self._lastoper1["text"] = "Expand:"
            self._lastoper2["text"] = rv
            self._prodlist.selection_clear(0, "end")
            index = self._productions.index(rv)
            self._prodlist.selection_set(index)
            self._animate_expand(old_frontier[0])
            return True
        else:
            self._lastoper1["text"] = "Expand:"
            self._lastoper2["text"] = "(all expansions tried)"
            return False

    def _match(self, *e):
        if self._animating_lock:
            return
        old_frontier = self._parser.frontier()
        rv = self._parser.match()
        if rv is not None:
            self._lastoper1["text"] = "Match:"
            self._lastoper2["text"] = rv
            self._animate_match(old_frontier[0])
            return True
        else:
            self._lastoper1["text"] = "Match:"
            self._lastoper2["text"] = "(failed)"
            return False

    def _backtrack(self, *e):
        if self._animating_lock:
            return
        if self._parser.backtrack():
            elt = self._parser.tree()
            for i in self._parser.frontier()[0]:
                elt = elt[i]
            self._lastoper1["text"] = "Backtrack"
            self._lastoper2["text"] = ""
            if isinstance(elt, Tree):
                self._animate_backtrack(self._parser.frontier()[0])
            else:
                self._animate_match_backtrack(self._parser.frontier()[0])
            return True
        else:
            self._autostep = 0
            self._lastoper1["text"] = "Finished"
            self._lastoper2["text"] = ""
            return False

    def about(self, *e):
        ABOUT = (
            "NLTK Recursive Descent Parser Application\n" + "Written by Edward Loper"
        )
        TITLE = "About: Recursive Descent Parser Application"
        try:
            from tkinter.messagebox import Message

            Message(message=ABOUT, title=TITLE).show()
        except:
            ShowText(self._top, TITLE, ABOUT)

    def help(self, *e):
        self._autostep = 0
        # The default font's not very legible; try using 'fixed' instead.
        try:
            ShowText(
                self._top,
                "Help: Recursive Descent Parser Application",
                (__doc__ or "").strip(),
                width=75,
                font="fixed",
            )
        except:
            ShowText(
                self._top,
                "Help: Recursive Descent Parser Application",
                (__doc__ or "").strip(),
                width=75,
            )

    def postscript(self, *e):
        self._autostep = 0
        self._cframe.print_to_file()

    def mainloop(self, *args, **kwargs):
        """
        Enter the Tkinter mainloop.  This function must be called if
        this demo is created from a non-interactive program (e.g.
        from a secript); otherwise, the demo will close as soon as
        the script completes.
        """
        if in_idle():
            return
        self._top.mainloop(*args, **kwargs)

    def resize(self, size=None):
        if size is not None:
            self._size.set(size)
        size = self._size.get()
        self._font.configure(size=-(abs(size)))
        self._boldfont.configure(size=-(abs(size)))
        self._sysfont.configure(size=-(abs(size)))
        self._bigfont.configure(size=-(abs(size + 2)))
        self._redraw()

    #########################################
    ##  Expand Production Selection
    #########################################

    def _toggle_grammar(self, *e):
        if self._show_grammar.get():
            self._prodframe.pack(
                fill="both", side="left", padx=2, after=self._feedbackframe
            )
            self._lastoper1["text"] = "Show Grammar"
        else:
            self._prodframe.pack_forget()
            self._lastoper1["text"] = "Hide Grammar"
        self._lastoper2["text"] = ""

    #     def toggle_grammar(self, *e):
    #         self._show_grammar = not self._show_grammar
    #         if self._show_grammar:
    #             self._prodframe.pack(fill='both', expand='y', side='left',
    #                                  after=self._feedbackframe)
    #             self._lastoper1['text'] = 'Show Grammar'
    #         else:
    #             self._prodframe.pack_forget()
    #             self._lastoper1['text'] = 'Hide Grammar'
    #         self._lastoper2['text'] = ''

    def _prodlist_select(self, event):
        selection = self._prodlist.curselection()
        if len(selection) != 1:
            return
        index = int(selection[0])
        old_frontier = self._parser.frontier()
        production = self._parser.expand(self._productions[index])

        if production:
            self._lastoper1["text"] = "Expand:"
            self._lastoper2["text"] = production
            self._prodlist.selection_clear(0, "end")
            self._prodlist.selection_set(index)
            self._animate_expand(old_frontier[0])
        else:
            # Reset the production selections.
            self._prodlist.selection_clear(0, "end")
            for prod in self._parser.expandable_productions():
                index = self._productions.index(prod)
                self._prodlist.selection_set(index)

    #########################################
    ##  Animation
    #########################################

    def _animate_expand(self, treeloc):
        oldwidget = self._get(self._tree, treeloc)
        oldtree = oldwidget.parent()
        top = not isinstance(oldtree.parent(), TreeSegmentWidget)

        tree = self._parser.tree()
        for i in treeloc:
            tree = tree[i]

        widget = tree_to_treesegment(
            self._canvas,
            tree,
            node_font=self._boldfont,
            leaf_color="white",
            tree_width=2,
            tree_color="white",
            node_color="white",
            leaf_font=self._font,
        )
        widget.label()["color"] = "#20a050"

        (oldx, oldy) = oldtree.label().bbox()[:2]
        (newx, newy) = widget.label().bbox()[:2]
        widget.move(oldx - newx, oldy - newy)

        if top:
            self._cframe.add_widget(widget, 0, 5)
            widget.move(30 - widget.label().bbox()[0], 0)
            self._tree = widget
        else:
            oldtree.parent().replace_child(oldtree, widget)

        # Move the children over so they don't overlap.
        # Line the children up in a strange way.
        if widget.subtrees():
            dx = (
                oldx
                + widget.label().width() / 2
                - widget.subtrees()[0].bbox()[0] / 2
                - widget.subtrees()[0].bbox()[2] / 2
            )
            for subtree in widget.subtrees():
                subtree.move(dx, 0)

        self._makeroom(widget)

        if top:
            self._cframe.destroy_widget(oldtree)
        else:
            oldtree.destroy()

        colors = [
            "gray%d" % (10 * int(10 * x / self._animation_frames.get()))
            for x in range(self._animation_frames.get(), 0, -1)
        ]

        # Move the text string down, if necessary.
        dy = widget.bbox()[3] + 30 - self._canvas.coords(self._textline)[1]
        if dy > 0:
            for twidget in self._textwidgets:
                twidget.move(0, dy)
            self._canvas.move(self._textline, 0, dy)

        self._animate_expand_frame(widget, colors)

    def _makeroom(self, treeseg):
        """
        Make sure that no sibling tree bbox's overlap.
        """
        parent = treeseg.parent()
        if not isinstance(parent, TreeSegmentWidget):
            return

        index = parent.subtrees().index(treeseg)

        # Handle siblings to the right
        rsiblings = parent.subtrees()[index + 1 :]
        if rsiblings:
            dx = treeseg.bbox()[2] - rsiblings[0].bbox()[0] + 10
            for sibling in rsiblings:
                sibling.move(dx, 0)

        # Handle siblings to the left
        if index > 0:
            lsibling = parent.subtrees()[index - 1]
            dx = max(0, lsibling.bbox()[2] - treeseg.bbox()[0] + 10)
            treeseg.move(dx, 0)

        # Keep working up the tree.
        self._makeroom(parent)

    def _animate_expand_frame(self, widget, colors):
        if len(colors) > 0:
            self._animating_lock = 1
            widget["color"] = colors[0]
            for subtree in widget.subtrees():
                if isinstance(subtree, TreeSegmentWidget):
                    subtree.label()["color"] = colors[0]
                else:
                    subtree["color"] = colors[0]
            self._top.after(50, self._animate_expand_frame, widget, colors[1:])
        else:
            widget["color"] = "black"
            for subtree in widget.subtrees():
                if isinstance(subtree, TreeSegmentWidget):
                    subtree.label()["color"] = "black"
                else:
                    subtree["color"] = "black"
            self._redraw_quick()
            widget.label()["color"] = "black"
            self._animating_lock = 0
            if self._autostep:
                self._step()

    def _animate_backtrack(self, treeloc):
        # Flash red first, if we're animating.
        if self._animation_frames.get() == 0:
            colors = []
        else:
            colors = ["#a00000", "#000000", "#a00000"]
        colors += [
            "gray%d" % (10 * int(10 * x / (self._animation_frames.get())))
            for x in range(1, self._animation_frames.get() + 1)
        ]

        widgets = [self._get(self._tree, treeloc).parent()]
        for subtree in widgets[0].subtrees():
            if isinstance(subtree, TreeSegmentWidget):
                widgets.append(subtree.label())
            else:
                widgets.append(subtree)

        self._animate_backtrack_frame(widgets, colors)

    def _animate_backtrack_frame(self, widgets, colors):
        if len(colors) > 0:
            self._animating_lock = 1
            for widget in widgets:
                widget["color"] = colors[0]
            self._top.after(50, self._animate_backtrack_frame, widgets, colors[1:])
        else:
            for widget in widgets[0].subtrees():
                widgets[0].remove_child(widget)
                widget.destroy()
            self._redraw_quick()
            self._animating_lock = 0
            if self._autostep:
                self._step()

    def _animate_match_backtrack(self, treeloc):
        widget = self._get(self._tree, treeloc)
        node = widget.parent().label()
        dy = (node.bbox()[3] - widget.bbox()[1] + 14) / max(
            1, self._animation_frames.get()
        )
        self._animate_match_backtrack_frame(self._animation_frames.get(), widget, dy)

    def _animate_match(self, treeloc):
        widget = self._get(self._tree, treeloc)

        dy = (self._textwidgets[0].bbox()[1] - widget.bbox()[3] - 10.0) / max(
            1, self._animation_frames.get()
        )
        self._animate_match_frame(self._animation_frames.get(), widget, dy)

    def _animate_match_frame(self, frame, widget, dy):
        if frame > 0:
            self._animating_lock = 1
            widget.move(0, dy)
            self._top.after(10, self._animate_match_frame, frame - 1, widget, dy)
        else:
            widget["color"] = "#006040"
            self._redraw_quick()
            self._animating_lock = 0
            if self._autostep:
                self._step()

    def _animate_match_backtrack_frame(self, frame, widget, dy):
        if frame > 0:
            self._animating_lock = 1
            widget.move(0, dy)
            self._top.after(
                10, self._animate_match_backtrack_frame, frame - 1, widget, dy
            )
        else:
            widget.parent().remove_child(widget)
            widget.destroy()
            self._animating_lock = 0
            if self._autostep:
                self._step()

    def edit_grammar(self, *e):
        CFGEditor(self._top, self._parser.grammar(), self.set_grammar)

    def set_grammar(self, grammar):
        self._parser.set_grammar(grammar)
        self._productions = list(grammar.productions())
        self._prodlist.delete(0, "end")
        for production in self._productions:
            self._prodlist.insert("end", (" %s" % production))

    def edit_sentence(self, *e):
        sentence = " ".join(self._sent)
        title = "Edit Text"
        instr = "Enter a new sentence to parse."
        EntryDialog(self._top, sentence, instr, self.set_sentence, title)

    def set_sentence(self, sentence):
        self._sent = sentence.split()  # [XX] use tagged?
        self.reset()


def app():
    """
    Create a recursive descent parser demo, using a simple grammar and
    text.
    """
    from nltk.grammar import CFG

    grammar = CFG.fromstring(
        """
    # Grammatical productions.
        S -> NP VP
        NP -> Det N PP | Det N
        VP -> V NP PP | V NP | V
        PP -> P NP
    # Lexical productions.
        NP -> 'I'
        Det -> 'the' | 'a'
        N -> 'man' | 'park' | 'dog' | 'telescope'
        V -> 'ate' | 'saw'
        P -> 'in' | 'under' | 'with'
    """
    )

    sent = "the dog saw a man in the park".split()

    RecursiveDescentApp(grammar, sent).mainloop()


if __name__ == "__main__":
    app()

__all__ = ["app"]
