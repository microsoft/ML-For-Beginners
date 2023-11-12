# Natural Language Toolkit: Shift-Reduce Parser Application
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A graphical tool for exploring the shift-reduce parser.

The shift-reduce parser maintains a stack, which records the structure
of the portion of the text that has been parsed.  The stack is
initially empty.  Its contents are shown on the left side of the main
canvas.

On the right side of the main canvas is the remaining text.  This is
the portion of the text which has not yet been considered by the
parser.

The parser builds up a tree structure for the text using two
operations:

  - "shift" moves the first token from the remaining text to the top
    of the stack.  In the demo, the top of the stack is its right-hand
    side.
  - "reduce" uses a grammar production to combine the rightmost stack
    elements into a single tree token.

You can control the parser's operation by using the "shift" and
"reduce" buttons; or you can use the "step" button to let the parser
automatically decide which operation to apply.  The parser uses the
following rules to decide which operation to apply:

  - Only shift if no reductions are available.
  - If multiple reductions are available, then apply the reduction
    whose CFG production is listed earliest in the grammar.

The "reduce" button applies the reduction whose CFG production is
listed earliest in the grammar.  There are two ways to manually choose
which reduction to apply:

  - Click on a CFG production from the list of available reductions,
    on the left side of the main window.  The reduction based on that
    production will be applied to the top of the stack.
  - Click on one of the stack elements.  A popup window will appear,
    containing all available reductions.  Select one, and it will be
    applied to the top of the stack.

Note that reductions can only be applied to the top of the stack.

Keyboard Shortcuts::
      [Space]\t Perform the next shift or reduce operation
      [s]\t Perform a shift operation
      [r]\t Perform a reduction operation
      [Ctrl-z]\t Undo most recent operation
      [Delete]\t Reset the parser
      [g]\t Show/hide available production list
      [Ctrl-a]\t Toggle animations
      [h]\t Help
      [Ctrl-p]\t Print
      [q]\t Quit

"""

from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
from tkinter.font import Font

from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import CanvasFrame, EntryDialog, ShowText, TextWidget
from nltk.parse import SteppingShiftReduceParser
from nltk.tree import Tree
from nltk.util import in_idle

"""
Possible future improvements:
  - button/window to change and/or select text.  Just pop up a window
    with an entry, and let them modify the text; and then retokenize
    it?  Maybe give a warning if it contains tokens whose types are
    not in the grammar.
  - button/window to change and/or select grammar.  Select from
    several alternative grammars?  Or actually change the grammar?  If
    the later, then I'd want to define nltk.draw.cfg, which would be
    responsible for that.
"""


class ShiftReduceApp:
    """
    A graphical tool for exploring the shift-reduce parser.  The tool
    displays the parser's stack and the remaining text, and allows the
    user to control the parser's operation.  In particular, the user
    can shift tokens onto the stack, and can perform reductions on the
    top elements of the stack.  A "step" button simply steps through
    the parsing process, performing the operations that
    ``nltk.parse.ShiftReduceParser`` would use.
    """

    def __init__(self, grammar, sent, trace=0):
        self._sent = sent
        self._parser = SteppingShiftReduceParser(grammar, trace)

        # Set up the main window.
        self._top = Tk()
        self._top.title("Shift Reduce Parser Application")

        # Animations.  animating_lock is a lock to prevent the demo
        # from performing new operations while it's animating.
        self._animating_lock = 0
        self._animate = IntVar(self._top)
        self._animate.set(10)  # = medium

        # The user can hide the grammar.
        self._show_grammar = IntVar(self._top)
        self._show_grammar.set(1)

        # Initialize fonts.
        self._init_fonts(self._top)

        # Set up key bindings.
        self._init_bindings()

        # Create the basic frames.
        self._init_menubar(self._top)
        self._init_buttons(self._top)
        self._init_feedback(self._top)
        self._init_grammar(self._top)
        self._init_canvas(self._top)

        # A popup menu for reducing.
        self._reduce_menu = Menu(self._canvas, tearoff=0)

        # Reset the demo, and set the feedback frame to empty.
        self.reset()
        self._lastoper1["text"] = ""

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

    def _init_grammar(self, parent):
        # Grammar view.
        self._prodframe = listframe = Frame(parent)
        self._prodframe.pack(fill="both", side="left", padx=2)
        self._prodlist_label = Label(
            self._prodframe, font=self._boldfont, text="Available Reductions"
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
            self._prodlist.insert("end", (" %s" % production))
        self._prodlist.config(height=min(len(self._productions), 25))

        # Add a scrollbar if there are more than 25 productions.
        if 1:  # len(self._productions) > 25:
            listscroll = Scrollbar(self._prodframe, orient="vertical")
            self._prodlist.config(yscrollcommand=listscroll.set)
            listscroll.config(command=self._prodlist.yview)
            listscroll.pack(side="left", fill="y")

        # If they select a production, apply it.
        self._prodlist.bind("<<ListboxSelect>>", self._prodlist_select)

        # When they hover over a production, highlight it.
        self._hover = -1
        self._prodlist.bind("<Motion>", self._highlight_hover)
        self._prodlist.bind("<Leave>", self._clear_hover)

    def _init_bindings(self):
        # Quit
        self._top.bind("<Control-q>", self.destroy)
        self._top.bind("<Control-x>", self.destroy)
        self._top.bind("<Alt-q>", self.destroy)
        self._top.bind("<Alt-x>", self.destroy)

        # Ops (step, shift, reduce, undo)
        self._top.bind("<space>", self.step)
        self._top.bind("<s>", self.shift)
        self._top.bind("<Alt-s>", self.shift)
        self._top.bind("<Control-s>", self.shift)
        self._top.bind("<r>", self.reduce)
        self._top.bind("<Alt-r>", self.reduce)
        self._top.bind("<Control-r>", self.reduce)
        self._top.bind("<Delete>", self.reset)
        self._top.bind("<u>", self.undo)
        self._top.bind("<Alt-u>", self.undo)
        self._top.bind("<Control-u>", self.undo)
        self._top.bind("<Control-z>", self.undo)
        self._top.bind("<BackSpace>", self.undo)

        # Misc
        self._top.bind("<Control-p>", self.postscript)
        self._top.bind("<Control-h>", self.help)
        self._top.bind("<F1>", self.help)
        self._top.bind("<Control-g>", self.edit_grammar)
        self._top.bind("<Control-t>", self.edit_sentence)

        # Animation speed control
        self._top.bind("-", lambda e, a=self._animate: a.set(20))
        self._top.bind("=", lambda e, a=self._animate: a.set(10))
        self._top.bind("+", lambda e, a=self._animate: a.set(4))

    def _init_buttons(self, parent):
        # Set up the frames.
        self._buttonframe = buttonframe = Frame(parent)
        buttonframe.pack(fill="none", side="bottom")
        Button(
            buttonframe,
            text="Step",
            background="#90c0d0",
            foreground="black",
            command=self.step,
        ).pack(side="left")
        Button(
            buttonframe,
            text="Shift",
            underline=0,
            background="#90f090",
            foreground="black",
            command=self.shift,
        ).pack(side="left")
        Button(
            buttonframe,
            text="Reduce",
            underline=0,
            background="#90f090",
            foreground="black",
            command=self.reduce,
        ).pack(side="left")
        Button(
            buttonframe,
            text="Undo",
            underline=0,
            background="#f0a0a0",
            foreground="black",
            command=self.undo,
        ).pack(side="left")

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
            label="Shift", underline=0, command=self.shift, accelerator="Ctrl-s"
        )
        rulemenu.add_command(
            label="Reduce", underline=0, command=self.reduce, accelerator="Ctrl-r"
        )
        rulemenu.add_separator()
        rulemenu.add_command(
            label="Undo", underline=0, command=self.undo, accelerator="Ctrl-u"
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
            label="No Animation", underline=0, variable=self._animate, value=0
        )
        animatemenu.add_radiobutton(
            label="Slow Animation",
            underline=0,
            variable=self._animate,
            value=20,
            accelerator="-",
        )
        animatemenu.add_radiobutton(
            label="Normal Animation",
            underline=0,
            variable=self._animate,
            value=10,
            accelerator="=",
        )
        animatemenu.add_radiobutton(
            label="Fast Animation",
            underline=0,
            variable=self._animate,
            value=4,
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
            width=525,
            closeenough=10,
            border=2,
            relief="sunken",
        )
        self._cframe.pack(expand=1, fill="both", side="top", pady=2)
        canvas = self._canvas = self._cframe.canvas()

        self._stackwidgets = []
        self._rtextwidgets = []
        self._titlebar = canvas.create_rectangle(
            0, 0, 0, 0, fill="#c0f0f0", outline="black"
        )
        self._exprline = canvas.create_line(0, 0, 0, 0, dash=".")
        self._stacktop = canvas.create_line(0, 0, 0, 0, fill="#408080")
        size = self._size.get() + 4
        self._stacklabel = TextWidget(
            canvas, "Stack", color="#004040", font=self._boldfont
        )
        self._rtextlabel = TextWidget(
            canvas, "Remaining Text", color="#004040", font=self._boldfont
        )
        self._cframe.add_widget(self._stacklabel)
        self._cframe.add_widget(self._rtextlabel)

    #########################################
    ##  Main draw procedure
    #########################################

    def _redraw(self):
        scrollregion = self._canvas["scrollregion"].split()
        (cx1, cy1, cx2, cy2) = (int(c) for c in scrollregion)

        # Delete the old stack & rtext widgets.
        for stackwidget in self._stackwidgets:
            self._cframe.destroy_widget(stackwidget)
        self._stackwidgets = []
        for rtextwidget in self._rtextwidgets:
            self._cframe.destroy_widget(rtextwidget)
        self._rtextwidgets = []

        # Position the titlebar & exprline
        (x1, y1, x2, y2) = self._stacklabel.bbox()
        y = y2 - y1 + 10
        self._canvas.coords(self._titlebar, -5000, 0, 5000, y - 4)
        self._canvas.coords(self._exprline, 0, y * 2 - 10, 5000, y * 2 - 10)

        # Position the titlebar labels..
        (x1, y1, x2, y2) = self._stacklabel.bbox()
        self._stacklabel.move(5 - x1, 3 - y1)
        (x1, y1, x2, y2) = self._rtextlabel.bbox()
        self._rtextlabel.move(cx2 - x2 - 5, 3 - y1)

        # Draw the stack.
        stackx = 5
        for tok in self._parser.stack():
            if isinstance(tok, Tree):
                attribs = {
                    "tree_color": "#4080a0",
                    "tree_width": 2,
                    "node_font": self._boldfont,
                    "node_color": "#006060",
                    "leaf_color": "#006060",
                    "leaf_font": self._font,
                }
                widget = tree_to_treesegment(self._canvas, tok, **attribs)
                widget.label()["color"] = "#000000"
            else:
                widget = TextWidget(self._canvas, tok, color="#000000", font=self._font)
            widget.bind_click(self._popup_reduce)
            self._stackwidgets.append(widget)
            self._cframe.add_widget(widget, stackx, y)
            stackx = widget.bbox()[2] + 10

        # Draw the remaining text.
        rtextwidth = 0
        for tok in self._parser.remaining_text():
            widget = TextWidget(self._canvas, tok, color="#000000", font=self._font)
            self._rtextwidgets.append(widget)
            self._cframe.add_widget(widget, rtextwidth, y)
            rtextwidth = widget.bbox()[2] + 4

        # Allow enough room to shift the next token (for animations)
        if len(self._rtextwidgets) > 0:
            stackx += self._rtextwidgets[0].width()

        # Move the remaining text to the correct location (keep it
        # right-justified, when possible); and move the remaining text
        # label, if necessary.
        stackx = max(stackx, self._stacklabel.width() + 25)
        rlabelwidth = self._rtextlabel.width() + 10
        if stackx >= cx2 - max(rtextwidth, rlabelwidth):
            cx2 = stackx + max(rtextwidth, rlabelwidth)
        for rtextwidget in self._rtextwidgets:
            rtextwidget.move(4 + cx2 - rtextwidth, 0)
        self._rtextlabel.move(cx2 - self._rtextlabel.bbox()[2] - 5, 0)

        midx = (stackx + cx2 - max(rtextwidth, rlabelwidth)) / 2
        self._canvas.coords(self._stacktop, midx, 0, midx, 5000)
        (x1, y1, x2, y2) = self._stacklabel.bbox()

        # Set up binding to allow them to shift a token by dragging it.
        if len(self._rtextwidgets) > 0:

            def drag_shift(widget, midx=midx, self=self):
                if widget.bbox()[0] < midx:
                    self.shift()
                else:
                    self._redraw()

            self._rtextwidgets[0].bind_drag(drag_shift)
            self._rtextwidgets[0].bind_click(self.shift)

        # Draw the stack top.
        self._highlight_productions()

    def _draw_stack_top(self, widget):
        # hack..
        midx = widget.bbox()[2] + 50
        self._canvas.coords(self._stacktop, midx, 0, midx, 5000)

    def _highlight_productions(self):
        # Highlight the productions that can be reduced.
        self._prodlist.selection_clear(0, "end")
        for prod in self._parser.reducible_productions():
            index = self._productions.index(prod)
            self._prodlist.selection_set(index)

    #########################################
    ##  Button Callbacks
    #########################################

    def destroy(self, *e):
        if self._top is None:
            return
        self._top.destroy()
        self._top = None

    def reset(self, *e):
        self._parser.initialize(self._sent)
        self._lastoper1["text"] = "Reset App"
        self._lastoper2["text"] = ""
        self._redraw()

    def step(self, *e):
        if self.reduce():
            return True
        elif self.shift():
            return True
        else:
            if list(self._parser.parses()):
                self._lastoper1["text"] = "Finished:"
                self._lastoper2["text"] = "Success"
            else:
                self._lastoper1["text"] = "Finished:"
                self._lastoper2["text"] = "Failure"

    def shift(self, *e):
        if self._animating_lock:
            return
        if self._parser.shift():
            tok = self._parser.stack()[-1]
            self._lastoper1["text"] = "Shift:"
            self._lastoper2["text"] = "%r" % tok
            if self._animate.get():
                self._animate_shift()
            else:
                self._redraw()
            return True
        return False

    def reduce(self, *e):
        if self._animating_lock:
            return
        production = self._parser.reduce()
        if production:
            self._lastoper1["text"] = "Reduce:"
            self._lastoper2["text"] = "%s" % production
            if self._animate.get():
                self._animate_reduce()
            else:
                self._redraw()
        return production

    def undo(self, *e):
        if self._animating_lock:
            return
        if self._parser.undo():
            self._redraw()

    def postscript(self, *e):
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

    #########################################
    ##  Menubar callbacks
    #########################################

    def resize(self, size=None):
        if size is not None:
            self._size.set(size)
        size = self._size.get()
        self._font.configure(size=-(abs(size)))
        self._boldfont.configure(size=-(abs(size)))
        self._sysfont.configure(size=-(abs(size)))

        # self._stacklabel['font'] = ('helvetica', -size-4, 'bold')
        # self._rtextlabel['font'] = ('helvetica', -size-4, 'bold')
        # self._lastoper_label['font'] = ('helvetica', -size)
        # self._lastoper1['font'] = ('helvetica', -size)
        # self._lastoper2['font'] = ('helvetica', -size)
        # self._prodlist['font'] = ('helvetica', -size)
        # self._prodlist_label['font'] = ('helvetica', -size-2, 'bold')
        self._redraw()

    def help(self, *e):
        # The default font's not very legible; try using 'fixed' instead.
        try:
            ShowText(
                self._top,
                "Help: Shift-Reduce Parser Application",
                (__doc__ or "").strip(),
                width=75,
                font="fixed",
            )
        except:
            ShowText(
                self._top,
                "Help: Shift-Reduce Parser Application",
                (__doc__ or "").strip(),
                width=75,
            )

    def about(self, *e):
        ABOUT = "NLTK Shift-Reduce Parser Application\n" + "Written by Edward Loper"
        TITLE = "About: Shift-Reduce Parser Application"
        try:
            from tkinter.messagebox import Message

            Message(message=ABOUT, title=TITLE).show()
        except:
            ShowText(self._top, TITLE, ABOUT)

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

    def set_sentence(self, sent):
        self._sent = sent.split()  # [XX] use tagged?
        self.reset()

    #########################################
    ##  Reduce Production Selection
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

    def _prodlist_select(self, event):
        selection = self._prodlist.curselection()
        if len(selection) != 1:
            return
        index = int(selection[0])
        production = self._parser.reduce(self._productions[index])
        if production:
            self._lastoper1["text"] = "Reduce:"
            self._lastoper2["text"] = "%s" % production
            if self._animate.get():
                self._animate_reduce()
            else:
                self._redraw()
        else:
            # Reset the production selections.
            self._prodlist.selection_clear(0, "end")
            for prod in self._parser.reducible_productions():
                index = self._productions.index(prod)
                self._prodlist.selection_set(index)

    def _popup_reduce(self, widget):
        # Remove old commands.
        productions = self._parser.reducible_productions()
        if len(productions) == 0:
            return

        self._reduce_menu.delete(0, "end")
        for production in productions:
            self._reduce_menu.add_command(label=str(production), command=self.reduce)
        self._reduce_menu.post(
            self._canvas.winfo_pointerx(), self._canvas.winfo_pointery()
        )

    #########################################
    ##  Animations
    #########################################

    def _animate_shift(self):
        # What widget are we shifting?
        widget = self._rtextwidgets[0]

        # Where are we shifting from & to?
        right = widget.bbox()[0]
        if len(self._stackwidgets) == 0:
            left = 5
        else:
            left = self._stackwidgets[-1].bbox()[2] + 10

        # Start animating.
        dt = self._animate.get()
        dx = (left - right) * 1.0 / dt
        self._animate_shift_frame(dt, widget, dx)

    def _animate_shift_frame(self, frame, widget, dx):
        if frame > 0:
            self._animating_lock = 1
            widget.move(dx, 0)
            self._top.after(10, self._animate_shift_frame, frame - 1, widget, dx)
        else:
            # but: stacktop??

            # Shift the widget to the stack.
            del self._rtextwidgets[0]
            self._stackwidgets.append(widget)
            self._animating_lock = 0

            # Display the available productions.
            self._draw_stack_top(widget)
            self._highlight_productions()

    def _animate_reduce(self):
        # What widgets are we shifting?
        numwidgets = len(self._parser.stack()[-1])  # number of children
        widgets = self._stackwidgets[-numwidgets:]

        # How far are we moving?
        if isinstance(widgets[0], TreeSegmentWidget):
            ydist = 15 + widgets[0].label().height()
        else:
            ydist = 15 + widgets[0].height()

        # Start animating.
        dt = self._animate.get()
        dy = ydist * 2.0 / dt
        self._animate_reduce_frame(dt / 2, widgets, dy)

    def _animate_reduce_frame(self, frame, widgets, dy):
        if frame > 0:
            self._animating_lock = 1
            for widget in widgets:
                widget.move(0, dy)
            self._top.after(10, self._animate_reduce_frame, frame - 1, widgets, dy)
        else:
            del self._stackwidgets[-len(widgets) :]
            for widget in widgets:
                self._cframe.remove_widget(widget)
            tok = self._parser.stack()[-1]
            if not isinstance(tok, Tree):
                raise ValueError()
            label = TextWidget(
                self._canvas, str(tok.label()), color="#006060", font=self._boldfont
            )
            widget = TreeSegmentWidget(self._canvas, label, widgets, width=2)
            (x1, y1, x2, y2) = self._stacklabel.bbox()
            y = y2 - y1 + 10
            if not self._stackwidgets:
                x = 5
            else:
                x = self._stackwidgets[-1].bbox()[2] + 10
            self._cframe.add_widget(widget, x, y)
            self._stackwidgets.append(widget)

            # Display the available productions.
            self._draw_stack_top(widget)
            self._highlight_productions()

            #             # Delete the old widgets..
            #             del self._stackwidgets[-len(widgets):]
            #             for widget in widgets:
            #                 self._cframe.destroy_widget(widget)
            #
            #             # Make a new one.
            #             tok = self._parser.stack()[-1]
            #             if isinstance(tok, Tree):
            #                 attribs = {'tree_color': '#4080a0', 'tree_width': 2,
            #                            'node_font': bold, 'node_color': '#006060',
            #                            'leaf_color': '#006060', 'leaf_font':self._font}
            #                 widget = tree_to_treesegment(self._canvas, tok.type(),
            #                                              **attribs)
            #                 widget.node()['color'] = '#000000'
            #             else:
            #                 widget = TextWidget(self._canvas, tok.type(),
            #                                     color='#000000', font=self._font)
            #             widget.bind_click(self._popup_reduce)
            #             (x1, y1, x2, y2) = self._stacklabel.bbox()
            #             y = y2-y1+10
            #             if not self._stackwidgets: x = 5
            #             else: x = self._stackwidgets[-1].bbox()[2] + 10
            #             self._cframe.add_widget(widget, x, y)
            #             self._stackwidgets.append(widget)

            # self._redraw()
            self._animating_lock = 0

    #########################################
    ##  Hovering.
    #########################################

    def _highlight_hover(self, event):
        # What production are we hovering over?
        index = self._prodlist.nearest(event.y)
        if self._hover == index:
            return

        # Clear any previous hover highlighting.
        self._clear_hover()

        # If the production corresponds to an available reduction,
        # highlight the stack.
        selection = [int(s) for s in self._prodlist.curselection()]
        if index in selection:
            rhslen = len(self._productions[index].rhs())
            for stackwidget in self._stackwidgets[-rhslen:]:
                if isinstance(stackwidget, TreeSegmentWidget):
                    stackwidget.label()["color"] = "#00a000"
                else:
                    stackwidget["color"] = "#00a000"

        # Remember what production we're hovering over.
        self._hover = index

    def _clear_hover(self, *event):
        # Clear any previous hover highlighting.
        if self._hover == -1:
            return
        self._hover = -1
        for stackwidget in self._stackwidgets:
            if isinstance(stackwidget, TreeSegmentWidget):
                stackwidget.label()["color"] = "black"
            else:
                stackwidget["color"] = "black"


def app():
    """
    Create a shift reduce parser app, using a simple grammar and
    text.
    """

    from nltk.grammar import CFG, Nonterminal, Production

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

    # tokenize the sentence
    sent = "my dog saw a man in the park with a statue".split()

    ShiftReduceApp(grammar, sent).mainloop()


if __name__ == "__main__":
    app()

__all__ = ["app"]
