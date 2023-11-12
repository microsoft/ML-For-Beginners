# Natural Language Toolkit: GUI Demo for Glue Semantics with Discourse
#                           Representation Theory (DRT) as meaning language
#
# Author: Dan Garrette <dhgarrette@gmail.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

try:
    from tkinter import Button, Frame, IntVar, Label, Listbox, Menu, Scrollbar, Tk
    from tkinter.font import Font

    from nltk.draw.util import CanvasFrame, ShowText

except ImportError:
    """Ignore ImportError because tkinter might not be available."""

from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle


class DrtGlueDemo:
    def __init__(self, examples):
        # Set up the main window.
        self._top = Tk()
        self._top.title("DRT Glue Demo")

        # Set up key bindings.
        self._init_bindings()

        # Initialize the fonts.self._error = None
        self._init_fonts(self._top)

        self._examples = examples
        self._readingCache = [None for example in examples]

        # The user can hide the grammar.
        self._show_grammar = IntVar(self._top)
        self._show_grammar.set(1)

        # Set the data to None
        self._curExample = -1
        self._readings = []
        self._drs = None
        self._drsWidget = None
        self._error = None

        self._init_glue()

        # Create the basic frames.
        self._init_menubar(self._top)
        self._init_buttons(self._top)
        self._init_exampleListbox(self._top)
        self._init_readingListbox(self._top)
        self._init_canvas(self._top)

        # Resize callback
        self._canvas.bind("<Configure>", self._configure)

    #########################################
    ##  Initialization Helpers
    #########################################

    def _init_glue(self):
        tagger = RegexpTagger(
            [
                ("^(David|Mary|John)$", "NNP"),
                (
                    "^(walks|sees|eats|chases|believes|gives|sleeps|chases|persuades|tries|seems|leaves)$",
                    "VB",
                ),
                ("^(go|order|vanish|find|approach)$", "VB"),
                ("^(a)$", "ex_quant"),
                ("^(every)$", "univ_quant"),
                ("^(sandwich|man|dog|pizza|unicorn|cat|senator)$", "NN"),
                ("^(big|gray|former)$", "JJ"),
                ("^(him|himself)$", "PRP"),
            ]
        )

        depparser = MaltParser(tagger=tagger)
        self._glue = DrtGlue(depparser=depparser, remove_duplicates=False)

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

    def _init_exampleListbox(self, parent):
        self._exampleFrame = listframe = Frame(parent)
        self._exampleFrame.pack(fill="both", side="left", padx=2)
        self._exampleList_label = Label(
            self._exampleFrame, font=self._boldfont, text="Examples"
        )
        self._exampleList_label.pack()
        self._exampleList = Listbox(
            self._exampleFrame,
            selectmode="single",
            relief="groove",
            background="white",
            foreground="#909090",
            font=self._font,
            selectforeground="#004040",
            selectbackground="#c0f0c0",
        )

        self._exampleList.pack(side="right", fill="both", expand=1)

        for example in self._examples:
            self._exampleList.insert("end", ("  %s" % example))
        self._exampleList.config(height=min(len(self._examples), 25), width=40)

        # Add a scrollbar if there are more than 25 examples.
        if len(self._examples) > 25:
            listscroll = Scrollbar(self._exampleFrame, orient="vertical")
            self._exampleList.config(yscrollcommand=listscroll.set)
            listscroll.config(command=self._exampleList.yview)
            listscroll.pack(side="left", fill="y")

        # If they select a example, apply it.
        self._exampleList.bind("<<ListboxSelect>>", self._exampleList_select)

    def _init_readingListbox(self, parent):
        self._readingFrame = listframe = Frame(parent)
        self._readingFrame.pack(fill="both", side="left", padx=2)
        self._readingList_label = Label(
            self._readingFrame, font=self._boldfont, text="Readings"
        )
        self._readingList_label.pack()
        self._readingList = Listbox(
            self._readingFrame,
            selectmode="single",
            relief="groove",
            background="white",
            foreground="#909090",
            font=self._font,
            selectforeground="#004040",
            selectbackground="#c0f0c0",
        )

        self._readingList.pack(side="right", fill="both", expand=1)

        # Add a scrollbar if there are more than 25 examples.
        listscroll = Scrollbar(self._readingFrame, orient="vertical")
        self._readingList.config(yscrollcommand=listscroll.set)
        listscroll.config(command=self._readingList.yview)
        listscroll.pack(side="right", fill="y")

        self._populate_readingListbox()

    def _populate_readingListbox(self):
        # Populate the listbox with integers
        self._readingList.delete(0, "end")
        for i in range(len(self._readings)):
            self._readingList.insert("end", ("  %s" % (i + 1)))
        self._readingList.config(height=min(len(self._readings), 25), width=5)

        # If they select a example, apply it.
        self._readingList.bind("<<ListboxSelect>>", self._readingList_select)

    def _init_bindings(self):
        # Key bindings are a good thing.
        self._top.bind("<Control-q>", self.destroy)
        self._top.bind("<Control-x>", self.destroy)
        self._top.bind("<Escape>", self.destroy)
        self._top.bind("n", self.next)
        self._top.bind("<space>", self.next)
        self._top.bind("p", self.prev)
        self._top.bind("<BackSpace>", self.prev)

    def _init_buttons(self, parent):
        # Set up the frames.
        self._buttonframe = buttonframe = Frame(parent)
        buttonframe.pack(fill="none", side="bottom", padx=3, pady=2)
        Button(
            buttonframe,
            text="Prev",
            background="#90c0d0",
            foreground="black",
            command=self.prev,
        ).pack(side="left")
        Button(
            buttonframe,
            text="Next",
            background="#90c0d0",
            foreground="black",
            command=self.next,
        ).pack(side="left")

    def _configure(self, event):
        self._autostep = 0
        (x1, y1, x2, y2) = self._cframe.scrollregion()
        y2 = event.height - 6
        self._canvas["scrollregion"] = "%d %d %d %d" % (x1, y1, x2, y2)
        self._redraw()

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
            label="Exit", underline=1, command=self.destroy, accelerator="q"
        )
        menubar.add_cascade(label="File", underline=0, menu=filemenu)

        actionmenu = Menu(menubar, tearoff=0)
        actionmenu.add_command(
            label="Next", underline=0, command=self.next, accelerator="n, Space"
        )
        actionmenu.add_command(
            label="Previous", underline=0, command=self.prev, accelerator="p, Backspace"
        )
        menubar.add_cascade(label="Action", underline=0, menu=actionmenu)

        optionmenu = Menu(menubar, tearoff=0)
        optionmenu.add_checkbutton(
            label="Remove Duplicates",
            underline=0,
            variable=self._glue.remove_duplicates,
            command=self._toggle_remove_duplicates,
            accelerator="r",
        )
        menubar.add_cascade(label="Options", underline=0, menu=optionmenu)

        viewmenu = Menu(menubar, tearoff=0)
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

        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", underline=0, command=self.about)
        menubar.add_cascade(label="Help", underline=0, menu=helpmenu)

        parent.config(menu=menubar)

    #########################################
    ##  Main draw procedure
    #########################################

    def _redraw(self):
        canvas = self._canvas

        # Delete the old DRS, widgets, etc.
        if self._drsWidget is not None:
            self._drsWidget.clear()

        if self._drs:
            self._drsWidget = DrsWidget(self._canvas, self._drs)
            self._drsWidget.draw()

        if self._error:
            self._drsWidget = DrsWidget(self._canvas, self._error)
            self._drsWidget.draw()

    #########################################
    ##  Button Callbacks
    #########################################

    def destroy(self, *e):
        self._autostep = 0
        if self._top is None:
            return
        self._top.destroy()
        self._top = None

    def prev(self, *e):
        selection = self._readingList.curselection()
        readingListSize = self._readingList.size()

        # there are readings
        if readingListSize > 0:
            # if one reading is currently selected
            if len(selection) == 1:
                index = int(selection[0])

                # if it's on (or before) the first item
                if index <= 0:
                    self._select_previous_example()
                else:
                    self._readingList_store_selection(index - 1)

            else:
                # select its first reading
                self._readingList_store_selection(readingListSize - 1)

        else:
            self._select_previous_example()

    def _select_previous_example(self):
        # if the current example is not the first example
        if self._curExample > 0:
            self._exampleList_store_selection(self._curExample - 1)
        else:
            # go to the last example
            self._exampleList_store_selection(len(self._examples) - 1)

    def next(self, *e):
        selection = self._readingList.curselection()
        readingListSize = self._readingList.size()

        # if there are readings
        if readingListSize > 0:
            # if one reading is currently selected
            if len(selection) == 1:
                index = int(selection[0])

                # if it's on (or past) the last item
                if index >= (readingListSize - 1):
                    self._select_next_example()
                else:
                    self._readingList_store_selection(index + 1)

            else:
                # select its first reading
                self._readingList_store_selection(0)

        else:
            self._select_next_example()

    def _select_next_example(self):
        # if the current example is not the last example
        if self._curExample < len(self._examples) - 1:
            self._exampleList_store_selection(self._curExample + 1)
        else:
            # go to the first example
            self._exampleList_store_selection(0)

    def about(self, *e):
        ABOUT = (
            "NLTK Discourse Representation Theory (DRT) Glue Semantics Demo\n"
            + "Written by Daniel H. Garrette"
        )
        TITLE = "About: NLTK DRT Glue Demo"
        try:
            from tkinter.messagebox import Message

            Message(message=ABOUT, title=TITLE).show()
        except:
            ShowText(self._top, TITLE, ABOUT)

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

    def _toggle_remove_duplicates(self):
        self._glue.remove_duplicates = not self._glue.remove_duplicates

        self._exampleList.selection_clear(0, "end")
        self._readings = []
        self._populate_readingListbox()
        self._readingCache = [None for ex in self._examples]
        self._curExample = -1
        self._error = None

        self._drs = None
        self._redraw()

    def _exampleList_select(self, event):
        selection = self._exampleList.curselection()
        if len(selection) != 1:
            return
        self._exampleList_store_selection(int(selection[0]))

    def _exampleList_store_selection(self, index):
        self._curExample = index
        example = self._examples[index]

        self._exampleList.selection_clear(0, "end")
        if example:
            cache = self._readingCache[index]
            if cache:
                if isinstance(cache, list):
                    self._readings = cache
                    self._error = None
                else:
                    self._readings = []
                    self._error = cache
            else:
                try:
                    self._readings = self._glue.parse_to_meaning(example)
                    self._error = None
                    self._readingCache[index] = self._readings
                except Exception as e:
                    self._readings = []
                    self._error = DrtVariableExpression(Variable("Error: " + str(e)))
                    self._readingCache[index] = self._error

                    # add a star to the end of the example
                    self._exampleList.delete(index)
                    self._exampleList.insert(index, ("  %s *" % example))
                    self._exampleList.config(
                        height=min(len(self._examples), 25), width=40
                    )

            self._populate_readingListbox()

            self._exampleList.selection_set(index)

            self._drs = None
            self._redraw()

    def _readingList_select(self, event):
        selection = self._readingList.curselection()
        if len(selection) != 1:
            return
        self._readingList_store_selection(int(selection[0]))

    def _readingList_store_selection(self, index):
        reading = self._readings[index]

        self._readingList.selection_clear(0, "end")
        if reading:
            self._readingList.selection_set(index)

            self._drs = reading.simplify().normalize().resolve_anaphora()

            self._redraw()


class DrsWidget:
    def __init__(self, canvas, drs, **attribs):
        self._drs = drs
        self._canvas = canvas
        canvas.font = Font(
            font=canvas.itemcget(canvas.create_text(0, 0, text=""), "font")
        )
        canvas._BUFFER = 3
        self.bbox = (0, 0, 0, 0)

    def draw(self):
        (right, bottom) = DrsDrawer(self._drs, canvas=self._canvas).draw()
        self.bbox = (0, 0, right + 1, bottom + 1)

    def clear(self):
        self._canvas.create_rectangle(self.bbox, fill="white", width="0")


def demo():
    examples = [
        "John walks",
        "David sees Mary",
        "David eats a sandwich",
        "every man chases a dog",
        #                'every man believes a dog yawns',
        #                'John gives David a sandwich',
        "John chases himself",
        #                'John persuades David to order a pizza',
        #                'John tries to go',
        #                'John tries to find a unicorn',
        #                'John seems to vanish',
        #                'a unicorn seems to approach',
        #                'every big cat leaves',
        #                'every gray cat leaves',
        #                'every big gray cat leaves',
        #                'a former senator leaves',
        #                'John likes a cat',
        #                'John likes every cat',
        #                'he walks',
        #                'John walks and he leaves'
    ]
    DrtGlueDemo(examples).mainloop()


if __name__ == "__main__":
    demo()
