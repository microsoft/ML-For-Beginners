# Natural Language Toolkit: Regexp Chunk Parser Application
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A graphical tool for exploring the regular expression based chunk
parser ``nltk.chunk.RegexpChunkParser``.
"""

# Todo: Add a way to select the development set from the menubar.  This
# might just need to be a selection box (conll vs treebank etc) plus
# configuration parameters to select what's being chunked (eg VP vs NP)
# and what part of the data is being used as the development set.

import random
import re
import textwrap
import time
from tkinter import (
    Button,
    Canvas,
    Checkbutton,
    Frame,
    IntVar,
    Label,
    Menu,
    Scrollbar,
    Text,
    Tk,
)
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font

from nltk.chunk import ChunkScore, RegexpChunkParser
from nltk.chunk.regexp import RegexpChunkRule
from nltk.corpus import conll2000, treebank_chunk
from nltk.draw.util import ShowText
from nltk.tree import Tree
from nltk.util import in_idle


class RegexpChunkApp:
    """
    A graphical tool for exploring the regular expression based chunk
    parser ``nltk.chunk.RegexpChunkParser``.

    See ``HELP`` for instructional text.
    """

    ##/////////////////////////////////////////////////////////////////
    ##  Help Text
    ##/////////////////////////////////////////////////////////////////

    #: A dictionary mapping from part of speech tags to descriptions,
    #: which is used in the help text.  (This should probably live with
    #: the conll and/or treebank corpus instead.)
    TAGSET = {
        "CC": "Coordinating conjunction",
        "PRP$": "Possessive pronoun",
        "CD": "Cardinal number",
        "RB": "Adverb",
        "DT": "Determiner",
        "RBR": "Adverb, comparative",
        "EX": "Existential there",
        "RBS": "Adverb, superlative",
        "FW": "Foreign word",
        "RP": "Particle",
        "JJ": "Adjective",
        "TO": "to",
        "JJR": "Adjective, comparative",
        "UH": "Interjection",
        "JJS": "Adjective, superlative",
        "VB": "Verb, base form",
        "LS": "List item marker",
        "VBD": "Verb, past tense",
        "MD": "Modal",
        "NNS": "Noun, plural",
        "NN": "Noun, singular or masps",
        "VBN": "Verb, past participle",
        "VBZ": "Verb,3rd ps. sing. present",
        "NNP": "Proper noun, singular",
        "NNPS": "Proper noun plural",
        "WDT": "wh-determiner",
        "PDT": "Predeterminer",
        "WP": "wh-pronoun",
        "POS": "Possessive ending",
        "WP$": "Possessive wh-pronoun",
        "PRP": "Personal pronoun",
        "WRB": "wh-adverb",
        "(": "open parenthesis",
        ")": "close parenthesis",
        "``": "open quote",
        ",": "comma",
        "''": "close quote",
        ".": "period",
        "#": "pound sign (currency marker)",
        "$": "dollar sign (currency marker)",
        "IN": "Preposition/subord. conjunction",
        "SYM": "Symbol (mathematical or scientific)",
        "VBG": "Verb, gerund/present participle",
        "VBP": "Verb, non-3rd ps. sing. present",
        ":": "colon",
    }

    #: Contents for the help box.  This is a list of tuples, one for
    #: each help page, where each tuple has four elements:
    #:   - A title (displayed as a tab)
    #:   - A string description of tabstops (see Tkinter.Text for details)
    #:   - The text contents for the help page.  You can use expressions
    #:     like <red>...</red> to colorize the text; see ``HELP_AUTOTAG``
    #:     for a list of tags you can use for colorizing.
    HELP = [
        (
            "Help",
            "20",
            "Welcome to the regular expression chunk-parser grammar editor.  "
            "You can use this editor to develop and test chunk parser grammars "
            "based on NLTK's RegexpChunkParser class.\n\n"
            # Help box.
            "Use this box ('Help') to learn more about the editor; click on the "
            "tabs for help on specific topics:"
            "<indent>\n"
            "Rules: grammar rule types\n"
            "Regexps: regular expression syntax\n"
            "Tags: part of speech tags\n</indent>\n"
            # Grammar.
            "Use the upper-left box ('Grammar') to edit your grammar.  "
            "Each line of your grammar specifies a single 'rule', "
            "which performs an action such as creating a chunk or merging "
            "two chunks.\n\n"
            # Dev set.
            "The lower-left box ('Development Set') runs your grammar on the "
            "development set, and displays the results.  "
            "Your grammar's chunks are <highlight>highlighted</highlight>, and "
            "the correct (gold standard) chunks are "
            "<underline>underlined</underline>.  If they "
            "match, they are displayed in <green>green</green>; otherwise, "
            "they are displayed in <red>red</red>.  The box displays a single "
            "sentence from the development set at a time; use the scrollbar or "
            "the next/previous buttons view additional sentences.\n\n"
            # Performance
            "The lower-right box ('Evaluation') tracks the performance of "
            "your grammar on the development set.  The 'precision' axis "
            "indicates how many of your grammar's chunks are correct; and "
            "the 'recall' axis indicates how many of the gold standard "
            "chunks your system generated.  Typically, you should try to "
            "design a grammar that scores high on both metrics.  The "
            "exact precision and recall of the current grammar, as well "
            "as their harmonic mean (the 'f-score'), are displayed in "
            "the status bar at the bottom of the window.",
        ),
        (
            "Rules",
            "10",
            "<h1>{...regexp...}</h1>"
            "<indent>\nChunk rule: creates new chunks from words matching "
            "regexp.</indent>\n\n"
            "<h1>}...regexp...{</h1>"
            "<indent>\nStrip rule: removes words matching regexp from existing "
            "chunks.</indent>\n\n"
            "<h1>...regexp1...}{...regexp2...</h1>"
            "<indent>\nSplit rule: splits chunks that match regexp1 followed by "
            "regexp2 in two.</indent>\n\n"
            "<h1>...regexp...{}...regexp...</h1>"
            "<indent>\nMerge rule: joins consecutive chunks that match regexp1 "
            "and regexp2</indent>\n",
        ),
        (
            "Regexps",
            "10 60",
            # "Regular Expression Syntax Summary:\n\n"
            "<h1>Pattern\t\tMatches...</h1>\n"
            "<hangindent>"
            "\t<<var>T</var>>\ta word with tag <var>T</var> "
            "(where <var>T</var> may be a regexp).\n"
            "\t<var>x</var>?\tan optional <var>x</var>\n"
            "\t<var>x</var>+\ta sequence of 1 or more <var>x</var>'s\n"
            "\t<var>x</var>*\ta sequence of 0 or more <var>x</var>'s\n"
            "\t<var>x</var>|<var>y</var>\t<var>x</var> or <var>y</var>\n"
            "\t.\tmatches any character\n"
            "\t(<var>x</var>)\tTreats <var>x</var> as a group\n"
            "\t# <var>x...</var>\tTreats <var>x...</var> "
            "(to the end of the line) as a comment\n"
            "\t\\<var>C</var>\tmatches character <var>C</var> "
            "(useful when <var>C</var> is a special character "
            "like + or #)\n"
            "</hangindent>"
            "\n<h1>Examples:</h1>\n"
            "<hangindent>"
            "\t<regexp><NN></regexp>\n"
            '\t\tMatches <match>"cow/NN"</match>\n'
            '\t\tMatches <match>"green/NN"</match>\n'
            "\t<regexp><VB.*></regexp>\n"
            '\t\tMatches <match>"eating/VBG"</match>\n'
            '\t\tMatches <match>"ate/VBD"</match>\n'
            "\t<regexp><IN><DT><NN></regexp>\n"
            '\t\tMatches <match>"on/IN the/DT car/NN"</match>\n'
            "\t<regexp><RB>?<VBD></regexp>\n"
            '\t\tMatches <match>"ran/VBD"</match>\n'
            '\t\tMatches <match>"slowly/RB ate/VBD"</match>\n'
            r"\t<regexp><\#><CD> # This is a comment...</regexp>\n"
            '\t\tMatches <match>"#/# 100/CD"</match>\n'
            "</hangindent>",
        ),
        (
            "Tags",
            "10 60",
            "<h1>Part of Speech Tags:</h1>\n"
            + "<hangindent>"
            + "<<TAGSET>>"
            + "</hangindent>\n",  # this gets auto-substituted w/ self.TAGSET
        ),
    ]

    HELP_AUTOTAG = [
        ("red", dict(foreground="#a00")),
        ("green", dict(foreground="#080")),
        ("highlight", dict(background="#ddd")),
        ("underline", dict(underline=True)),
        ("h1", dict(underline=True)),
        ("indent", dict(lmargin1=20, lmargin2=20)),
        ("hangindent", dict(lmargin1=0, lmargin2=60)),
        ("var", dict(foreground="#88f")),
        ("regexp", dict(foreground="#ba7")),
        ("match", dict(foreground="#6a6")),
    ]

    ##/////////////////////////////////////////////////////////////////
    ##  Config Parameters
    ##/////////////////////////////////////////////////////////////////

    _EVAL_DELAY = 1
    """If the user has not pressed any key for this amount of time (in
       seconds), and the current grammar has not been evaluated, then
       the eval demon will evaluate it."""

    _EVAL_CHUNK = 15
    """The number of sentences that should be evaluated by the eval
       demon each time it runs."""
    _EVAL_FREQ = 0.2
    """The frequency (in seconds) at which the eval demon is run"""
    _EVAL_DEMON_MIN = 0.02
    """The minimum amount of time that the eval demon should take each time
       it runs -- if it takes less than this time, _EVAL_CHUNK will be
       modified upwards."""
    _EVAL_DEMON_MAX = 0.04
    """The maximum amount of time that the eval demon should take each time
       it runs -- if it takes more than this time, _EVAL_CHUNK will be
       modified downwards."""

    _GRAMMARBOX_PARAMS = dict(
        width=40,
        height=12,
        background="#efe",
        highlightbackground="#efe",
        highlightthickness=1,
        relief="groove",
        border=2,
        wrap="word",
    )
    _HELPBOX_PARAMS = dict(
        width=15,
        height=15,
        background="#efe",
        highlightbackground="#efe",
        foreground="#555",
        highlightthickness=1,
        relief="groove",
        border=2,
        wrap="word",
    )
    _DEVSETBOX_PARAMS = dict(
        width=70,
        height=10,
        background="#eef",
        highlightbackground="#eef",
        highlightthickness=1,
        relief="groove",
        border=2,
        wrap="word",
        tabs=(30,),
    )
    _STATUS_PARAMS = dict(background="#9bb", relief="groove", border=2)
    _FONT_PARAMS = dict(family="helvetica", size=-20)
    _FRAME_PARAMS = dict(background="#777", padx=2, pady=2, border=3)
    _EVALBOX_PARAMS = dict(
        background="#eef",
        highlightbackground="#eef",
        highlightthickness=1,
        relief="groove",
        border=2,
        width=300,
        height=280,
    )
    _BUTTON_PARAMS = dict(
        background="#777", activebackground="#777", highlightbackground="#777"
    )
    _HELPTAB_BG_COLOR = "#aba"
    _HELPTAB_FG_COLOR = "#efe"

    _HELPTAB_FG_PARAMS = dict(background="#efe")
    _HELPTAB_BG_PARAMS = dict(background="#aba")
    _HELPTAB_SPACER = 6

    def normalize_grammar(self, grammar):
        # Strip comments
        grammar = re.sub(r"((\\.|[^#])*)(#.*)?", r"\1", grammar)
        # Normalize whitespace
        grammar = re.sub(" +", " ", grammar)
        grammar = re.sub(r"\n\s+", r"\n", grammar)
        grammar = grammar.strip()
        # [xx] Hack: automatically backslash $!
        grammar = re.sub(r"([^\\])\$", r"\1\\$", grammar)
        return grammar

    def __init__(
        self,
        devset_name="conll2000",
        devset=None,
        grammar="",
        chunk_label="NP",
        tagset=None,
    ):
        """
        :param devset_name: The name of the development set; used for
            display & for save files.  If either the name 'treebank'
            or the name 'conll2000' is used, and devset is None, then
            devset will be set automatically.
        :param devset: A list of chunked sentences
        :param grammar: The initial grammar to display.
        :param tagset: Dictionary from tags to string descriptions, used
            for the help page.  Defaults to ``self.TAGSET``.
        """
        self._chunk_label = chunk_label

        if tagset is None:
            tagset = self.TAGSET
        self.tagset = tagset

        # Named development sets:
        if devset is None:
            if devset_name == "conll2000":
                devset = conll2000.chunked_sents("train.txt")  # [:100]
            elif devset == "treebank":
                devset = treebank_chunk.chunked_sents()  # [:100]
            else:
                raise ValueError("Unknown development set %s" % devset_name)

        self.chunker = None
        """The chunker built from the grammar string"""

        self.grammar = grammar
        """The unparsed grammar string"""

        self.normalized_grammar = None
        """A normalized version of ``self.grammar``."""

        self.grammar_changed = 0
        """The last time() that the grammar was changed."""

        self.devset = devset
        """The development set -- a list of chunked sentences."""

        self.devset_name = devset_name
        """The name of the development set (for save files)."""

        self.devset_index = -1
        """The index into the development set of the first instance
           that's currently being viewed."""

        self._last_keypress = 0
        """The time() when a key was most recently pressed"""

        self._history = []
        """A list of (grammar, precision, recall, fscore) tuples for
           grammars that the user has already tried."""

        self._history_index = 0
        """When the user is scrolling through previous grammars, this
           is used to keep track of which grammar they're looking at."""

        self._eval_grammar = None
        """The grammar that is being currently evaluated by the eval
           demon."""

        self._eval_normalized_grammar = None
        """A normalized copy of ``_eval_grammar``."""

        self._eval_index = 0
        """The index of the next sentence in the development set that
           should be looked at by the eval demon."""

        self._eval_score = ChunkScore(chunk_label=chunk_label)
        """The ``ChunkScore`` object that's used to keep track of the score
        of the current grammar on the development set."""

        # Set up the main window.
        top = self.top = Tk()
        top.geometry("+50+50")
        top.title("Regexp Chunk Parser App")
        top.bind("<Control-q>", self.destroy)

        # Variable that restricts how much of the devset we look at.
        self._devset_size = IntVar(top)
        self._devset_size.set(100)

        # Set up all the tkinter widgets
        self._init_fonts(top)
        self._init_widgets(top)
        self._init_bindings(top)
        self._init_menubar(top)
        self.grammarbox.focus()

        # If a grammar was given, then display it.
        if grammar:
            self.grammarbox.insert("end", grammar + "\n")
            self.grammarbox.mark_set("insert", "1.0")

        # Display the first item in the development set
        self.show_devset(0)
        self.update()

    def _init_bindings(self, top):
        top.bind("<Control-n>", self._devset_next)
        top.bind("<Control-p>", self._devset_prev)
        top.bind("<Control-t>", self.toggle_show_trace)
        top.bind("<KeyPress>", self.update)
        top.bind("<Control-s>", lambda e: self.save_grammar())
        top.bind("<Control-o>", lambda e: self.load_grammar())
        self.grammarbox.bind("<Control-t>", self.toggle_show_trace)
        self.grammarbox.bind("<Control-n>", self._devset_next)
        self.grammarbox.bind("<Control-p>", self._devset_prev)

        # Redraw the eval graph when the window size changes
        self.evalbox.bind("<Configure>", self._eval_plot)

    def _init_fonts(self, top):
        # TWhat's our font size (default=same as sysfont)
        self._size = IntVar(top)
        self._size.set(20)
        self._font = Font(family="helvetica", size=-self._size.get())
        self._smallfont = Font(
            family="helvetica", size=-(int(self._size.get() * 14 // 20))
        )

    def _init_menubar(self, parent):
        menubar = Menu(parent)

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Reset Application", underline=0, command=self.reset)
        filemenu.add_command(
            label="Save Current Grammar",
            underline=0,
            accelerator="Ctrl-s",
            command=self.save_grammar,
        )
        filemenu.add_command(
            label="Load Grammar",
            underline=0,
            accelerator="Ctrl-o",
            command=self.load_grammar,
        )

        filemenu.add_command(
            label="Save Grammar History", underline=13, command=self.save_history
        )

        filemenu.add_command(
            label="Exit", underline=1, command=self.destroy, accelerator="Ctrl-q"
        )
        menubar.add_cascade(label="File", underline=0, menu=filemenu)

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
            value=16,
            command=self.resize,
        )
        viewmenu.add_radiobutton(
            label="Medium",
            variable=self._size,
            underline=0,
            value=20,
            command=self.resize,
        )
        viewmenu.add_radiobutton(
            label="Large",
            variable=self._size,
            underline=0,
            value=24,
            command=self.resize,
        )
        viewmenu.add_radiobutton(
            label="Huge",
            variable=self._size,
            underline=0,
            value=34,
            command=self.resize,
        )
        menubar.add_cascade(label="View", underline=0, menu=viewmenu)

        devsetmenu = Menu(menubar, tearoff=0)
        devsetmenu.add_radiobutton(
            label="50 sentences",
            variable=self._devset_size,
            value=50,
            command=self.set_devset_size,
        )
        devsetmenu.add_radiobutton(
            label="100 sentences",
            variable=self._devset_size,
            value=100,
            command=self.set_devset_size,
        )
        devsetmenu.add_radiobutton(
            label="200 sentences",
            variable=self._devset_size,
            value=200,
            command=self.set_devset_size,
        )
        devsetmenu.add_radiobutton(
            label="500 sentences",
            variable=self._devset_size,
            value=500,
            command=self.set_devset_size,
        )
        menubar.add_cascade(label="Development-Set", underline=0, menu=devsetmenu)

        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", underline=0, command=self.about)
        menubar.add_cascade(label="Help", underline=0, menu=helpmenu)

        parent.config(menu=menubar)

    def toggle_show_trace(self, *e):
        if self._showing_trace:
            self.show_devset()
        else:
            self.show_trace()
        return "break"

    _SCALE_N = 5  # center on the last 5 examples.
    _DRAW_LINES = False

    def _eval_plot(self, *e, **config):
        width = config.get("width", self.evalbox.winfo_width())
        height = config.get("height", self.evalbox.winfo_height())

        # Clear the canvas
        self.evalbox.delete("all")

        # Draw the precision & recall labels.
        tag = self.evalbox.create_text(
            10, height // 2 - 10, justify="left", anchor="w", text="Precision"
        )
        left, right = self.evalbox.bbox(tag)[2] + 5, width - 10
        tag = self.evalbox.create_text(
            left + (width - left) // 2,
            height - 10,
            anchor="s",
            text="Recall",
            justify="center",
        )
        top, bot = 10, self.evalbox.bbox(tag)[1] - 10

        # Draw masks for clipping the plot.
        bg = self._EVALBOX_PARAMS["background"]
        self.evalbox.lower(
            self.evalbox.create_rectangle(0, 0, left - 1, 5000, fill=bg, outline=bg)
        )
        self.evalbox.lower(
            self.evalbox.create_rectangle(0, bot + 1, 5000, 5000, fill=bg, outline=bg)
        )

        # Calculate the plot's scale.
        if self._autoscale.get() and len(self._history) > 1:
            max_precision = max_recall = 0
            min_precision = min_recall = 1
            for i in range(1, min(len(self._history), self._SCALE_N + 1)):
                grammar, precision, recall, fmeasure = self._history[-i]
                min_precision = min(precision, min_precision)
                min_recall = min(recall, min_recall)
                max_precision = max(precision, max_precision)
                max_recall = max(recall, max_recall)
            #             if max_precision-min_precision > max_recall-min_recall:
            #                 min_recall -= (max_precision-min_precision)/2
            #                 max_recall += (max_precision-min_precision)/2
            #             else:
            #                 min_precision -= (max_recall-min_recall)/2
            #                 max_precision += (max_recall-min_recall)/2
            #             if min_recall < 0:
            #                 max_recall -= min_recall
            #                 min_recall = 0
            #             if min_precision < 0:
            #                 max_precision -= min_precision
            #                 min_precision = 0
            min_precision = max(min_precision - 0.01, 0)
            min_recall = max(min_recall - 0.01, 0)
            max_precision = min(max_precision + 0.01, 1)
            max_recall = min(max_recall + 0.01, 1)
        else:
            min_precision = min_recall = 0
            max_precision = max_recall = 1

        # Draw the axis lines & grid lines
        for i in range(11):
            x = left + (right - left) * (
                (i / 10.0 - min_recall) / (max_recall - min_recall)
            )
            y = bot - (bot - top) * (
                (i / 10.0 - min_precision) / (max_precision - min_precision)
            )
            if left < x < right:
                self.evalbox.create_line(x, top, x, bot, fill="#888")
            if top < y < bot:
                self.evalbox.create_line(left, y, right, y, fill="#888")
        self.evalbox.create_line(left, top, left, bot)
        self.evalbox.create_line(left, bot, right, bot)

        # Display the plot's scale
        self.evalbox.create_text(
            left - 3,
            bot,
            justify="right",
            anchor="se",
            text="%d%%" % (100 * min_precision),
        )
        self.evalbox.create_text(
            left - 3,
            top,
            justify="right",
            anchor="ne",
            text="%d%%" % (100 * max_precision),
        )
        self.evalbox.create_text(
            left,
            bot + 3,
            justify="center",
            anchor="nw",
            text="%d%%" % (100 * min_recall),
        )
        self.evalbox.create_text(
            right,
            bot + 3,
            justify="center",
            anchor="ne",
            text="%d%%" % (100 * max_recall),
        )

        # Display the scores.
        prev_x = prev_y = None
        for i, (_, precision, recall, fscore) in enumerate(self._history):
            x = left + (right - left) * (
                (recall - min_recall) / (max_recall - min_recall)
            )
            y = bot - (bot - top) * (
                (precision - min_precision) / (max_precision - min_precision)
            )
            if i == self._history_index:
                self.evalbox.create_oval(
                    x - 2, y - 2, x + 2, y + 2, fill="#0f0", outline="#000"
                )
                self.status["text"] = (
                    "Precision: %.2f%%\t" % (precision * 100)
                    + "Recall: %.2f%%\t" % (recall * 100)
                    + "F-score: %.2f%%" % (fscore * 100)
                )
            else:
                self.evalbox.lower(
                    self.evalbox.create_oval(
                        x - 2, y - 2, x + 2, y + 2, fill="#afa", outline="#8c8"
                    )
                )
            if prev_x is not None and self._eval_lines.get():
                self.evalbox.lower(
                    self.evalbox.create_line(prev_x, prev_y, x, y, fill="#8c8")
                )
            prev_x, prev_y = x, y

    _eval_demon_running = False

    def _eval_demon(self):
        if self.top is None:
            return
        if self.chunker is None:
            self._eval_demon_running = False
            return

        # Note our starting time.
        t0 = time.time()

        # If are still typing, then wait for them to finish.
        if (
            time.time() - self._last_keypress < self._EVAL_DELAY
            and self.normalized_grammar != self._eval_normalized_grammar
        ):
            self._eval_demon_running = True
            return self.top.after(int(self._EVAL_FREQ * 1000), self._eval_demon)

        # If the grammar changed, restart the evaluation.
        if self.normalized_grammar != self._eval_normalized_grammar:
            # Check if we've seen this grammar already.  If so, then
            # just use the old evaluation values.
            for (g, p, r, f) in self._history:
                if self.normalized_grammar == self.normalize_grammar(g):
                    self._history.append((g, p, r, f))
                    self._history_index = len(self._history) - 1
                    self._eval_plot()
                    self._eval_demon_running = False
                    self._eval_normalized_grammar = None
                    return
            self._eval_index = 0
            self._eval_score = ChunkScore(chunk_label=self._chunk_label)
            self._eval_grammar = self.grammar
            self._eval_normalized_grammar = self.normalized_grammar

        # If the grammar is empty, the don't bother evaluating it, or
        # recording it in history -- the score will just be 0.
        if self.normalized_grammar.strip() == "":
            # self._eval_index = self._devset_size.get()
            self._eval_demon_running = False
            return

        # Score the next set of examples
        for gold in self.devset[
            self._eval_index : min(
                self._eval_index + self._EVAL_CHUNK, self._devset_size.get()
            )
        ]:
            guess = self._chunkparse(gold.leaves())
            self._eval_score.score(gold, guess)

        # update our index in the devset.
        self._eval_index += self._EVAL_CHUNK

        # Check if we're done
        if self._eval_index >= self._devset_size.get():
            self._history.append(
                (
                    self._eval_grammar,
                    self._eval_score.precision(),
                    self._eval_score.recall(),
                    self._eval_score.f_measure(),
                )
            )
            self._history_index = len(self._history) - 1
            self._eval_plot()
            self._eval_demon_running = False
            self._eval_normalized_grammar = None
        else:
            progress = 100 * self._eval_index / self._devset_size.get()
            self.status["text"] = "Evaluating on Development Set (%d%%)" % progress
            self._eval_demon_running = True
            self._adaptively_modify_eval_chunk(time.time() - t0)
            self.top.after(int(self._EVAL_FREQ * 1000), self._eval_demon)

    def _adaptively_modify_eval_chunk(self, t):
        """
        Modify _EVAL_CHUNK to try to keep the amount of time that the
        eval demon takes between _EVAL_DEMON_MIN and _EVAL_DEMON_MAX.

        :param t: The amount of time that the eval demon took.
        """
        if t > self._EVAL_DEMON_MAX and self._EVAL_CHUNK > 5:
            self._EVAL_CHUNK = min(
                self._EVAL_CHUNK - 1,
                max(
                    int(self._EVAL_CHUNK * (self._EVAL_DEMON_MAX / t)),
                    self._EVAL_CHUNK - 10,
                ),
            )
        elif t < self._EVAL_DEMON_MIN:
            self._EVAL_CHUNK = max(
                self._EVAL_CHUNK + 1,
                min(
                    int(self._EVAL_CHUNK * (self._EVAL_DEMON_MIN / t)),
                    self._EVAL_CHUNK + 10,
                ),
            )

    def _init_widgets(self, top):
        frame0 = Frame(top, **self._FRAME_PARAMS)
        frame0.grid_columnconfigure(0, weight=4)
        frame0.grid_columnconfigure(3, weight=2)
        frame0.grid_rowconfigure(1, weight=1)
        frame0.grid_rowconfigure(5, weight=1)

        # The grammar
        self.grammarbox = Text(frame0, font=self._font, **self._GRAMMARBOX_PARAMS)
        self.grammarlabel = Label(
            frame0,
            font=self._font,
            text="Grammar:",
            highlightcolor="black",
            background=self._GRAMMARBOX_PARAMS["background"],
        )
        self.grammarlabel.grid(column=0, row=0, sticky="SW")
        self.grammarbox.grid(column=0, row=1, sticky="NEWS")

        # Scroll bar for grammar
        grammar_scrollbar = Scrollbar(frame0, command=self.grammarbox.yview)
        grammar_scrollbar.grid(column=1, row=1, sticky="NWS")
        self.grammarbox.config(yscrollcommand=grammar_scrollbar.set)

        # grammar buttons
        bg = self._FRAME_PARAMS["background"]
        frame3 = Frame(frame0, background=bg)
        frame3.grid(column=0, row=2, sticky="EW")
        Button(
            frame3,
            text="Prev Grammar",
            command=self._history_prev,
            **self._BUTTON_PARAMS,
        ).pack(side="left")
        Button(
            frame3,
            text="Next Grammar",
            command=self._history_next,
            **self._BUTTON_PARAMS,
        ).pack(side="left")

        # Help box
        self.helpbox = Text(frame0, font=self._smallfont, **self._HELPBOX_PARAMS)
        self.helpbox.grid(column=3, row=1, sticky="NEWS")
        self.helptabs = {}
        bg = self._FRAME_PARAMS["background"]
        helptab_frame = Frame(frame0, background=bg)
        helptab_frame.grid(column=3, row=0, sticky="SW")
        for i, (tab, tabstops, text) in enumerate(self.HELP):
            label = Label(helptab_frame, text=tab, font=self._smallfont)
            label.grid(column=i * 2, row=0, sticky="S")
            # help_frame.grid_columnconfigure(i, weight=1)
            # label.pack(side='left')
            label.bind("<ButtonPress>", lambda e, tab=tab: self.show_help(tab))
            self.helptabs[tab] = label
            Frame(
                helptab_frame, height=1, width=self._HELPTAB_SPACER, background=bg
            ).grid(column=i * 2 + 1, row=0)
        self.helptabs[self.HELP[0][0]].configure(font=self._font)
        self.helpbox.tag_config("elide", elide=True)
        for (tag, params) in self.HELP_AUTOTAG:
            self.helpbox.tag_config("tag-%s" % tag, **params)
        self.show_help(self.HELP[0][0])

        # Scroll bar for helpbox
        help_scrollbar = Scrollbar(frame0, command=self.helpbox.yview)
        self.helpbox.config(yscrollcommand=help_scrollbar.set)
        help_scrollbar.grid(column=4, row=1, sticky="NWS")

        # The dev set
        frame4 = Frame(frame0, background=self._FRAME_PARAMS["background"])
        self.devsetbox = Text(frame4, font=self._font, **self._DEVSETBOX_PARAMS)
        self.devsetbox.pack(expand=True, fill="both")
        self.devsetlabel = Label(
            frame0,
            font=self._font,
            text="Development Set:",
            justify="right",
            background=self._DEVSETBOX_PARAMS["background"],
        )
        self.devsetlabel.grid(column=0, row=4, sticky="SW")
        frame4.grid(column=0, row=5, sticky="NEWS")

        # dev set scrollbars
        self.devset_scroll = Scrollbar(frame0, command=self._devset_scroll)
        self.devset_scroll.grid(column=1, row=5, sticky="NWS")
        self.devset_xscroll = Scrollbar(
            frame4, command=self.devsetbox.xview, orient="horiz"
        )
        self.devsetbox["xscrollcommand"] = self.devset_xscroll.set
        self.devset_xscroll.pack(side="bottom", fill="x")

        # dev set buttons
        bg = self._FRAME_PARAMS["background"]
        frame1 = Frame(frame0, background=bg)
        frame1.grid(column=0, row=7, sticky="EW")
        Button(
            frame1,
            text="Prev Example (Ctrl-p)",
            command=self._devset_prev,
            **self._BUTTON_PARAMS,
        ).pack(side="left")
        Button(
            frame1,
            text="Next Example (Ctrl-n)",
            command=self._devset_next,
            **self._BUTTON_PARAMS,
        ).pack(side="left")
        self.devset_button = Button(
            frame1,
            text="Show example",
            command=self.show_devset,
            state="disabled",
            **self._BUTTON_PARAMS,
        )
        self.devset_button.pack(side="right")
        self.trace_button = Button(
            frame1, text="Show trace", command=self.show_trace, **self._BUTTON_PARAMS
        )
        self.trace_button.pack(side="right")

        # evaluation box
        self.evalbox = Canvas(frame0, **self._EVALBOX_PARAMS)
        label = Label(
            frame0,
            font=self._font,
            text="Evaluation:",
            justify="right",
            background=self._EVALBOX_PARAMS["background"],
        )
        label.grid(column=3, row=4, sticky="SW")
        self.evalbox.grid(column=3, row=5, sticky="NEWS", columnspan=2)

        # evaluation box buttons
        bg = self._FRAME_PARAMS["background"]
        frame2 = Frame(frame0, background=bg)
        frame2.grid(column=3, row=7, sticky="EW")
        self._autoscale = IntVar(self.top)
        self._autoscale.set(False)
        Checkbutton(
            frame2,
            variable=self._autoscale,
            command=self._eval_plot,
            text="Zoom",
            **self._BUTTON_PARAMS,
        ).pack(side="left")
        self._eval_lines = IntVar(self.top)
        self._eval_lines.set(False)
        Checkbutton(
            frame2,
            variable=self._eval_lines,
            command=self._eval_plot,
            text="Lines",
            **self._BUTTON_PARAMS,
        ).pack(side="left")
        Button(frame2, text="History", **self._BUTTON_PARAMS).pack(side="right")

        # The status label
        self.status = Label(frame0, font=self._font, **self._STATUS_PARAMS)
        self.status.grid(column=0, row=9, sticky="NEW", padx=3, pady=2, columnspan=5)

        # Help box & devset box can't be edited.
        self.helpbox["state"] = "disabled"
        self.devsetbox["state"] = "disabled"

        # Spacers
        bg = self._FRAME_PARAMS["background"]
        Frame(frame0, height=10, width=0, background=bg).grid(column=0, row=3)
        Frame(frame0, height=0, width=10, background=bg).grid(column=2, row=0)
        Frame(frame0, height=6, width=0, background=bg).grid(column=0, row=8)

        # pack the frame.
        frame0.pack(fill="both", expand=True)

        # Set up colors for the devset box
        self.devsetbox.tag_config("true-pos", background="#afa", underline="True")
        self.devsetbox.tag_config("false-neg", underline="True", foreground="#800")
        self.devsetbox.tag_config("false-pos", background="#faa")
        self.devsetbox.tag_config("trace", foreground="#666", wrap="none")
        self.devsetbox.tag_config("wrapindent", lmargin2=30, wrap="none")
        self.devsetbox.tag_config("error", foreground="#800")

        # And for the grammarbox
        self.grammarbox.tag_config("error", background="#fec")
        self.grammarbox.tag_config("comment", foreground="#840")
        self.grammarbox.tag_config("angle", foreground="#00f")
        self.grammarbox.tag_config("brace", foreground="#0a0")
        self.grammarbox.tag_config("hangindent", lmargin1=0, lmargin2=40)

    _showing_trace = False

    def show_trace(self, *e):
        self._showing_trace = True
        self.trace_button["state"] = "disabled"
        self.devset_button["state"] = "normal"

        self.devsetbox["state"] = "normal"
        # self.devsetbox['wrap'] = 'none'
        self.devsetbox.delete("1.0", "end")
        self.devsetlabel["text"] = "Development Set (%d/%d)" % (
            (self.devset_index + 1, self._devset_size.get())
        )

        if self.chunker is None:
            self.devsetbox.insert("1.0", "Trace: waiting for a valid grammar.")
            self.devsetbox.tag_add("error", "1.0", "end")
            return  # can't do anything more

        gold_tree = self.devset[self.devset_index]
        rules = self.chunker.rules()

        # Calculate the tag sequence
        tagseq = "\t"
        charnum = [1]
        for wordnum, (word, pos) in enumerate(gold_tree.leaves()):
            tagseq += "%s " % pos
            charnum.append(len(tagseq))
        self.charnum = {
            (i, j): charnum[j]
            for i in range(len(rules) + 1)
            for j in range(len(charnum))
        }
        self.linenum = {i: i * 2 + 2 for i in range(len(rules) + 1)}

        for i in range(len(rules) + 1):
            if i == 0:
                self.devsetbox.insert("end", "Start:\n")
                self.devsetbox.tag_add("trace", "end -2c linestart", "end -2c")
            else:
                self.devsetbox.insert("end", "Apply %s:\n" % rules[i - 1])
                self.devsetbox.tag_add("trace", "end -2c linestart", "end -2c")
            # Display the tag sequence.
            self.devsetbox.insert("end", tagseq + "\n")
            self.devsetbox.tag_add("wrapindent", "end -2c linestart", "end -2c")
            # Run a partial parser, and extract gold & test chunks
            chunker = RegexpChunkParser(rules[:i])
            test_tree = self._chunkparse(gold_tree.leaves())
            gold_chunks = self._chunks(gold_tree)
            test_chunks = self._chunks(test_tree)
            # Compare them.
            for chunk in gold_chunks.intersection(test_chunks):
                self._color_chunk(i, chunk, "true-pos")
            for chunk in gold_chunks - test_chunks:
                self._color_chunk(i, chunk, "false-neg")
            for chunk in test_chunks - gold_chunks:
                self._color_chunk(i, chunk, "false-pos")
        self.devsetbox.insert("end", "Finished.\n")
        self.devsetbox.tag_add("trace", "end -2c linestart", "end -2c")

        # This is a hack, because the x-scrollbar isn't updating its
        # position right -- I'm not sure what the underlying cause is
        # though.  (This is on OS X w/ python 2.5)
        self.top.after(100, self.devset_xscroll.set, 0, 0.3)

    def show_help(self, tab):
        self.helpbox["state"] = "normal"
        self.helpbox.delete("1.0", "end")
        for (name, tabstops, text) in self.HELP:
            if name == tab:
                text = text.replace(
                    "<<TAGSET>>",
                    "\n".join(
                        "\t%s\t%s" % item
                        for item in sorted(
                            list(self.tagset.items()),
                            key=lambda t_w: re.match(r"\w+", t_w[0])
                            and (0, t_w[0])
                            or (1, t_w[0]),
                        )
                    ),
                )

                self.helptabs[name].config(**self._HELPTAB_FG_PARAMS)
                self.helpbox.config(tabs=tabstops)
                self.helpbox.insert("1.0", text + "\n" * 20)
                C = "1.0 + %d chars"
                for (tag, params) in self.HELP_AUTOTAG:
                    pattern = f"(?s)(<{tag}>)(.*?)(</{tag}>)"
                    for m in re.finditer(pattern, text):
                        self.helpbox.tag_add("elide", C % m.start(1), C % m.end(1))
                        self.helpbox.tag_add(
                            "tag-%s" % tag, C % m.start(2), C % m.end(2)
                        )
                        self.helpbox.tag_add("elide", C % m.start(3), C % m.end(3))
            else:
                self.helptabs[name].config(**self._HELPTAB_BG_PARAMS)
        self.helpbox["state"] = "disabled"

    def _history_prev(self, *e):
        self._view_history(self._history_index - 1)
        return "break"

    def _history_next(self, *e):
        self._view_history(self._history_index + 1)
        return "break"

    def _view_history(self, index):
        # Bounds & sanity checking:
        index = max(0, min(len(self._history) - 1, index))
        if not self._history:
            return
        # Already viewing the requested history item?
        if index == self._history_index:
            return
        # Show the requested grammar.  It will get added to _history
        # only if they edit it (causing self.update() to get run.)
        self.grammarbox["state"] = "normal"
        self.grammarbox.delete("1.0", "end")
        self.grammarbox.insert("end", self._history[index][0])
        self.grammarbox.mark_set("insert", "1.0")
        self._history_index = index
        self._syntax_highlight_grammar(self._history[index][0])
        # Record the normalized grammar & regenerate the chunker.
        self.normalized_grammar = self.normalize_grammar(self._history[index][0])
        if self.normalized_grammar:
            rules = [
                RegexpChunkRule.fromstring(line)
                for line in self.normalized_grammar.split("\n")
            ]
        else:
            rules = []
        self.chunker = RegexpChunkParser(rules)
        # Show the score.
        self._eval_plot()
        # Update the devset box
        self._highlight_devset()
        if self._showing_trace:
            self.show_trace()
        # Update the grammar label
        if self._history_index < len(self._history) - 1:
            self.grammarlabel["text"] = "Grammar {}/{}:".format(
                self._history_index + 1,
                len(self._history),
            )
        else:
            self.grammarlabel["text"] = "Grammar:"

    def _devset_next(self, *e):
        self._devset_scroll("scroll", 1, "page")
        return "break"

    def _devset_prev(self, *e):
        self._devset_scroll("scroll", -1, "page")
        return "break"

    def destroy(self, *e):
        if self.top is None:
            return
        self.top.destroy()
        self.top = None

    def _devset_scroll(self, command, *args):
        N = 1  # size of a page -- one sentence.
        showing_trace = self._showing_trace
        if command == "scroll" and args[1].startswith("unit"):
            self.show_devset(self.devset_index + int(args[0]))
        elif command == "scroll" and args[1].startswith("page"):
            self.show_devset(self.devset_index + N * int(args[0]))
        elif command == "moveto":
            self.show_devset(int(float(args[0]) * self._devset_size.get()))
        else:
            assert 0, f"bad scroll command {command} {args}"
        if showing_trace:
            self.show_trace()

    def show_devset(self, index=None):
        if index is None:
            index = self.devset_index

        # Bounds checking
        index = min(max(0, index), self._devset_size.get() - 1)

        if index == self.devset_index and not self._showing_trace:
            return
        self.devset_index = index

        self._showing_trace = False
        self.trace_button["state"] = "normal"
        self.devset_button["state"] = "disabled"

        # Clear the text box.
        self.devsetbox["state"] = "normal"
        self.devsetbox["wrap"] = "word"
        self.devsetbox.delete("1.0", "end")
        self.devsetlabel["text"] = "Development Set (%d/%d)" % (
            (self.devset_index + 1, self._devset_size.get())
        )

        # Add the sentences
        sample = self.devset[self.devset_index : self.devset_index + 1]
        self.charnum = {}
        self.linenum = {0: 1}
        for sentnum, sent in enumerate(sample):
            linestr = ""
            for wordnum, (word, pos) in enumerate(sent.leaves()):
                self.charnum[sentnum, wordnum] = len(linestr)
                linestr += f"{word}/{pos} "
                self.charnum[sentnum, wordnum + 1] = len(linestr)
            self.devsetbox.insert("end", linestr[:-1] + "\n\n")

        # Highlight chunks in the dev set
        if self.chunker is not None:
            self._highlight_devset()
        self.devsetbox["state"] = "disabled"

        # Update the scrollbar
        first = self.devset_index / self._devset_size.get()
        last = (self.devset_index + 2) / self._devset_size.get()
        self.devset_scroll.set(first, last)

    def _chunks(self, tree):
        chunks = set()
        wordnum = 0
        for child in tree:
            if isinstance(child, Tree):
                if child.label() == self._chunk_label:
                    chunks.add((wordnum, wordnum + len(child)))
                wordnum += len(child)
            else:
                wordnum += 1
        return chunks

    def _syntax_highlight_grammar(self, grammar):
        if self.top is None:
            return
        self.grammarbox.tag_remove("comment", "1.0", "end")
        self.grammarbox.tag_remove("angle", "1.0", "end")
        self.grammarbox.tag_remove("brace", "1.0", "end")
        self.grammarbox.tag_add("hangindent", "1.0", "end")
        for lineno, line in enumerate(grammar.split("\n")):
            if not line.strip():
                continue
            m = re.match(r"(\\.|[^#])*(#.*)?", line)
            comment_start = None
            if m.group(2):
                comment_start = m.start(2)
                s = "%d.%d" % (lineno + 1, m.start(2))
                e = "%d.%d" % (lineno + 1, m.end(2))
                self.grammarbox.tag_add("comment", s, e)
            for m in re.finditer("[<>{}]", line):
                if comment_start is not None and m.start() >= comment_start:
                    break
                s = "%d.%d" % (lineno + 1, m.start())
                e = "%d.%d" % (lineno + 1, m.end())
                if m.group() in "<>":
                    self.grammarbox.tag_add("angle", s, e)
                else:
                    self.grammarbox.tag_add("brace", s, e)

    def _grammarcheck(self, grammar):
        if self.top is None:
            return
        self.grammarbox.tag_remove("error", "1.0", "end")
        self._grammarcheck_errs = []
        for lineno, line in enumerate(grammar.split("\n")):
            line = re.sub(r"((\\.|[^#])*)(#.*)?", r"\1", line)
            line = line.strip()
            if line:
                try:
                    RegexpChunkRule.fromstring(line)
                except ValueError as e:
                    self.grammarbox.tag_add(
                        "error", "%s.0" % (lineno + 1), "%s.0 lineend" % (lineno + 1)
                    )
        self.status["text"] = ""

    def update(self, *event):
        # Record when update was called (for grammarcheck)
        if event:
            self._last_keypress = time.time()

        # Read the grammar from the Text box.
        self.grammar = grammar = self.grammarbox.get("1.0", "end")

        # If the grammar hasn't changed, do nothing:
        normalized_grammar = self.normalize_grammar(grammar)
        if normalized_grammar == self.normalized_grammar:
            return
        else:
            self.normalized_grammar = normalized_grammar

        # If the grammar has changed, and we're looking at history,
        # then stop looking at history.
        if self._history_index < len(self._history) - 1:
            self.grammarlabel["text"] = "Grammar:"

        self._syntax_highlight_grammar(grammar)

        # The grammar has changed; try parsing it.  If it doesn't
        # parse, do nothing.  (flag error location?)
        try:
            # Note: the normalized grammar has no blank lines.
            if normalized_grammar:
                rules = [
                    RegexpChunkRule.fromstring(line)
                    for line in normalized_grammar.split("\n")
                ]
            else:
                rules = []
        except ValueError as e:
            # Use the un-normalized grammar for error highlighting.
            self._grammarcheck(grammar)
            self.chunker = None
            return

        self.chunker = RegexpChunkParser(rules)
        self.grammarbox.tag_remove("error", "1.0", "end")
        self.grammar_changed = time.time()
        # Display the results
        if self._showing_trace:
            self.show_trace()
        else:
            self._highlight_devset()
        # Start the eval demon
        if not self._eval_demon_running:
            self._eval_demon()

    def _highlight_devset(self, sample=None):
        if sample is None:
            sample = self.devset[self.devset_index : self.devset_index + 1]

        self.devsetbox.tag_remove("true-pos", "1.0", "end")
        self.devsetbox.tag_remove("false-neg", "1.0", "end")
        self.devsetbox.tag_remove("false-pos", "1.0", "end")

        # Run the grammar on the test cases.
        for sentnum, gold_tree in enumerate(sample):
            # Run the chunk parser
            test_tree = self._chunkparse(gold_tree.leaves())
            # Extract gold & test chunks
            gold_chunks = self._chunks(gold_tree)
            test_chunks = self._chunks(test_tree)
            # Compare them.
            for chunk in gold_chunks.intersection(test_chunks):
                self._color_chunk(sentnum, chunk, "true-pos")
            for chunk in gold_chunks - test_chunks:
                self._color_chunk(sentnum, chunk, "false-neg")
            for chunk in test_chunks - gold_chunks:
                self._color_chunk(sentnum, chunk, "false-pos")

    def _chunkparse(self, words):
        try:
            return self.chunker.parse(words)
        except (ValueError, IndexError) as e:
            # There's an error somewhere in the grammar, but we're not sure
            # exactly where, so just mark the whole grammar as bad.
            # E.g., this is caused by: "({<NN>})"
            self.grammarbox.tag_add("error", "1.0", "end")
            # Treat it as tagging nothing:
            return words

    def _color_chunk(self, sentnum, chunk, tag):
        start, end = chunk
        self.devsetbox.tag_add(
            tag,
            f"{self.linenum[sentnum]}.{self.charnum[sentnum, start]}",
            f"{self.linenum[sentnum]}.{self.charnum[sentnum, end] - 1}",
        )

    def reset(self):
        # Clear various variables
        self.chunker = None
        self.grammar = None
        self.normalized_grammar = None
        self.grammar_changed = 0
        self._history = []
        self._history_index = 0
        # Update the on-screen display.
        self.grammarbox.delete("1.0", "end")
        self.show_devset(0)
        self.update()
        # self._eval_plot()

    SAVE_GRAMMAR_TEMPLATE = (
        "# Regexp Chunk Parsing Grammar\n"
        "# Saved %(date)s\n"
        "#\n"
        "# Development set: %(devset)s\n"
        "#   Precision: %(precision)s\n"
        "#   Recall:    %(recall)s\n"
        "#   F-score:   %(fscore)s\n\n"
        "%(grammar)s\n"
    )

    def save_grammar(self, filename=None):
        if not filename:
            ftypes = [("Chunk Gramamr", ".chunk"), ("All files", "*")]
            filename = asksaveasfilename(filetypes=ftypes, defaultextension=".chunk")
            if not filename:
                return
        if self._history and self.normalized_grammar == self.normalize_grammar(
            self._history[-1][0]
        ):
            precision, recall, fscore = (
                "%.2f%%" % (100 * v) for v in self._history[-1][1:]
            )
        elif self.chunker is None:
            precision = recall = fscore = "Grammar not well formed"
        else:
            precision = recall = fscore = "Not finished evaluation yet"

        with open(filename, "w") as outfile:
            outfile.write(
                self.SAVE_GRAMMAR_TEMPLATE
                % dict(
                    date=time.ctime(),
                    devset=self.devset_name,
                    precision=precision,
                    recall=recall,
                    fscore=fscore,
                    grammar=self.grammar.strip(),
                )
            )

    def load_grammar(self, filename=None):
        if not filename:
            ftypes = [("Chunk Gramamr", ".chunk"), ("All files", "*")]
            filename = askopenfilename(filetypes=ftypes, defaultextension=".chunk")
            if not filename:
                return
        self.grammarbox.delete("1.0", "end")
        self.update()
        with open(filename) as infile:
            grammar = infile.read()
        grammar = re.sub(
            r"^\# Regexp Chunk Parsing Grammar[\s\S]*" "F-score:.*\n", "", grammar
        ).lstrip()
        self.grammarbox.insert("1.0", grammar)
        self.update()

    def save_history(self, filename=None):
        if not filename:
            ftypes = [("Chunk Gramamr History", ".txt"), ("All files", "*")]
            filename = asksaveasfilename(filetypes=ftypes, defaultextension=".txt")
            if not filename:
                return

        with open(filename, "w") as outfile:
            outfile.write("# Regexp Chunk Parsing Grammar History\n")
            outfile.write("# Saved %s\n" % time.ctime())
            outfile.write("# Development set: %s\n" % self.devset_name)
            for i, (g, p, r, f) in enumerate(self._history):
                hdr = (
                    "Grammar %d/%d (precision=%.2f%%, recall=%.2f%%, "
                    "fscore=%.2f%%)"
                    % (i + 1, len(self._history), p * 100, r * 100, f * 100)
                )
                outfile.write("\n%s\n" % hdr)
                outfile.write("".join("  %s\n" % line for line in g.strip().split()))

            if not (
                self._history
                and self.normalized_grammar
                == self.normalize_grammar(self._history[-1][0])
            ):
                if self.chunker is None:
                    outfile.write("\nCurrent Grammar (not well-formed)\n")
                else:
                    outfile.write("\nCurrent Grammar (not evaluated)\n")
                outfile.write(
                    "".join("  %s\n" % line for line in self.grammar.strip().split())
                )

    def about(self, *e):
        ABOUT = "NLTK RegExp Chunk Parser Application\n" + "Written by Edward Loper"
        TITLE = "About: Regular Expression Chunk Parser Application"
        try:
            from tkinter.messagebox import Message

            Message(message=ABOUT, title=TITLE).show()
        except:
            ShowText(self.top, TITLE, ABOUT)

    def set_devset_size(self, size=None):
        if size is not None:
            self._devset_size.set(size)
        self._devset_size.set(min(len(self.devset), self._devset_size.get()))
        self.show_devset(1)
        self.show_devset(0)
        # what about history?  Evaluated at diff dev set sizes!

    def resize(self, size=None):
        if size is not None:
            self._size.set(size)
        size = self._size.get()
        self._font.configure(size=-(abs(size)))
        self._smallfont.configure(size=min(-10, -(abs(size)) * 14 // 20))

    def mainloop(self, *args, **kwargs):
        """
        Enter the Tkinter mainloop.  This function must be called if
        this demo is created from a non-interactive program (e.g.
        from a secript); otherwise, the demo will close as soon as
        the script completes.
        """
        if in_idle():
            return
        self.top.mainloop(*args, **kwargs)


def app():
    RegexpChunkApp().mainloop()


if __name__ == "__main__":
    app()

__all__ = ["app"]
