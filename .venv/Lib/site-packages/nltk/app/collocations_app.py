# Natural Language Toolkit: Collocations Application
# Much of the GUI code is imported from concordance.py; We intend to merge these tools together
# Copyright (C) 2001-2023 NLTK Project
# Author: Sumukh Ghodke <sghodke@csse.unimelb.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#


import queue as q
import threading
from tkinter import (
    END,
    LEFT,
    SUNKEN,
    Button,
    Frame,
    IntVar,
    Label,
    Menu,
    OptionMenu,
    Scrollbar,
    StringVar,
    Text,
    Tk,
)
from tkinter.font import Font

from nltk.corpus import (
    alpino,
    brown,
    cess_cat,
    cess_esp,
    floresta,
    indian,
    mac_morpho,
    machado,
    nps_chat,
    sinica_treebank,
    treebank,
)
from nltk.probability import FreqDist
from nltk.util import in_idle

CORPUS_LOADED_EVENT = "<<CL_EVENT>>"
ERROR_LOADING_CORPUS_EVENT = "<<ELC_EVENT>>"
POLL_INTERVAL = 100

_DEFAULT = "English: Brown Corpus (Humor)"
_CORPORA = {
    "Catalan: CESS-CAT Corpus": lambda: cess_cat.words(),
    "English: Brown Corpus": lambda: brown.words(),
    "English: Brown Corpus (Press)": lambda: brown.words(
        categories=["news", "editorial", "reviews"]
    ),
    "English: Brown Corpus (Religion)": lambda: brown.words(categories="religion"),
    "English: Brown Corpus (Learned)": lambda: brown.words(categories="learned"),
    "English: Brown Corpus (Science Fiction)": lambda: brown.words(
        categories="science_fiction"
    ),
    "English: Brown Corpus (Romance)": lambda: brown.words(categories="romance"),
    "English: Brown Corpus (Humor)": lambda: brown.words(categories="humor"),
    "English: NPS Chat Corpus": lambda: nps_chat.words(),
    "English: Wall Street Journal Corpus": lambda: treebank.words(),
    "Chinese: Sinica Corpus": lambda: sinica_treebank.words(),
    "Dutch: Alpino Corpus": lambda: alpino.words(),
    "Hindi: Indian Languages Corpus": lambda: indian.words(files="hindi.pos"),
    "Portuguese: Floresta Corpus (Portugal)": lambda: floresta.words(),
    "Portuguese: MAC-MORPHO Corpus (Brazil)": lambda: mac_morpho.words(),
    "Portuguese: Machado Corpus (Brazil)": lambda: machado.words(),
    "Spanish: CESS-ESP Corpus": lambda: cess_esp.words(),
}


class CollocationsView:
    _BACKGROUND_COLOUR = "#FFF"  # white

    def __init__(self):
        self.queue = q.Queue()
        self.model = CollocationsModel(self.queue)
        self.top = Tk()
        self._init_top(self.top)
        self._init_menubar()
        self._init_widgets(self.top)
        self.load_corpus(self.model.DEFAULT_CORPUS)
        self.after = self.top.after(POLL_INTERVAL, self._poll)

    def _init_top(self, top):
        top.geometry("550x650+50+50")
        top.title("NLTK Collocations List")
        top.bind("<Control-q>", self.destroy)
        top.protocol("WM_DELETE_WINDOW", self.destroy)
        top.minsize(550, 650)

    def _init_widgets(self, parent):
        self.main_frame = Frame(
            parent, dict(background=self._BACKGROUND_COLOUR, padx=1, pady=1, border=1)
        )
        self._init_corpus_select(self.main_frame)
        self._init_results_box(self.main_frame)
        self._init_paging(self.main_frame)
        self._init_status(self.main_frame)
        self.main_frame.pack(fill="both", expand=True)

    def _init_corpus_select(self, parent):
        innerframe = Frame(parent, background=self._BACKGROUND_COLOUR)
        self.var = StringVar(innerframe)
        self.var.set(self.model.DEFAULT_CORPUS)
        Label(
            innerframe,
            justify=LEFT,
            text=" Corpus: ",
            background=self._BACKGROUND_COLOUR,
            padx=2,
            pady=1,
            border=0,
        ).pack(side="left")

        other_corpora = list(self.model.CORPORA.keys()).remove(
            self.model.DEFAULT_CORPUS
        )
        om = OptionMenu(
            innerframe,
            self.var,
            self.model.DEFAULT_CORPUS,
            command=self.corpus_selected,
            *self.model.non_default_corpora()
        )
        om["borderwidth"] = 0
        om["highlightthickness"] = 1
        om.pack(side="left")
        innerframe.pack(side="top", fill="x", anchor="n")

    def _init_status(self, parent):
        self.status = Label(
            parent,
            justify=LEFT,
            relief=SUNKEN,
            background=self._BACKGROUND_COLOUR,
            border=0,
            padx=1,
            pady=0,
        )
        self.status.pack(side="top", anchor="sw")

    def _init_menubar(self):
        self._result_size = IntVar(self.top)
        menubar = Menu(self.top)

        filemenu = Menu(menubar, tearoff=0, borderwidth=0)
        filemenu.add_command(
            label="Exit", underline=1, command=self.destroy, accelerator="Ctrl-q"
        )
        menubar.add_cascade(label="File", underline=0, menu=filemenu)

        editmenu = Menu(menubar, tearoff=0)
        rescntmenu = Menu(editmenu, tearoff=0)
        rescntmenu.add_radiobutton(
            label="20",
            variable=self._result_size,
            underline=0,
            value=20,
            command=self.set_result_size,
        )
        rescntmenu.add_radiobutton(
            label="50",
            variable=self._result_size,
            underline=0,
            value=50,
            command=self.set_result_size,
        )
        rescntmenu.add_radiobutton(
            label="100",
            variable=self._result_size,
            underline=0,
            value=100,
            command=self.set_result_size,
        )
        rescntmenu.invoke(1)
        editmenu.add_cascade(label="Result Count", underline=0, menu=rescntmenu)

        menubar.add_cascade(label="Edit", underline=0, menu=editmenu)
        self.top.config(menu=menubar)

    def set_result_size(self, **kwargs):
        self.model.result_count = self._result_size.get()

    def _init_results_box(self, parent):
        innerframe = Frame(parent)
        i1 = Frame(innerframe)
        i2 = Frame(innerframe)
        vscrollbar = Scrollbar(i1, borderwidth=1)
        hscrollbar = Scrollbar(i2, borderwidth=1, orient="horiz")
        self.results_box = Text(
            i1,
            font=Font(family="courier", size="16"),
            state="disabled",
            borderwidth=1,
            yscrollcommand=vscrollbar.set,
            xscrollcommand=hscrollbar.set,
            wrap="none",
            width="40",
            height="20",
            exportselection=1,
        )
        self.results_box.pack(side="left", fill="both", expand=True)
        vscrollbar.pack(side="left", fill="y", anchor="e")
        vscrollbar.config(command=self.results_box.yview)
        hscrollbar.pack(side="left", fill="x", expand=True, anchor="w")
        hscrollbar.config(command=self.results_box.xview)
        # there is no other way of avoiding the overlap of scrollbars while using pack layout manager!!!
        Label(i2, text="   ", background=self._BACKGROUND_COLOUR).pack(
            side="left", anchor="e"
        )
        i1.pack(side="top", fill="both", expand=True, anchor="n")
        i2.pack(side="bottom", fill="x", anchor="s")
        innerframe.pack(side="top", fill="both", expand=True)

    def _init_paging(self, parent):
        innerframe = Frame(parent, background=self._BACKGROUND_COLOUR)
        self.prev = prev = Button(
            innerframe,
            text="Previous",
            command=self.previous,
            width="10",
            borderwidth=1,
            highlightthickness=1,
            state="disabled",
        )
        prev.pack(side="left", anchor="center")
        self.next = next = Button(
            innerframe,
            text="Next",
            command=self.__next__,
            width="10",
            borderwidth=1,
            highlightthickness=1,
            state="disabled",
        )
        next.pack(side="right", anchor="center")
        innerframe.pack(side="top", fill="y")
        self.reset_current_page()

    def reset_current_page(self):
        self.current_page = -1

    def _poll(self):
        try:
            event = self.queue.get(block=False)
        except q.Empty:
            pass
        else:
            if event == CORPUS_LOADED_EVENT:
                self.handle_corpus_loaded(event)
            elif event == ERROR_LOADING_CORPUS_EVENT:
                self.handle_error_loading_corpus(event)
        self.after = self.top.after(POLL_INTERVAL, self._poll)

    def handle_error_loading_corpus(self, event):
        self.status["text"] = "Error in loading " + self.var.get()
        self.unfreeze_editable()
        self.clear_results_box()
        self.freeze_editable()
        self.reset_current_page()

    def handle_corpus_loaded(self, event):
        self.status["text"] = self.var.get() + " is loaded"
        self.unfreeze_editable()
        self.clear_results_box()
        self.reset_current_page()
        # self.next()
        collocations = self.model.next(self.current_page + 1)
        self.write_results(collocations)
        self.current_page += 1

    def corpus_selected(self, *args):
        new_selection = self.var.get()
        self.load_corpus(new_selection)

    def previous(self):
        self.freeze_editable()
        collocations = self.model.prev(self.current_page - 1)
        self.current_page = self.current_page - 1
        self.clear_results_box()
        self.write_results(collocations)
        self.unfreeze_editable()

    def __next__(self):
        self.freeze_editable()
        collocations = self.model.next(self.current_page + 1)
        self.clear_results_box()
        self.write_results(collocations)
        self.current_page += 1
        self.unfreeze_editable()

    def load_corpus(self, selection):
        if self.model.selected_corpus != selection:
            self.status["text"] = "Loading " + selection + "..."
            self.freeze_editable()
            self.model.load_corpus(selection)

    def freeze_editable(self):
        self.prev["state"] = "disabled"
        self.next["state"] = "disabled"

    def clear_results_box(self):
        self.results_box["state"] = "normal"
        self.results_box.delete("1.0", END)
        self.results_box["state"] = "disabled"

    def fire_event(self, event):
        # Firing an event so that rendering of widgets happen in the mainloop thread
        self.top.event_generate(event, when="tail")

    def destroy(self, *e):
        if self.top is None:
            return
        self.top.after_cancel(self.after)
        self.top.destroy()
        self.top = None

    def mainloop(self, *args, **kwargs):
        if in_idle():
            return
        self.top.mainloop(*args, **kwargs)

    def unfreeze_editable(self):
        self.set_paging_button_states()

    def set_paging_button_states(self):
        if self.current_page == -1 or self.current_page == 0:
            self.prev["state"] = "disabled"
        else:
            self.prev["state"] = "normal"
        if self.model.is_last_page(self.current_page):
            self.next["state"] = "disabled"
        else:
            self.next["state"] = "normal"

    def write_results(self, results):
        self.results_box["state"] = "normal"
        row = 1
        for each in results:
            self.results_box.insert(str(row) + ".0", each[0] + " " + each[1] + "\n")
            row += 1
        self.results_box["state"] = "disabled"


class CollocationsModel:
    def __init__(self, queue):
        self.result_count = None
        self.selected_corpus = None
        self.collocations = None
        self.CORPORA = _CORPORA
        self.DEFAULT_CORPUS = _DEFAULT
        self.queue = queue
        self.reset_results()

    def reset_results(self):
        self.result_pages = []
        self.results_returned = 0

    def load_corpus(self, name):
        self.selected_corpus = name
        self.collocations = None
        runner_thread = self.LoadCorpus(name, self)
        runner_thread.start()
        self.reset_results()

    def non_default_corpora(self):
        copy = []
        copy.extend(list(self.CORPORA.keys()))
        copy.remove(self.DEFAULT_CORPUS)
        copy.sort()
        return copy

    def is_last_page(self, number):
        if number < len(self.result_pages):
            return False
        return self.results_returned + (
            number - len(self.result_pages)
        ) * self.result_count >= len(self.collocations)

    def next(self, page):
        if (len(self.result_pages) - 1) < page:
            for i in range(page - (len(self.result_pages) - 1)):
                self.result_pages.append(
                    self.collocations[
                        self.results_returned : self.results_returned
                        + self.result_count
                    ]
                )
                self.results_returned += self.result_count
        return self.result_pages[page]

    def prev(self, page):
        if page == -1:
            return []
        return self.result_pages[page]

    class LoadCorpus(threading.Thread):
        def __init__(self, name, model):
            threading.Thread.__init__(self)
            self.model, self.name = model, name

        def run(self):
            try:
                words = self.model.CORPORA[self.name]()
                from operator import itemgetter

                text = [w for w in words if len(w) > 2]
                fd = FreqDist(tuple(text[i : i + 2]) for i in range(len(text) - 1))
                vocab = FreqDist(text)
                scored = [
                    ((w1, w2), fd[(w1, w2)] ** 3 / (vocab[w1] * vocab[w2]))
                    for w1, w2 in fd
                ]
                scored.sort(key=itemgetter(1), reverse=True)
                self.model.collocations = list(map(itemgetter(0), scored))
                self.model.queue.put(CORPUS_LOADED_EVENT)
            except Exception as e:
                print(e)
                self.model.queue.put(ERROR_LOADING_CORPUS_EVENT)


# def collocations():
#    colloc_strings = [w1 + ' ' + w2 for w1, w2 in self._collocations[:num]]


def app():
    c = CollocationsView()
    c.mainloop()


if __name__ == "__main__":
    app()

__all__ = ["app"]
