# Natural Language Toolkit: Concordance Application
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Sumukh Ghodke <sghodke@csse.unimelb.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import queue as q
import re
import threading
from tkinter import (
    END,
    LEFT,
    SUNKEN,
    Button,
    Entry,
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
    nps_chat,
    sinica_treebank,
    treebank,
)
from nltk.draw.util import ShowText
from nltk.util import in_idle

WORD_OR_TAG = "[^/ ]+"
BOUNDARY = r"\b"

CORPUS_LOADED_EVENT = "<<CL_EVENT>>"
SEARCH_TERMINATED_EVENT = "<<ST_EVENT>>"
SEARCH_ERROR_EVENT = "<<SE_EVENT>>"
ERROR_LOADING_CORPUS_EVENT = "<<ELC_EVENT>>"

POLL_INTERVAL = 50

# NB All corpora must be specified in a lambda expression so as not to be
# loaded when the module is imported.

_DEFAULT = "English: Brown Corpus (Humor, simplified)"
_CORPORA = {
    "Catalan: CESS-CAT Corpus (simplified)": lambda: cess_cat.tagged_sents(
        tagset="universal"
    ),
    "English: Brown Corpus": lambda: brown.tagged_sents(),
    "English: Brown Corpus (simplified)": lambda: brown.tagged_sents(
        tagset="universal"
    ),
    "English: Brown Corpus (Press, simplified)": lambda: brown.tagged_sents(
        categories=["news", "editorial", "reviews"], tagset="universal"
    ),
    "English: Brown Corpus (Religion, simplified)": lambda: brown.tagged_sents(
        categories="religion", tagset="universal"
    ),
    "English: Brown Corpus (Learned, simplified)": lambda: brown.tagged_sents(
        categories="learned", tagset="universal"
    ),
    "English: Brown Corpus (Science Fiction, simplified)": lambda: brown.tagged_sents(
        categories="science_fiction", tagset="universal"
    ),
    "English: Brown Corpus (Romance, simplified)": lambda: brown.tagged_sents(
        categories="romance", tagset="universal"
    ),
    "English: Brown Corpus (Humor, simplified)": lambda: brown.tagged_sents(
        categories="humor", tagset="universal"
    ),
    "English: NPS Chat Corpus": lambda: nps_chat.tagged_posts(),
    "English: NPS Chat Corpus (simplified)": lambda: nps_chat.tagged_posts(
        tagset="universal"
    ),
    "English: Wall Street Journal Corpus": lambda: treebank.tagged_sents(),
    "English: Wall Street Journal Corpus (simplified)": lambda: treebank.tagged_sents(
        tagset="universal"
    ),
    "Chinese: Sinica Corpus": lambda: sinica_treebank.tagged_sents(),
    "Chinese: Sinica Corpus (simplified)": lambda: sinica_treebank.tagged_sents(
        tagset="universal"
    ),
    "Dutch: Alpino Corpus": lambda: alpino.tagged_sents(),
    "Dutch: Alpino Corpus (simplified)": lambda: alpino.tagged_sents(
        tagset="universal"
    ),
    "Hindi: Indian Languages Corpus": lambda: indian.tagged_sents(files="hindi.pos"),
    "Hindi: Indian Languages Corpus (simplified)": lambda: indian.tagged_sents(
        files="hindi.pos", tagset="universal"
    ),
    "Portuguese: Floresta Corpus (Portugal)": lambda: floresta.tagged_sents(),
    "Portuguese: Floresta Corpus (Portugal, simplified)": lambda: floresta.tagged_sents(
        tagset="universal"
    ),
    "Portuguese: MAC-MORPHO Corpus (Brazil)": lambda: mac_morpho.tagged_sents(),
    "Portuguese: MAC-MORPHO Corpus (Brazil, simplified)": lambda: mac_morpho.tagged_sents(
        tagset="universal"
    ),
    "Spanish: CESS-ESP Corpus (simplified)": lambda: cess_esp.tagged_sents(
        tagset="universal"
    ),
}


class ConcordanceSearchView:
    _BACKGROUND_COLOUR = "#FFF"  # white

    # Colour of highlighted results
    _HIGHLIGHT_WORD_COLOUR = "#F00"  # red
    _HIGHLIGHT_WORD_TAG = "HL_WRD_TAG"

    _HIGHLIGHT_LABEL_COLOUR = "#C0C0C0"  # dark grey
    _HIGHLIGHT_LABEL_TAG = "HL_LBL_TAG"

    # Percentage of text left of the scrollbar position
    _FRACTION_LEFT_TEXT = 0.30

    def __init__(self):
        self.queue = q.Queue()
        self.model = ConcordanceSearchModel(self.queue)
        self.top = Tk()
        self._init_top(self.top)
        self._init_menubar()
        self._init_widgets(self.top)
        self.load_corpus(self.model.DEFAULT_CORPUS)
        self.after = self.top.after(POLL_INTERVAL, self._poll)

    def _init_top(self, top):
        top.geometry("950x680+50+50")
        top.title("NLTK Concordance Search")
        top.bind("<Control-q>", self.destroy)
        top.protocol("WM_DELETE_WINDOW", self.destroy)
        top.minsize(950, 680)

    def _init_widgets(self, parent):
        self.main_frame = Frame(
            parent, dict(background=self._BACKGROUND_COLOUR, padx=1, pady=1, border=1)
        )
        self._init_corpus_select(self.main_frame)
        self._init_query_box(self.main_frame)
        self._init_results_box(self.main_frame)
        self._init_paging(self.main_frame)
        self._init_status(self.main_frame)
        self.main_frame.pack(fill="both", expand=True)

    def _init_menubar(self):
        self._result_size = IntVar(self.top)
        self._cntx_bf_len = IntVar(self.top)
        self._cntx_af_len = IntVar(self.top)
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

        cntxmenu = Menu(editmenu, tearoff=0)
        cntxbfmenu = Menu(cntxmenu, tearoff=0)
        cntxbfmenu.add_radiobutton(
            label="60 characters",
            variable=self._cntx_bf_len,
            underline=0,
            value=60,
            command=self.set_cntx_bf_len,
        )
        cntxbfmenu.add_radiobutton(
            label="80 characters",
            variable=self._cntx_bf_len,
            underline=0,
            value=80,
            command=self.set_cntx_bf_len,
        )
        cntxbfmenu.add_radiobutton(
            label="100 characters",
            variable=self._cntx_bf_len,
            underline=0,
            value=100,
            command=self.set_cntx_bf_len,
        )
        cntxbfmenu.invoke(1)
        cntxmenu.add_cascade(label="Before", underline=0, menu=cntxbfmenu)

        cntxafmenu = Menu(cntxmenu, tearoff=0)
        cntxafmenu.add_radiobutton(
            label="70 characters",
            variable=self._cntx_af_len,
            underline=0,
            value=70,
            command=self.set_cntx_af_len,
        )
        cntxafmenu.add_radiobutton(
            label="90 characters",
            variable=self._cntx_af_len,
            underline=0,
            value=90,
            command=self.set_cntx_af_len,
        )
        cntxafmenu.add_radiobutton(
            label="110 characters",
            variable=self._cntx_af_len,
            underline=0,
            value=110,
            command=self.set_cntx_af_len,
        )
        cntxafmenu.invoke(1)
        cntxmenu.add_cascade(label="After", underline=0, menu=cntxafmenu)

        editmenu.add_cascade(label="Context", underline=0, menu=cntxmenu)

        menubar.add_cascade(label="Edit", underline=0, menu=editmenu)

        self.top.config(menu=menubar)

    def set_result_size(self, **kwargs):
        self.model.result_count = self._result_size.get()

    def set_cntx_af_len(self, **kwargs):
        self._char_after = self._cntx_af_len.get()

    def set_cntx_bf_len(self, **kwargs):
        self._char_before = self._cntx_bf_len.get()

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

    def _init_query_box(self, parent):
        innerframe = Frame(parent, background=self._BACKGROUND_COLOUR)
        another = Frame(innerframe, background=self._BACKGROUND_COLOUR)
        self.query_box = Entry(another, width=60)
        self.query_box.pack(side="left", fill="x", pady=25, anchor="center")
        self.search_button = Button(
            another,
            text="Search",
            command=self.search,
            borderwidth=1,
            highlightthickness=1,
        )
        self.search_button.pack(side="left", fill="x", pady=25, anchor="center")
        self.query_box.bind("<KeyPress-Return>", self.search_enter_keypress_handler)
        another.pack()
        innerframe.pack(side="top", fill="x", anchor="n")

    def search_enter_keypress_handler(self, *event):
        self.search()

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
        self.results_box.tag_config(
            self._HIGHLIGHT_WORD_TAG, foreground=self._HIGHLIGHT_WORD_COLOUR
        )
        self.results_box.tag_config(
            self._HIGHLIGHT_LABEL_TAG, foreground=self._HIGHLIGHT_LABEL_COLOUR
        )
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
        self.current_page = 0

    def previous(self):
        self.clear_results_box()
        self.freeze_editable()
        self.model.prev(self.current_page - 1)

    def __next__(self):
        self.clear_results_box()
        self.freeze_editable()
        self.model.next(self.current_page + 1)

    def about(self, *e):
        ABOUT = "NLTK Concordance Search Demo\n"
        TITLE = "About: NLTK Concordance Search Demo"
        try:
            from tkinter.messagebox import Message

            Message(message=ABOUT, title=TITLE, parent=self.main_frame).show()
        except:
            ShowText(self.top, TITLE, ABOUT)

    def _bind_event_handlers(self):
        self.top.bind(CORPUS_LOADED_EVENT, self.handle_corpus_loaded)
        self.top.bind(SEARCH_TERMINATED_EVENT, self.handle_search_terminated)
        self.top.bind(SEARCH_ERROR_EVENT, self.handle_search_error)
        self.top.bind(ERROR_LOADING_CORPUS_EVENT, self.handle_error_loading_corpus)

    def _poll(self):
        try:
            event = self.queue.get(block=False)
        except q.Empty:
            pass
        else:
            if event == CORPUS_LOADED_EVENT:
                self.handle_corpus_loaded(event)
            elif event == SEARCH_TERMINATED_EVENT:
                self.handle_search_terminated(event)
            elif event == SEARCH_ERROR_EVENT:
                self.handle_search_error(event)
            elif event == ERROR_LOADING_CORPUS_EVENT:
                self.handle_error_loading_corpus(event)
        self.after = self.top.after(POLL_INTERVAL, self._poll)

    def handle_error_loading_corpus(self, event):
        self.status["text"] = "Error in loading " + self.var.get()
        self.unfreeze_editable()
        self.clear_all()
        self.freeze_editable()

    def handle_corpus_loaded(self, event):
        self.status["text"] = self.var.get() + " is loaded"
        self.unfreeze_editable()
        self.clear_all()
        self.query_box.focus_set()

    def handle_search_terminated(self, event):
        # todo: refactor the model such that it is less state sensitive
        results = self.model.get_results()
        self.write_results(results)
        self.status["text"] = ""
        if len(results) == 0:
            self.status["text"] = "No results found for " + self.model.query
        else:
            self.current_page = self.model.last_requested_page
        self.unfreeze_editable()
        self.results_box.xview_moveto(self._FRACTION_LEFT_TEXT)

    def handle_search_error(self, event):
        self.status["text"] = "Error in query " + self.model.query
        self.unfreeze_editable()

    def corpus_selected(self, *args):
        new_selection = self.var.get()
        self.load_corpus(new_selection)

    def load_corpus(self, selection):
        if self.model.selected_corpus != selection:
            self.status["text"] = "Loading " + selection + "..."
            self.freeze_editable()
            self.model.load_corpus(selection)

    def search(self):
        self.current_page = 0
        self.clear_results_box()
        self.model.reset_results()
        query = self.query_box.get()
        if len(query.strip()) == 0:
            return
        self.status["text"] = "Searching for " + query
        self.freeze_editable()
        self.model.search(query, self.current_page + 1)

    def write_results(self, results):
        self.results_box["state"] = "normal"
        row = 1
        for each in results:
            sent, pos1, pos2 = each[0].strip(), each[1], each[2]
            if len(sent) != 0:
                if pos1 < self._char_before:
                    sent, pos1, pos2 = self.pad(sent, pos1, pos2)
                sentence = sent[pos1 - self._char_before : pos1 + self._char_after]
                if not row == len(results):
                    sentence += "\n"
                self.results_box.insert(str(row) + ".0", sentence)
                word_markers, label_markers = self.words_and_labels(sent, pos1, pos2)
                for marker in word_markers:
                    self.results_box.tag_add(
                        self._HIGHLIGHT_WORD_TAG,
                        str(row) + "." + str(marker[0]),
                        str(row) + "." + str(marker[1]),
                    )
                for marker in label_markers:
                    self.results_box.tag_add(
                        self._HIGHLIGHT_LABEL_TAG,
                        str(row) + "." + str(marker[0]),
                        str(row) + "." + str(marker[1]),
                    )
                row += 1
        self.results_box["state"] = "disabled"

    def words_and_labels(self, sentence, pos1, pos2):
        search_exp = sentence[pos1:pos2]
        words, labels = [], []
        labeled_words = search_exp.split(" ")
        index = 0
        for each in labeled_words:
            if each == "":
                index += 1
            else:
                word, label = each.split("/")
                words.append(
                    (self._char_before + index, self._char_before + index + len(word))
                )
                index += len(word) + 1
                labels.append(
                    (self._char_before + index, self._char_before + index + len(label))
                )
                index += len(label)
            index += 1
        return words, labels

    def pad(self, sent, hstart, hend):
        if hstart >= self._char_before:
            return sent, hstart, hend
        d = self._char_before - hstart
        sent = "".join([" "] * d) + sent
        return sent, hstart + d, hend + d

    def destroy(self, *e):
        if self.top is None:
            return
        self.top.after_cancel(self.after)
        self.top.destroy()
        self.top = None

    def clear_all(self):
        self.query_box.delete(0, END)
        self.model.reset_query()
        self.clear_results_box()

    def clear_results_box(self):
        self.results_box["state"] = "normal"
        self.results_box.delete("1.0", END)
        self.results_box["state"] = "disabled"

    def freeze_editable(self):
        self.query_box["state"] = "disabled"
        self.search_button["state"] = "disabled"
        self.prev["state"] = "disabled"
        self.next["state"] = "disabled"

    def unfreeze_editable(self):
        self.query_box["state"] = "normal"
        self.search_button["state"] = "normal"
        self.set_paging_button_states()

    def set_paging_button_states(self):
        if self.current_page == 0 or self.current_page == 1:
            self.prev["state"] = "disabled"
        else:
            self.prev["state"] = "normal"
        if self.model.has_more_pages(self.current_page):
            self.next["state"] = "normal"
        else:
            self.next["state"] = "disabled"

    def fire_event(self, event):
        # Firing an event so that rendering of widgets happen in the mainloop thread
        self.top.event_generate(event, when="tail")

    def mainloop(self, *args, **kwargs):
        if in_idle():
            return
        self.top.mainloop(*args, **kwargs)


class ConcordanceSearchModel:
    def __init__(self, queue):
        self.queue = queue
        self.CORPORA = _CORPORA
        self.DEFAULT_CORPUS = _DEFAULT
        self.selected_corpus = None
        self.reset_query()
        self.reset_results()
        self.result_count = None
        self.last_sent_searched = 0

    def non_default_corpora(self):
        copy = []
        copy.extend(list(self.CORPORA.keys()))
        copy.remove(self.DEFAULT_CORPUS)
        copy.sort()
        return copy

    def load_corpus(self, name):
        self.selected_corpus = name
        self.tagged_sents = []
        runner_thread = self.LoadCorpus(name, self)
        runner_thread.start()

    def search(self, query, page):
        self.query = query
        self.last_requested_page = page
        self.SearchCorpus(self, page, self.result_count).start()

    def next(self, page):
        self.last_requested_page = page
        if len(self.results) < page:
            self.search(self.query, page)
        else:
            self.queue.put(SEARCH_TERMINATED_EVENT)

    def prev(self, page):
        self.last_requested_page = page
        self.queue.put(SEARCH_TERMINATED_EVENT)

    def reset_results(self):
        self.last_sent_searched = 0
        self.results = []
        self.last_page = None

    def reset_query(self):
        self.query = None

    def set_results(self, page, resultset):
        self.results.insert(page - 1, resultset)

    def get_results(self):
        return self.results[self.last_requested_page - 1]

    def has_more_pages(self, page):
        if self.results == [] or self.results[0] == []:
            return False
        if self.last_page is None:
            return True
        return page < self.last_page

    class LoadCorpus(threading.Thread):
        def __init__(self, name, model):
            threading.Thread.__init__(self)
            self.model, self.name = model, name

        def run(self):
            try:
                ts = self.model.CORPORA[self.name]()
                self.model.tagged_sents = [
                    " ".join(w + "/" + t for (w, t) in sent) for sent in ts
                ]
                self.model.queue.put(CORPUS_LOADED_EVENT)
            except Exception as e:
                print(e)
                self.model.queue.put(ERROR_LOADING_CORPUS_EVENT)

    class SearchCorpus(threading.Thread):
        def __init__(self, model, page, count):
            self.model, self.count, self.page = model, count, page
            threading.Thread.__init__(self)

        def run(self):
            q = self.processed_query()
            sent_pos, i, sent_count = [], 0, 0
            for sent in self.model.tagged_sents[self.model.last_sent_searched :]:
                try:
                    m = re.search(q, sent)
                except re.error:
                    self.model.reset_results()
                    self.model.queue.put(SEARCH_ERROR_EVENT)
                    return
                if m:
                    sent_pos.append((sent, m.start(), m.end()))
                    i += 1
                    if i > self.count:
                        self.model.last_sent_searched += sent_count - 1
                        break
                sent_count += 1
            if self.count >= len(sent_pos):
                self.model.last_sent_searched += sent_count - 1
                self.model.last_page = self.page
                self.model.set_results(self.page, sent_pos)
            else:
                self.model.set_results(self.page, sent_pos[:-1])
            self.model.queue.put(SEARCH_TERMINATED_EVENT)

        def processed_query(self):
            new = []
            for term in self.model.query.split():
                term = re.sub(r"\.", r"[^/ ]", term)
                if re.match("[A-Z]+$", term):
                    new.append(BOUNDARY + WORD_OR_TAG + "/" + term + BOUNDARY)
                elif "/" in term:
                    new.append(BOUNDARY + term + BOUNDARY)
                else:
                    new.append(BOUNDARY + term + "/" + WORD_OR_TAG + BOUNDARY)
            return " ".join(new)


def app():
    d = ConcordanceSearchView()
    d.mainloop()


if __name__ == "__main__":
    app()

__all__ = ["app"]
