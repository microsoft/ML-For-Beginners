# Natural Language Toolkit: WordNet Browser Application
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Jussi Salmela <jtsalmela@users.sourceforge.net>
#         Paul Bone <pbone@students.csse.unimelb.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A WordNet Browser application which launches the default browser
(if it is not already running) and opens a new tab with a connection
to http://localhost:port/ .  It also starts an HTTP server on the
specified port and begins serving browser requests.  The default
port is 8000.  (For command-line help, run "python wordnet -h")
This application requires that the user's web browser supports
Javascript.

BrowServer is a server for browsing the NLTK Wordnet database It first
launches a browser client to be used for browsing and then starts
serving the requests of that and maybe other clients

Usage::

    browserver.py -h
    browserver.py [-s] [-p <port>]

Options::

    -h or --help
        Display this help message.

    -l <file> or --log-file <file>
        Logs messages to the given file, If this option is not specified
        messages are silently dropped.

    -p <port> or --port <port>
        Run the web server on this TCP port, defaults to 8000.

    -s or --server-mode
        Do not start a web browser, and do not allow a user to
        shutdown the server through the web interface.
"""
# TODO: throughout this package variable names and docstrings need
# modifying to be compliant with NLTK's coding standards.  Tests also
# need to be develop to ensure this continues to work in the face of
# changes to other NLTK packages.

import base64
import copy
import getopt
import io
import os
import pickle
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

# Allow this program to run inside the NLTK source tree.
from sys import argv
from urllib.parse import unquote_plus

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset

firstClient = True

# True if we're not also running a web browser.  The value f server_mode
# gets set by demo().
server_mode = None

# If set this is a file object for writing log messages.
logfile = None


class MyServerHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_head()

    def do_GET(self):
        global firstClient
        sp = self.path[1:]
        if unquote_plus(sp) == "SHUTDOWN THE SERVER":
            if server_mode:
                page = "Server must be killed with SIGTERM."
                type = "text/plain"
            else:
                print("Server shutting down!")
                os._exit(0)

        elif sp == "":  # First request.
            type = "text/html"
            if not server_mode and firstClient:
                firstClient = False
                page = get_static_index_page(True)
            else:
                page = get_static_index_page(False)
            word = "green"

        elif sp.endswith(".html"):  # Trying to fetch a HTML file TODO:
            type = "text/html"
            usp = unquote_plus(sp)
            if usp == "NLTK Wordnet Browser Database Info.html":
                word = "* Database Info *"
                if os.path.isfile(usp):
                    with open(usp) as infile:
                        page = infile.read()
                else:
                    page = (
                        (html_header % word) + "<p>The database info file:"
                        "<p><b>"
                        + usp
                        + "</b>"
                        + "<p>was not found. Run this:"
                        + "<p><b>python dbinfo_html.py</b>"
                        + "<p>to produce it."
                        + html_trailer
                    )
            else:
                # Handle files here.
                word = sp
                try:
                    page = get_static_page_by_path(usp)
                except FileNotFoundError:
                    page = "Internal error: Path for static page '%s' is unknown" % usp
                    # Set type to plain to prevent XSS by printing the path as HTML
                    type = "text/plain"
        elif sp.startswith("search"):
            # This doesn't seem to work with MWEs.
            type = "text/html"
            parts = (sp.split("?")[1]).split("&")
            word = [
                p.split("=")[1].replace("+", " ")
                for p in parts
                if p.startswith("nextWord")
            ][0]
            page, word = page_from_word(word)
        elif sp.startswith("lookup_"):
            # TODO add a variation of this that takes a non ecoded word or MWE.
            type = "text/html"
            sp = sp[len("lookup_") :]
            page, word = page_from_href(sp)
        elif sp == "start_page":
            # if this is the first request we should display help
            # information, and possibly set a default word.
            type = "text/html"
            page, word = page_from_word("wordnet")
        else:
            type = "text/plain"
            page = "Could not parse request: '%s'" % sp

        # Send result.
        self.send_head(type)
        self.wfile.write(page.encode("utf8"))

    def send_head(self, type=None):
        self.send_response(200)
        self.send_header("Content-type", type)
        self.end_headers()

    def log_message(self, format, *args):
        global logfile

        if logfile:
            logfile.write(
                "%s - - [%s] %s\n"
                % (self.address_string(), self.log_date_time_string(), format % args)
            )


def get_unique_counter_from_url(sp):
    """
    Extract the unique counter from the URL if it has one.  Otherwise return
    null.
    """
    pos = sp.rfind("%23")
    if pos != -1:
        return int(sp[(pos + 3) :])
    else:
        return None


def wnb(port=8000, runBrowser=True, logfilename=None):
    """
    Run NLTK Wordnet Browser Server.

    :param port: The port number for the server to listen on, defaults to
                 8000
    :type  port: int

    :param runBrowser: True to start a web browser and point it at the web
                       server.
    :type  runBrowser: bool
    """
    # The webbrowser module is unpredictable, typically it blocks if it uses
    # a console web browser, and doesn't block if it uses a GUI webbrowser,
    # so we need to force it to have a clear correct behaviour.
    #
    # Normally the server should run for as long as the user wants. they
    # should idealy be able to control this from the UI by closing the
    # window or tab.  Second best would be clicking a button to say
    # 'Shutdown' that first shutsdown the server and closes the window or
    # tab, or exits the text-mode browser.  Both of these are unfreasable.
    #
    # The next best alternative is to start the server, have it close when
    # it receives SIGTERM (default), and run the browser as well.  The user
    # may have to shutdown both programs.
    #
    # Since webbrowser may block, and the webserver will block, we must run
    # them in separate threads.
    #
    global server_mode, logfile
    server_mode = not runBrowser

    # Setup logging.
    if logfilename:
        try:
            logfile = open(logfilename, "a", 1)  # 1 means 'line buffering'
        except OSError as e:
            sys.stderr.write("Couldn't open %s for writing: %s", logfilename, e)
            sys.exit(1)
    else:
        logfile = None

    # Compute URL and start web browser
    url = "http://localhost:" + str(port)

    server_ready = None
    browser_thread = None

    if runBrowser:
        server_ready = threading.Event()
        browser_thread = startBrowser(url, server_ready)

    # Start the server.
    server = HTTPServer(("", port), MyServerHandler)
    if logfile:
        logfile.write("NLTK Wordnet browser server running serving: %s\n" % url)
    if runBrowser:
        server_ready.set()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    if runBrowser:
        browser_thread.join()

    if logfile:
        logfile.close()


def startBrowser(url, server_ready):
    def run():
        server_ready.wait()
        time.sleep(1)  # Wait a little bit more, there's still the chance of
        # a race condition.
        webbrowser.open(url, new=2, autoraise=1)

    t = threading.Thread(target=run)
    t.start()
    return t


#####################################################################
# Utilities
#####################################################################


"""
WordNet Browser Utilities.

This provides a backend to both wxbrowse and browserver.py.
"""

################################################################################
#
# Main logic for wordnet browser.
#

# This is wrapped inside a function since wn is only available if the
# WordNet corpus is installed.
def _pos_tuples():
    return [
        (wn.NOUN, "N", "noun"),
        (wn.VERB, "V", "verb"),
        (wn.ADJ, "J", "adj"),
        (wn.ADV, "R", "adv"),
    ]


def _pos_match(pos_tuple):
    """
    This function returns the complete pos tuple for the partial pos
    tuple given to it.  It attempts to match it against the first
    non-null component of the given pos tuple.
    """
    if pos_tuple[0] == "s":
        pos_tuple = ("a", pos_tuple[1], pos_tuple[2])
    for n, x in enumerate(pos_tuple):
        if x is not None:
            break
    for pt in _pos_tuples():
        if pt[n] == pos_tuple[n]:
            return pt
    return None


HYPONYM = 0
HYPERNYM = 1
CLASS_REGIONAL = 2
PART_HOLONYM = 3
PART_MERONYM = 4
ATTRIBUTE = 5
SUBSTANCE_HOLONYM = 6
SUBSTANCE_MERONYM = 7
MEMBER_HOLONYM = 8
MEMBER_MERONYM = 9
VERB_GROUP = 10
INSTANCE_HYPONYM = 12
INSTANCE_HYPERNYM = 13
CAUSE = 14
ALSO_SEE = 15
SIMILAR = 16
ENTAILMENT = 17
ANTONYM = 18
FRAMES = 19
PERTAINYM = 20

CLASS_CATEGORY = 21
CLASS_USAGE = 22
CLASS_REGIONAL = 23
CLASS_USAGE = 24
CLASS_CATEGORY = 11

DERIVATIONALLY_RELATED_FORM = 25

INDIRECT_HYPERNYMS = 26


def lemma_property(word, synset, func):
    def flattern(l):
        if l == []:
            return []
        else:
            return l[0] + flattern(l[1:])

    return flattern([func(l) for l in synset.lemmas() if l.name == word])


def rebuild_tree(orig_tree):
    node = orig_tree[0]
    children = orig_tree[1:]
    return (node, [rebuild_tree(t) for t in children])


def get_relations_data(word, synset):
    """
    Get synset relations data for a synset.  Note that this doesn't
    yet support things such as full hyponym vs direct hyponym.
    """
    if synset.pos() == wn.NOUN:
        return (
            (HYPONYM, "Hyponyms", synset.hyponyms()),
            (INSTANCE_HYPONYM, "Instance hyponyms", synset.instance_hyponyms()),
            (HYPERNYM, "Direct hypernyms", synset.hypernyms()),
            (
                INDIRECT_HYPERNYMS,
                "Indirect hypernyms",
                rebuild_tree(synset.tree(lambda x: x.hypernyms()))[1],
            ),
            #  hypernyms', 'Sister terms',
            (INSTANCE_HYPERNYM, "Instance hypernyms", synset.instance_hypernyms()),
            #            (CLASS_REGIONAL, ['domain term region'], ),
            (PART_HOLONYM, "Part holonyms", synset.part_holonyms()),
            (PART_MERONYM, "Part meronyms", synset.part_meronyms()),
            (SUBSTANCE_HOLONYM, "Substance holonyms", synset.substance_holonyms()),
            (SUBSTANCE_MERONYM, "Substance meronyms", synset.substance_meronyms()),
            (MEMBER_HOLONYM, "Member holonyms", synset.member_holonyms()),
            (MEMBER_MERONYM, "Member meronyms", synset.member_meronyms()),
            (ATTRIBUTE, "Attributes", synset.attributes()),
            (ANTONYM, "Antonyms", lemma_property(word, synset, lambda l: l.antonyms())),
            (
                DERIVATIONALLY_RELATED_FORM,
                "Derivationally related form",
                lemma_property(
                    word, synset, lambda l: l.derivationally_related_forms()
                ),
            ),
        )
    elif synset.pos() == wn.VERB:
        return (
            (ANTONYM, "Antonym", lemma_property(word, synset, lambda l: l.antonyms())),
            (HYPONYM, "Hyponym", synset.hyponyms()),
            (HYPERNYM, "Direct hypernyms", synset.hypernyms()),
            (
                INDIRECT_HYPERNYMS,
                "Indirect hypernyms",
                rebuild_tree(synset.tree(lambda x: x.hypernyms()))[1],
            ),
            (ENTAILMENT, "Entailments", synset.entailments()),
            (CAUSE, "Causes", synset.causes()),
            (ALSO_SEE, "Also see", synset.also_sees()),
            (VERB_GROUP, "Verb Groups", synset.verb_groups()),
            (
                DERIVATIONALLY_RELATED_FORM,
                "Derivationally related form",
                lemma_property(
                    word, synset, lambda l: l.derivationally_related_forms()
                ),
            ),
        )
    elif synset.pos() == wn.ADJ or synset.pos == wn.ADJ_SAT:
        return (
            (ANTONYM, "Antonym", lemma_property(word, synset, lambda l: l.antonyms())),
            (SIMILAR, "Similar to", synset.similar_tos()),
            # Participle of verb - not supported by corpus
            (
                PERTAINYM,
                "Pertainyms",
                lemma_property(word, synset, lambda l: l.pertainyms()),
            ),
            (ATTRIBUTE, "Attributes", synset.attributes()),
            (ALSO_SEE, "Also see", synset.also_sees()),
        )
    elif synset.pos() == wn.ADV:
        # This is weird. adverbs such as 'quick' and 'fast' don't seem
        # to have antonyms returned by the corpus.a
        return (
            (ANTONYM, "Antonym", lemma_property(word, synset, lambda l: l.antonyms())),
        )
        # Derived from adjective - not supported by corpus
    else:
        raise TypeError("Unhandles synset POS type: " + str(synset.pos()))


html_header = """
<!DOCTYPE html PUBLIC '-//W3C//DTD HTML 4.01//EN'
'http://www.w3.org/TR/html4/strict.dtd'>
<html>
<head>
<meta name='generator' content=
'HTML Tidy for Windows (vers 14 February 2006), see www.w3.org'>
<meta http-equiv='Content-Type' content=
'text/html; charset=us-ascii'>
<title>NLTK Wordnet Browser display of: %s</title></head>
<body bgcolor='#F5F5F5' text='#000000'>
"""
html_trailer = """
</body>
</html>
"""

explanation = """
<h3>Search Help</h3>
<ul><li>The display below the line is an example of the output the browser
shows you when you enter a search word. The search word was <b>green</b>.</li>
<li>The search result shows for different parts of speech the <b>synsets</b>
i.e. different meanings for the word.</li>
<li>All underlined texts are hypertext links. There are two types of links:
word links and others. Clicking a word link carries out a search for the word
in the Wordnet database.</li>
<li>Clicking a link of the other type opens a display section of data attached
to that link. Clicking that link a second time closes the section again.</li>
<li>Clicking <u>S:</u> opens a section showing the relations for that synset.
</li>
<li>Clicking on a relation name opens a section that displays the associated
synsets.</li>
<li>Type a search word in the <b>Word</b> field and start the search by the
<b>Enter/Return</b> key or click the <b>Search</b> button.</li>
</ul>
<hr width='100%'>
"""

# HTML oriented functions


def _bold(txt):
    return "<b>%s</b>" % txt


def _center(txt):
    return "<center>%s</center>" % txt


def _hlev(n, txt):
    return "<h%d>%s</h%d>" % (n, txt, n)


def _italic(txt):
    return "<i>%s</i>" % txt


def _li(txt):
    return "<li>%s</li>" % txt


def pg(word, body):
    """
    Return a HTML page of NLTK Browser format constructed from the
    word and body

    :param word: The word that the body corresponds to
    :type word: str
    :param body: The HTML body corresponding to the word
    :type body: str
    :return: a HTML page for the word-body combination
    :rtype: str
    """
    return (html_header % word) + body + html_trailer


def _ul(txt):
    return "<ul>" + txt + "</ul>"


def _abbc(txt):
    """
    abbc = asterisks, breaks, bold, center
    """
    return _center(_bold("<br>" * 10 + "*" * 10 + " " + txt + " " + "*" * 10))


full_hyponym_cont_text = _ul(_li(_italic("(has full hyponym continuation)"))) + "\n"


def _get_synset(synset_key):
    """
    The synset key is the unique name of the synset, this can be
    retrieved via synset.name()
    """
    return wn.synset(synset_key)


def _collect_one_synset(word, synset, synset_relations):
    """
    Returns the HTML string for one synset or word

    :param word: the current word
    :type word: str
    :param synset: a synset
    :type synset: synset
    :param synset_relations: information about which synset relations
    to display.
    :type synset_relations: dict(synset_key, set(relation_id))
    :return: The HTML string built for this synset
    :rtype: str
    """
    if isinstance(synset, tuple):  # It's a word
        raise NotImplementedError("word not supported by _collect_one_synset")

    typ = "S"
    pos_tuple = _pos_match((synset.pos(), None, None))
    assert pos_tuple is not None, "pos_tuple is null: synset.pos(): %s" % synset.pos()
    descr = pos_tuple[2]
    ref = copy.deepcopy(Reference(word, synset_relations))
    ref.toggle_synset(synset)
    synset_label = typ + ";"
    if synset.name() in synset_relations:
        synset_label = _bold(synset_label)
    s = f"<li>{make_lookup_link(ref, synset_label)} ({descr}) "

    def format_lemma(w):
        w = w.replace("_", " ")
        if w.lower() == word:
            return _bold(w)
        else:
            ref = Reference(w)
            return make_lookup_link(ref, w)

    s += ", ".join(format_lemma(l.name()) for l in synset.lemmas())

    gl = " ({}) <i>{}</i> ".format(
        synset.definition(),
        "; ".join('"%s"' % e for e in synset.examples()),
    )
    return s + gl + _synset_relations(word, synset, synset_relations) + "</li>\n"


def _collect_all_synsets(word, pos, synset_relations=dict()):
    """
    Return a HTML unordered list of synsets for the given word and
    part of speech.
    """
    return "<ul>%s\n</ul>\n" % "".join(
        _collect_one_synset(word, synset, synset_relations)
        for synset in wn.synsets(word, pos)
    )


def _synset_relations(word, synset, synset_relations):
    """
    Builds the HTML string for the relations of a synset

    :param word: The current word
    :type word: str
    :param synset: The synset for which we're building the relations.
    :type synset: Synset
    :param synset_relations: synset keys and relation types for which to display relations.
    :type synset_relations: dict(synset_key, set(relation_type))
    :return: The HTML for a synset's relations
    :rtype: str
    """

    if not synset.name() in synset_relations:
        return ""
    ref = Reference(word, synset_relations)

    def relation_html(r):
        if isinstance(r, Synset):
            return make_lookup_link(Reference(r.lemma_names()[0]), r.lemma_names()[0])
        elif isinstance(r, Lemma):
            return relation_html(r.synset())
        elif isinstance(r, tuple):
            # It's probably a tuple containing a Synset and a list of
            # similar tuples.  This forms a tree of synsets.
            return "{}\n<ul>{}</ul>\n".format(
                relation_html(r[0]),
                "".join("<li>%s</li>\n" % relation_html(sr) for sr in r[1]),
            )
        else:
            raise TypeError(
                "r must be a synset, lemma or list, it was: type(r) = %s, r = %s"
                % (type(r), r)
            )

    def make_synset_html(db_name, disp_name, rels):
        synset_html = "<i>%s</i>\n" % make_lookup_link(
            copy.deepcopy(ref).toggle_synset_relation(synset, db_name),
            disp_name,
        )

        if db_name in ref.synset_relations[synset.name()]:
            synset_html += "<ul>%s</ul>\n" % "".join(
                "<li>%s</li>\n" % relation_html(r) for r in rels
            )

        return synset_html

    html = (
        "<ul>"
        + "\n".join(
            "<li>%s</li>" % make_synset_html(*rel_data)
            for rel_data in get_relations_data(word, synset)
            if rel_data[2] != []
        )
        + "</ul>"
    )

    return html


class RestrictedUnpickler(pickle.Unpickler):
    """
    Unpickler that prevents any class or function from being used during loading.
    """

    def find_class(self, module, name):
        # Forbid every function
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")


class Reference:
    """
    A reference to a page that may be generated by page_word
    """

    def __init__(self, word, synset_relations=dict()):
        """
        Build a reference to a new page.

        word is the word or words (separated by commas) for which to
        search for synsets of

        synset_relations is a dictionary of synset keys to sets of
        synset relation identifaiers to unfold a list of synset
        relations for.
        """
        self.word = word
        self.synset_relations = synset_relations

    def encode(self):
        """
        Encode this reference into a string to be used in a URL.
        """
        # This uses a tuple rather than an object since the python
        # pickle representation is much smaller and there is no need
        # to represent the complete object.
        string = pickle.dumps((self.word, self.synset_relations), -1)
        return base64.urlsafe_b64encode(string).decode()

    @staticmethod
    def decode(string):
        """
        Decode a reference encoded with Reference.encode
        """
        string = base64.urlsafe_b64decode(string.encode())
        word, synset_relations = RestrictedUnpickler(io.BytesIO(string)).load()
        return Reference(word, synset_relations)

    def toggle_synset_relation(self, synset, relation):
        """
        Toggle the display of the relations for the given synset and
        relation type.

        This function will throw a KeyError if the synset is currently
        not being displayed.
        """
        if relation in self.synset_relations[synset.name()]:
            self.synset_relations[synset.name()].remove(relation)
        else:
            self.synset_relations[synset.name()].add(relation)

        return self

    def toggle_synset(self, synset):
        """
        Toggle displaying of the relation types for the given synset
        """
        if synset.name() in self.synset_relations:
            del self.synset_relations[synset.name()]
        else:
            self.synset_relations[synset.name()] = set()

        return self


def make_lookup_link(ref, label):
    return f'<a href="lookup_{ref.encode()}">{label}</a>'


def page_from_word(word):
    """
    Return a HTML page for the given word.

    :type word: str
    :param word: The currently active word
    :return: A tuple (page,word), where page is the new current HTML page
        to be sent to the browser and
        word is the new current word
    :rtype: A tuple (str,str)
    """
    return page_from_reference(Reference(word))


def page_from_href(href):
    """
    Returns a tuple of the HTML page built and the new current word

    :param href: The hypertext reference to be solved
    :type href: str
    :return: A tuple (page,word), where page is the new current HTML page
             to be sent to the browser and
             word is the new current word
    :rtype: A tuple (str,str)
    """
    return page_from_reference(Reference.decode(href))


def page_from_reference(href):
    """
    Returns a tuple of the HTML page built and the new current word

    :param href: The hypertext reference to be solved
    :type href: str
    :return: A tuple (page,word), where page is the new current HTML page
             to be sent to the browser and
             word is the new current word
    :rtype: A tuple (str,str)
    """
    word = href.word
    pos_forms = defaultdict(list)
    words = word.split(",")
    words = [w for w in [w.strip().lower().replace(" ", "_") for w in words] if w != ""]
    if len(words) == 0:
        # No words were found.
        return "", "Please specify a word to search for."

    # This looks up multiple words at once.  This is probably not
    # necessary and may lead to problems.
    for w in words:
        for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]:
            form = wn.morphy(w, pos)
            if form and form not in pos_forms[pos]:
                pos_forms[pos].append(form)
    body = ""
    for pos, pos_str, name in _pos_tuples():
        if pos in pos_forms:
            body += _hlev(3, name) + "\n"
            for w in pos_forms[pos]:
                # Not all words of exc files are in the database, skip
                # to the next word if a KeyError is raised.
                try:
                    body += _collect_all_synsets(w, pos, href.synset_relations)
                except KeyError:
                    pass
    if not body:
        body = "The word or words '%s' were not found in the dictionary." % word
    return body, word


#####################################################################
# Static pages
#####################################################################


def get_static_page_by_path(path):
    """
    Return a static HTML page from the path given.
    """
    if path == "index_2.html":
        return get_static_index_page(False)
    elif path == "index.html":
        return get_static_index_page(True)
    elif path == "NLTK Wordnet Browser Database Info.html":
        return "Display of Wordnet Database Statistics is not supported"
    elif path == "upper_2.html":
        return get_static_upper_page(False)
    elif path == "upper.html":
        return get_static_upper_page(True)
    elif path == "web_help.html":
        return get_static_web_help_page()
    elif path == "wx_help.html":
        return get_static_wx_help_page()
    raise FileNotFoundError()


def get_static_web_help_page():
    """
    Return the static web help page.
    """
    return """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
     <!-- Natural Language Toolkit: Wordnet Interface: Graphical Wordnet Browser
            Copyright (C) 2001-2023 NLTK Project
            Author: Jussi Salmela <jtsalmela@users.sourceforge.net>
            URL: <https://www.nltk.org/>
            For license information, see LICENSE.TXT -->
     <head>
          <meta http-equiv='Content-Type' content='text/html; charset=us-ascii'>
          <title>NLTK Wordnet Browser display of: * Help *</title>
     </head>
<body bgcolor='#F5F5F5' text='#000000'>
<h2>NLTK Wordnet Browser Help</h2>
<p>The NLTK Wordnet Browser is a tool to use in browsing the Wordnet database. It tries to behave like the Wordnet project's web browser but the difference is that the NLTK Wordnet Browser uses a local Wordnet database.
<p><b>You are using the Javascript client part of the NLTK Wordnet BrowseServer.</b> We assume your browser is in tab sheets enabled mode.</p>
<p>For background information on Wordnet, see the Wordnet project home page: <a href="https://wordnet.princeton.edu/"><b> https://wordnet.princeton.edu/</b></a>. For more information on the NLTK project, see the project home:
<a href="https://www.nltk.org/"><b>https://www.nltk.org/</b></a>. To get an idea of what the Wordnet version used by this browser includes choose <b>Show Database Info</b> from the <b>View</b> submenu.</p>
<h3>Word search</h3>
<p>The word to be searched is typed into the <b>New Word</b> field and the search started with Enter or by clicking the <b>Search</b> button. There is no uppercase/lowercase distinction: the search word is transformed to lowercase before the search.</p>
<p>In addition, the word does not have to be in base form. The browser tries to find the possible base form(s) by making certain morphological substitutions. Typing <b>fLIeS</b> as an obscure example gives one <a href="MfLIeS">this</a>. Click the previous link to see what this kind of search looks like and then come back to this page by using the <b>Alt+LeftArrow</b> key combination.</p>
<p>The result of a search is a display of one or more
<b>synsets</b> for every part of speech in which a form of the
search word was found to occur. A synset is a set of words
having the same sense or meaning. Each word in a synset that is
underlined is a hyperlink which can be clicked to trigger an
automatic search for that word.</p>
<p>Every synset has a hyperlink <b>S:</b> at the start of its
display line. Clicking that symbol shows you the name of every
<b>relation</b> that this synset is part of. Every relation name is a hyperlink that opens up a display for that relation. Clicking it another time closes the display again. Clicking another relation name on a line that has an opened relation closes the open relation and opens the clicked relation.</p>
<p>It is also possible to give two or more words or collocations to be searched at the same time separating them with a comma like this <a href="Mcheer up,clear up">cheer up,clear up</a>, for example. Click the previous link to see what this kind of search looks like and then come back to this page by using the <b>Alt+LeftArrow</b> key combination. As you could see the search result includes the synsets found in the same order than the forms were given in the search field.</p>
<p>
There are also word level (lexical) relations recorded in the Wordnet database. Opening this kind of relation displays lines with a hyperlink <b>W:</b> at their beginning. Clicking this link shows more info on the word in question.</p>
<h3>The Buttons</h3>
<p>The <b>Search</b> and <b>Help</b> buttons need no more explanation. </p>
<p>The <b>Show Database Info</b> button shows a collection of Wordnet database statistics.</p>
<p>The <b>Shutdown the Server</b> button is shown for the first client of the BrowServer program i.e. for the client that is automatically launched when the BrowServer is started but not for the succeeding clients in order to protect the server from accidental shutdowns.
</p></body>
</html>
"""


def get_static_welcome_message():
    """
    Get the static welcome page.
    """
    return """
<h3>Search Help</h3>
<ul><li>The display below the line is an example of the output the browser
shows you when you enter a search word. The search word was <b>green</b>.</li>
<li>The search result shows for different parts of speech the <b>synsets</b>
i.e. different meanings for the word.</li>
<li>All underlined texts are hypertext links. There are two types of links:
word links and others. Clicking a word link carries out a search for the word
in the Wordnet database.</li>
<li>Clicking a link of the other type opens a display section of data attached
to that link. Clicking that link a second time closes the section again.</li>
<li>Clicking <u>S:</u> opens a section showing the relations for that synset.</li>
<li>Clicking on a relation name opens a section that displays the associated
synsets.</li>
<li>Type a search word in the <b>Next Word</b> field and start the search by the
<b>Enter/Return</b> key or click the <b>Search</b> button.</li>
</ul>
"""


def get_static_index_page(with_shutdown):
    """
    Get the static index page.
    """
    template = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Frameset//EN"  "http://www.w3.org/TR/html4/frameset.dtd">
<HTML>
     <!-- Natural Language Toolkit: Wordnet Interface: Graphical Wordnet Browser
            Copyright (C) 2001-2023 NLTK Project
            Author: Jussi Salmela <jtsalmela@users.sourceforge.net>
            URL: <https://www.nltk.org/>
            For license information, see LICENSE.TXT -->
     <HEAD>
         <TITLE>NLTK Wordnet Browser</TITLE>
     </HEAD>

<frameset rows="7%%,93%%">
    <frame src="%s" name="header">
    <frame src="start_page" name="body">
</frameset>
</HTML>
"""
    if with_shutdown:
        upper_link = "upper.html"
    else:
        upper_link = "upper_2.html"

    return template % upper_link


def get_static_upper_page(with_shutdown):
    """
    Return the upper frame page,

    If with_shutdown is True then a 'shutdown' button is also provided
    to shutdown the server.
    """
    template = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
    <!-- Natural Language Toolkit: Wordnet Interface: Graphical Wordnet Browser
        Copyright (C) 2001-2023 NLTK Project
        Author: Jussi Salmela <jtsalmela@users.sourceforge.net>
        URL: <https://www.nltk.org/>
        For license information, see LICENSE.TXT -->
    <head>
                <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
        <title>Untitled Document</title>
    </head>
    <body>
    <form method="GET" action="search" target="body">
            Current Word:&nbsp;<input type="text" id="currentWord" size="10" disabled>
            Next Word:&nbsp;<input type="text" id="nextWord" name="nextWord" size="10">
            <input name="searchButton" type="submit" value="Search">
    </form>
        <a target="body" href="web_help.html">Help</a>
        %s

</body>
</html>
"""
    if with_shutdown:
        shutdown_link = '<a href="SHUTDOWN THE SERVER">Shutdown</a>'
    else:
        shutdown_link = ""

    return template % shutdown_link


def usage():
    """
    Display the command line help message.
    """
    print(__doc__)


def app():
    # Parse and interpret options.
    (opts, _) = getopt.getopt(
        argv[1:], "l:p:sh", ["logfile=", "port=", "server-mode", "help"]
    )
    port = 8000
    server_mode = False
    help_mode = False
    logfilename = None
    for (opt, value) in opts:
        if (opt == "-l") or (opt == "--logfile"):
            logfilename = str(value)
        elif (opt == "-p") or (opt == "--port"):
            port = int(value)
        elif (opt == "-s") or (opt == "--server-mode"):
            server_mode = True
        elif (opt == "-h") or (opt == "--help"):
            help_mode = True

    if help_mode:
        usage()
    else:
        wnb(port, not server_mode, logfilename)


if __name__ == "__main__":
    app()

__all__ = ["app"]
