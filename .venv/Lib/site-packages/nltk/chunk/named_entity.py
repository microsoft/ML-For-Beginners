# Natural Language Toolkit: Chunk parsing API
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Named entity chunker
"""

import os
import pickle
import re
from xml.etree import ElementTree as ET

from nltk.tag import ClassifierBasedTagger, pos_tag

try:
    from nltk.classify import MaxentClassifier
except ImportError:
    pass

from nltk.chunk.api import ChunkParserI
from nltk.chunk.util import ChunkScore
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.tree import Tree


class NEChunkParserTagger(ClassifierBasedTagger):
    """
    The IOB tagger used by the chunk parser.
    """

    def __init__(self, train):
        ClassifierBasedTagger.__init__(
            self, train=train, classifier_builder=self._classifier_builder
        )

    def _classifier_builder(self, train):
        return MaxentClassifier.train(
            train, algorithm="megam", gaussian_prior_sigma=1, trace=2
        )

    def _english_wordlist(self):
        try:
            wl = self._en_wordlist
        except AttributeError:
            from nltk.corpus import words

            self._en_wordlist = set(words.words("en-basic"))
            wl = self._en_wordlist
        return wl

    def _feature_detector(self, tokens, index, history):
        word = tokens[index][0]
        pos = simplify_pos(tokens[index][1])
        if index == 0:
            prevword = prevprevword = None
            prevpos = prevprevpos = None
            prevshape = prevtag = prevprevtag = None
        elif index == 1:
            prevword = tokens[index - 1][0].lower()
            prevprevword = None
            prevpos = simplify_pos(tokens[index - 1][1])
            prevprevpos = None
            prevtag = history[index - 1][0]
            prevshape = prevprevtag = None
        else:
            prevword = tokens[index - 1][0].lower()
            prevprevword = tokens[index - 2][0].lower()
            prevpos = simplify_pos(tokens[index - 1][1])
            prevprevpos = simplify_pos(tokens[index - 2][1])
            prevtag = history[index - 1]
            prevprevtag = history[index - 2]
            prevshape = shape(prevword)
        if index == len(tokens) - 1:
            nextword = nextnextword = None
            nextpos = nextnextpos = None
        elif index == len(tokens) - 2:
            nextword = tokens[index + 1][0].lower()
            nextpos = tokens[index + 1][1].lower()
            nextnextword = None
            nextnextpos = None
        else:
            nextword = tokens[index + 1][0].lower()
            nextpos = tokens[index + 1][1].lower()
            nextnextword = tokens[index + 2][0].lower()
            nextnextpos = tokens[index + 2][1].lower()

        # 89.6
        features = {
            "bias": True,
            "shape": shape(word),
            "wordlen": len(word),
            "prefix3": word[:3].lower(),
            "suffix3": word[-3:].lower(),
            "pos": pos,
            "word": word,
            "en-wordlist": (word in self._english_wordlist()),
            "prevtag": prevtag,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevword": prevword,
            "nextword": nextword,
            "word+nextpos": f"{word.lower()}+{nextpos}",
            "pos+prevtag": f"{pos}+{prevtag}",
            "shape+prevtag": f"{prevshape}+{prevtag}",
        }

        return features


class NEChunkParser(ChunkParserI):
    """
    Expected input: list of pos-tagged words
    """

    def __init__(self, train):
        self._train(train)

    def parse(self, tokens):
        """
        Each token should be a pos-tagged word
        """
        tagged = self._tagger.tag(tokens)
        tree = self._tagged_to_parse(tagged)
        return tree

    def _train(self, corpus):
        # Convert to tagged sequence
        corpus = [self._parse_to_tagged(s) for s in corpus]

        self._tagger = NEChunkParserTagger(train=corpus)

    def _tagged_to_parse(self, tagged_tokens):
        """
        Convert a list of tagged tokens to a chunk-parse tree.
        """
        sent = Tree("S", [])

        for (tok, tag) in tagged_tokens:
            if tag == "O":
                sent.append(tok)
            elif tag.startswith("B-"):
                sent.append(Tree(tag[2:], [tok]))
            elif tag.startswith("I-"):
                if sent and isinstance(sent[-1], Tree) and sent[-1].label() == tag[2:]:
                    sent[-1].append(tok)
                else:
                    sent.append(Tree(tag[2:], [tok]))
        return sent

    @staticmethod
    def _parse_to_tagged(sent):
        """
        Convert a chunk-parse tree to a list of tagged tokens.
        """
        toks = []
        for child in sent:
            if isinstance(child, Tree):
                if len(child) == 0:
                    print("Warning -- empty chunk in sentence")
                    continue
                toks.append((child[0], f"B-{child.label()}"))
                for tok in child[1:]:
                    toks.append((tok, f"I-{child.label()}"))
            else:
                toks.append((child, "O"))
        return toks


def shape(word):
    if re.match(r"[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$", word, re.UNICODE):
        return "number"
    elif re.match(r"\W+$", word, re.UNICODE):
        return "punct"
    elif re.match(r"\w+$", word, re.UNICODE):
        if word.istitle():
            return "upcase"
        elif word.islower():
            return "downcase"
        else:
            return "mixedcase"
    else:
        return "other"


def simplify_pos(s):
    if s.startswith("V"):
        return "V"
    else:
        return s.split("-")[0]


def postag_tree(tree):
    # Part-of-speech tagging.
    words = tree.leaves()
    tag_iter = (pos for (word, pos) in pos_tag(words))
    newtree = Tree("S", [])
    for child in tree:
        if isinstance(child, Tree):
            newtree.append(Tree(child.label(), []))
            for subchild in child:
                newtree[-1].append((subchild, next(tag_iter)))
        else:
            newtree.append((child, next(tag_iter)))
    return newtree


def load_ace_data(roots, fmt="binary", skip_bnews=True):
    for root in roots:
        for root, dirs, files in os.walk(root):
            if root.endswith("bnews") and skip_bnews:
                continue
            for f in files:
                if f.endswith(".sgm"):
                    yield from load_ace_file(os.path.join(root, f), fmt)


def load_ace_file(textfile, fmt):
    print(f"  - {os.path.split(textfile)[1]}")
    annfile = textfile + ".tmx.rdc.xml"

    # Read the xml file, and get a list of entities
    entities = []
    with open(annfile) as infile:
        xml = ET.parse(infile).getroot()
    for entity in xml.findall("document/entity"):
        typ = entity.find("entity_type").text
        for mention in entity.findall("entity_mention"):
            if mention.get("TYPE") != "NAME":
                continue  # only NEs
            s = int(mention.find("head/charseq/start").text)
            e = int(mention.find("head/charseq/end").text) + 1
            entities.append((s, e, typ))

    # Read the text file, and mark the entities.
    with open(textfile) as infile:
        text = infile.read()

    # Strip XML tags, since they don't count towards the indices
    text = re.sub("<(?!/?TEXT)[^>]+>", "", text)

    # Blank out anything before/after <TEXT>
    def subfunc(m):
        return " " * (m.end() - m.start() - 6)

    text = re.sub(r"[\s\S]*<TEXT>", subfunc, text)
    text = re.sub(r"</TEXT>[\s\S]*", "", text)

    # Simplify quotes
    text = re.sub("``", ' "', text)
    text = re.sub("''", '" ', text)

    entity_types = {typ for (s, e, typ) in entities}

    # Binary distinction (NE or not NE)
    if fmt == "binary":
        i = 0
        toks = Tree("S", [])
        for (s, e, typ) in sorted(entities):
            if s < i:
                s = i  # Overlapping!  Deal with this better?
            if e <= s:
                continue
            toks.extend(word_tokenize(text[i:s]))
            toks.append(Tree("NE", text[s:e].split()))
            i = e
        toks.extend(word_tokenize(text[i:]))
        yield toks

    # Multiclass distinction (NE type)
    elif fmt == "multiclass":
        i = 0
        toks = Tree("S", [])
        for (s, e, typ) in sorted(entities):
            if s < i:
                s = i  # Overlapping!  Deal with this better?
            if e <= s:
                continue
            toks.extend(word_tokenize(text[i:s]))
            toks.append(Tree(typ, text[s:e].split()))
            i = e
        toks.extend(word_tokenize(text[i:]))
        yield toks

    else:
        raise ValueError("bad fmt value")


# This probably belongs in a more general-purpose location (as does
# the parse_to_tagged function).
def cmp_chunks(correct, guessed):
    correct = NEChunkParser._parse_to_tagged(correct)
    guessed = NEChunkParser._parse_to_tagged(guessed)
    ellipsis = False
    for (w, ct), (w, gt) in zip(correct, guessed):
        if ct == gt == "O":
            if not ellipsis:
                print(f"  {ct:15} {gt:15} {w}")
                print("  {:15} {:15} {2}".format("...", "...", "..."))
                ellipsis = True
        else:
            ellipsis = False
            print(f"  {ct:15} {gt:15} {w}")


def build_model(fmt="binary"):
    print("Loading training data...")
    train_paths = [
        find("corpora/ace_data/ace.dev"),
        find("corpora/ace_data/ace.heldout"),
        find("corpora/ace_data/bbn.dev"),
        find("corpora/ace_data/muc.dev"),
    ]
    train_trees = load_ace_data(train_paths, fmt)
    train_data = [postag_tree(t) for t in train_trees]
    print("Training...")
    cp = NEChunkParser(train_data)
    del train_data

    print("Loading eval data...")
    eval_paths = [find("corpora/ace_data/ace.eval")]
    eval_trees = load_ace_data(eval_paths, fmt)
    eval_data = [postag_tree(t) for t in eval_trees]

    print("Evaluating...")
    chunkscore = ChunkScore()
    for i, correct in enumerate(eval_data):
        guess = cp.parse(correct.leaves())
        chunkscore.score(correct, guess)
        if i < 3:
            cmp_chunks(correct, guess)
    print(chunkscore)

    outfilename = f"/tmp/ne_chunker_{fmt}.pickle"
    print(f"Saving chunker to {outfilename}...")

    with open(outfilename, "wb") as outfile:
        pickle.dump(cp, outfile, -1)

    return cp


if __name__ == "__main__":
    # Make sure that the pickled object has the right class name:
    from nltk.chunk.named_entity import build_model

    build_model("binary")
    build_model("multiclass")
