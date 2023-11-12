# -*- coding: utf-8 -*-
'''Various noun phrase extractors.'''
from __future__ import unicode_literals, absolute_import

import nltk

from textblob.taggers import PatternTagger
from textblob.decorators import requires_nltk_corpus
from textblob.utils import tree2str, filter_insignificant
from textblob.base import BaseNPExtractor


class ChunkParser(nltk.ChunkParserI):

    def __init__(self):
        self._trained = False

    @requires_nltk_corpus
    def train(self):
        '''Train the Chunker on the ConLL-2000 corpus.'''
        train_data = [[(t, c) for _, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in
                      nltk.corpus.conll2000.chunked_sents('train.txt',
                                                    chunk_types=['NP'])]
        unigram_tagger = nltk.UnigramTagger(train_data)
        self.tagger = nltk.BigramTagger(train_data, backoff=unigram_tagger)
        self._trained = True

    def parse(self, sentence):
        '''Return the parse tree for the sentence.'''
        if not self._trained:
            self.train()
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in
                     zip(sentence, chunktags)]
        return nltk.chunk.util.conlltags2tree(conlltags)


class ConllExtractor(BaseNPExtractor):

    '''A noun phrase extractor that uses chunk parsing trained with the
    ConLL-2000 training corpus.
    '''

    POS_TAGGER = PatternTagger()

    # The context-free grammar with which to filter the noun phrases
    CFG = {
        ('NNP', 'NNP'): 'NNP',
        ('NN', 'NN'): 'NNI',
        ('NNI', 'NN'): 'NNI',
        ('JJ', 'JJ'): 'JJ',
        ('JJ', 'NN'): 'NNI',
        }

    # POS suffixes that will be ignored
    INSIGNIFICANT_SUFFIXES = ['DT', 'CC', 'PRP$', 'PRP']

    def __init__(self, parser=None):
        self.parser = ChunkParser() if not parser else parser

    def extract(self, text):
        '''Return a list of noun phrases (strings) for body of text.'''
        sentences = nltk.tokenize.sent_tokenize(text)
        noun_phrases = []
        for sentence in sentences:
            parsed = self._parse_sentence(sentence)
            # Get the string representation of each subtree that is a
            # noun phrase tree
            phrases = [_normalize_tags(filter_insignificant(each,
                       self.INSIGNIFICANT_SUFFIXES)) for each in parsed
                       if isinstance(each, nltk.tree.Tree) and each.label()
                       == 'NP' and len(filter_insignificant(each)) >= 1
                       and _is_match(each, cfg=self.CFG)]
            nps = [tree2str(phrase) for phrase in phrases]
            noun_phrases.extend(nps)
        return noun_phrases

    def _parse_sentence(self, sentence):
        '''Tag and parse a sentence (a plain, untagged string).'''
        tagged = self.POS_TAGGER.tag(sentence)
        return self.parser.parse(tagged)


class FastNPExtractor(BaseNPExtractor):

    '''A fast and simple noun phrase extractor.

    Credit to Shlomi Babluk. Link to original blog post:

        http://thetokenizer.com/2013/05/09/efficient-way-to-extract-the-main-topics-of-a-sentence/
    '''

    CFG = {
        ('NNP', 'NNP'): 'NNP',
        ('NN', 'NN'): 'NNI',
        ('NNI', 'NN'): 'NNI',
        ('JJ', 'JJ'): 'JJ',
        ('JJ', 'NN'): 'NNI',
        }

    def __init__(self):
        self._trained = False

    @requires_nltk_corpus
    def train(self):
        train_data = nltk.corpus.brown.tagged_sents(categories='news')
        regexp_tagger = nltk.RegexpTagger([
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
            (r'(-|:|;)$', ':'),
            (r'\'*$', 'MD'),
            (r'(The|the|A|a|An|an)$', 'AT'),
            (r'.*able$', 'JJ'),
            (r'^[A-Z].*$', 'NNP'),
            (r'.*ness$', 'NN'),
            (r'.*ly$', 'RB'),
            (r'.*s$', 'NNS'),
            (r'.*ing$', 'VBG'),
            (r'.*ed$', 'VBD'),
            (r'.*', 'NN'),
            ])
        unigram_tagger = nltk.UnigramTagger(train_data, backoff=regexp_tagger)
        self.tagger = nltk.BigramTagger(train_data, backoff=unigram_tagger)
        self._trained = True
        return None


    def _tokenize_sentence(self, sentence):
        '''Split the sentence into single words/tokens'''
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def extract(self, sentence):
        '''Return a list of noun phrases (strings) for body of text.'''
        if not self._trained:
            self.train()
        tokens = self._tokenize_sentence(sentence)
        tagged = self.tagger.tag(tokens)
        tags = _normalize_tags(tagged)
        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = t1[1], t2[1]
                value = self.CFG.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = '%s %s' % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break

        matches = [t[0] for t in tags if t[1] in ['NNP', 'NNI']]
        return matches


### Utility methods ###

def _normalize_tags(chunk):
    '''Normalize the corpus tags.
    ("NN", "NN-PL", "NNS") -> "NN"
    '''
    ret = []
    for word, tag in chunk:
        if tag == 'NP-TL' or tag == 'NP':
            ret.append((word, 'NNP'))
            continue
        if tag.endswith('-TL'):
            ret.append((word, tag[:-3]))
            continue
        if tag.endswith('S'):
            ret.append((word, tag[:-1]))
            continue
        ret.append((word, tag))
    return ret


def _is_match(tagged_phrase, cfg):
    '''Return whether or not a tagged phrases matches a context-free grammar.
    '''
    copy = list(tagged_phrase)  # A copy of the list
    merge = True
    while merge:
        merge = False
        for i in range(len(copy) - 1):
            first, second = copy[i], copy[i + 1]
            key = first[1], second[1]  # Tuple of tags e.g. ('NN', 'JJ')
            value = cfg.get(key, None)
            if value:
                merge = True
                copy.pop(i)
                copy.pop(i)
                match = '{0} {1}'.format(first[0], second[0])
                pos = value
                copy.insert(i, (match, pos))
                break
    match = any([t[1] in ('NNP', 'NNI') for t in copy])
    return match
