import unittest

from nltk.corpus import brown
from nltk.jsontags import JSONTaggedDecoder, JSONTaggedEncoder
from nltk.tag import (
    AffixTagger,
    BigramTagger,
    BrillTagger,
    BrillTaggerTrainer,
    DefaultTagger,
    NgramTagger,
    PerceptronTagger,
    RegexpTagger,
    TrigramTagger,
    UnigramTagger,
)
from nltk.tag.brill import nltkdemo18


class TestJSONSerialization(unittest.TestCase):
    def setUp(self):
        self.corpus = brown.tagged_sents()[:35]
        self.decoder = JSONTaggedDecoder()
        self.encoder = JSONTaggedEncoder()
        self.default_tagger = DefaultTagger("NN")

    def test_default_tagger(self):
        encoded = self.encoder.encode(self.default_tagger)
        decoded = self.decoder.decode(encoded)

        self.assertEqual(repr(self.default_tagger), repr(decoded))
        self.assertEqual(self.default_tagger._tag, decoded._tag)

    def test_regexp_tagger(self):
        tagger = RegexpTagger([(r".*", "NN")], backoff=self.default_tagger)

        encoded = self.encoder.encode(tagger)
        decoded = self.decoder.decode(encoded)

        self.assertEqual(repr(tagger), repr(decoded))
        self.assertEqual(repr(tagger.backoff), repr(decoded.backoff))
        self.assertEqual(tagger._regexps, decoded._regexps)

    def test_affix_tagger(self):
        tagger = AffixTagger(self.corpus, backoff=self.default_tagger)

        encoded = self.encoder.encode(tagger)
        decoded = self.decoder.decode(encoded)

        self.assertEqual(repr(tagger), repr(decoded))
        self.assertEqual(repr(tagger.backoff), repr(decoded.backoff))
        self.assertEqual(tagger._affix_length, decoded._affix_length)
        self.assertEqual(tagger._min_word_length, decoded._min_word_length)
        self.assertEqual(tagger._context_to_tag, decoded._context_to_tag)

    def test_ngram_taggers(self):
        unitagger = UnigramTagger(self.corpus, backoff=self.default_tagger)
        bitagger = BigramTagger(self.corpus, backoff=unitagger)
        tritagger = TrigramTagger(self.corpus, backoff=bitagger)
        ntagger = NgramTagger(4, self.corpus, backoff=tritagger)

        encoded = self.encoder.encode(ntagger)
        decoded = self.decoder.decode(encoded)

        self.assertEqual(repr(ntagger), repr(decoded))
        self.assertEqual(repr(tritagger), repr(decoded.backoff))
        self.assertEqual(repr(bitagger), repr(decoded.backoff.backoff))
        self.assertEqual(repr(unitagger), repr(decoded.backoff.backoff.backoff))
        self.assertEqual(
            repr(self.default_tagger), repr(decoded.backoff.backoff.backoff.backoff)
        )

    def test_perceptron_tagger(self):
        tagger = PerceptronTagger(load=False)
        tagger.train(self.corpus)

        encoded = self.encoder.encode(tagger)
        decoded = self.decoder.decode(encoded)

        self.assertEqual(tagger.model.weights, decoded.model.weights)
        self.assertEqual(tagger.tagdict, decoded.tagdict)
        self.assertEqual(tagger.classes, decoded.classes)

    def test_brill_tagger(self):
        trainer = BrillTaggerTrainer(
            self.default_tagger, nltkdemo18(), deterministic=True
        )
        tagger = trainer.train(self.corpus, max_rules=30)

        encoded = self.encoder.encode(tagger)
        decoded = self.decoder.decode(encoded)

        self.assertEqual(repr(tagger._initial_tagger), repr(decoded._initial_tagger))
        self.assertEqual(tagger._rules, decoded._rules)
        self.assertEqual(tagger._training_stats, decoded._training_stats)
