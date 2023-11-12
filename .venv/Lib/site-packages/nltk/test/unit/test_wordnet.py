"""
Unit tests for nltk.corpus.wordnet
See also nltk/test/wordnet.doctest
"""
import unittest

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic

wn.ensure_loaded()
S = wn.synset
L = wn.lemma


class WordnNetDemo(unittest.TestCase):
    def test_retrieve_synset(self):
        move_synset = S("go.v.21")
        self.assertEqual(move_synset.name(), "move.v.15")
        self.assertEqual(move_synset.lemma_names(), ["move", "go"])
        self.assertEqual(
            move_synset.definition(), "have a turn; make one's move in a game"
        )
        self.assertEqual(move_synset.examples(), ["Can I go now?"])

    def test_retrieve_synsets(self):
        self.assertEqual(sorted(wn.synsets("zap", pos="n")), [S("zap.n.01")])
        self.assertEqual(
            sorted(wn.synsets("zap", pos="v")),
            [S("microwave.v.01"), S("nuke.v.01"), S("zap.v.01"), S("zap.v.02")],
        )

    def test_hyperhyponyms(self):
        # Not every synset as hypernyms()
        self.assertEqual(S("travel.v.01").hypernyms(), [])
        self.assertEqual(S("travel.v.02").hypernyms(), [S("travel.v.03")])
        self.assertEqual(S("travel.v.03").hypernyms(), [])

        # Test hyper-/hyponyms.
        self.assertEqual(S("breakfast.n.1").hypernyms(), [S("meal.n.01")])
        first_five_meal_hypo = [
            S("banquet.n.02"),
            S("bite.n.04"),
            S("breakfast.n.01"),
            S("brunch.n.01"),
            S("buffet.n.02"),
        ]
        self.assertEqual(sorted(S("meal.n.1").hyponyms()[:5]), first_five_meal_hypo)
        self.assertEqual(S("Austen.n.1").instance_hypernyms(), [S("writer.n.01")])
        first_five_composer_hypo = [
            S("ambrose.n.01"),
            S("bach.n.01"),
            S("barber.n.01"),
            S("bartok.n.01"),
            S("beethoven.n.01"),
        ]
        self.assertEqual(
            S("composer.n.1").instance_hyponyms()[:5], first_five_composer_hypo
        )

        # Test root hyper-/hyponyms
        self.assertEqual(S("person.n.01").root_hypernyms(), [S("entity.n.01")])
        self.assertEqual(S("sail.v.01").root_hypernyms(), [S("travel.v.01")])
        self.assertEqual(
            S("fall.v.12").root_hypernyms(), [S("act.v.01"), S("fall.v.17")]
        )

    def test_derivationally_related_forms(self):
        # Test `derivationally_related_forms()`
        self.assertEqual(
            L("zap.v.03.nuke").derivationally_related_forms(),
            [L("atomic_warhead.n.01.nuke")],
        )
        self.assertEqual(
            L("zap.v.03.atomize").derivationally_related_forms(),
            [L("atomization.n.02.atomization")],
        )
        self.assertEqual(
            L("zap.v.03.atomise").derivationally_related_forms(),
            [L("atomization.n.02.atomisation")],
        )
        self.assertEqual(L("zap.v.03.zap").derivationally_related_forms(), [])

    def test_meronyms_holonyms(self):
        # Test meronyms, holonyms.
        self.assertEqual(
            S("dog.n.01").member_holonyms(), [S("canis.n.01"), S("pack.n.06")]
        )
        self.assertEqual(S("dog.n.01").part_meronyms(), [S("flag.n.07")])

        self.assertEqual(S("faculty.n.2").member_meronyms(), [S("professor.n.01")])
        self.assertEqual(S("copilot.n.1").member_holonyms(), [S("crew.n.01")])

        self.assertEqual(
            S("table.n.2").part_meronyms(),
            [S("leg.n.03"), S("tabletop.n.01"), S("tableware.n.01")],
        )
        self.assertEqual(S("course.n.7").part_holonyms(), [S("meal.n.01")])

        self.assertEqual(
            S("water.n.1").substance_meronyms(), [S("hydrogen.n.01"), S("oxygen.n.01")]
        )
        self.assertEqual(
            S("gin.n.1").substance_holonyms(),
            [
                S("gin_and_it.n.01"),
                S("gin_and_tonic.n.01"),
                S("martini.n.01"),
                S("pink_lady.n.01"),
            ],
        )

    def test_antonyms(self):
        # Test antonyms.
        self.assertEqual(
            L("leader.n.1.leader").antonyms(), [L("follower.n.01.follower")]
        )
        self.assertEqual(
            L("increase.v.1.increase").antonyms(), [L("decrease.v.01.decrease")]
        )

    def test_misc_relations(self):
        # Test misc relations.
        self.assertEqual(S("snore.v.1").entailments(), [S("sleep.v.01")])
        self.assertEqual(
            S("heavy.a.1").similar_tos(),
            [
                S("dense.s.03"),
                S("doughy.s.01"),
                S("heavier-than-air.s.01"),
                S("hefty.s.02"),
                S("massive.s.04"),
                S("non-buoyant.s.01"),
                S("ponderous.s.02"),
            ],
        )
        self.assertEqual(S("light.a.1").attributes(), [S("weight.n.01")])
        self.assertEqual(S("heavy.a.1").attributes(), [S("weight.n.01")])

        # Test pertainyms.
        self.assertEqual(
            L("English.a.1.English").pertainyms(), [L("england.n.01.England")]
        )

    def test_lch(self):
        # Test LCH.
        self.assertEqual(
            S("person.n.01").lowest_common_hypernyms(S("dog.n.01")),
            [S("organism.n.01")],
        )
        self.assertEqual(
            S("woman.n.01").lowest_common_hypernyms(S("girlfriend.n.02")),
            [S("woman.n.01")],
        )

    def test_domains(self):
        # Test domains.
        self.assertEqual(S("code.n.03").topic_domains(), [S("computer_science.n.01")])
        self.assertEqual(S("pukka.a.01").region_domains(), [S("india.n.01")])
        self.assertEqual(S("freaky.a.01").usage_domains(), [S("slang.n.02")])

    def test_in_topic_domains(self):
        # Test in domains.
        self.assertEqual(
            S("computer_science.n.01").in_topic_domains()[0], S("access.n.05")
        )
        self.assertEqual(S("germany.n.01").in_region_domains()[23], S("trillion.n.02"))
        self.assertEqual(S("slang.n.02").in_usage_domains()[1], S("airhead.n.01"))

    def test_wordnet_similarities(self):
        # Path based similarities.
        self.assertAlmostEqual(S("cat.n.01").path_similarity(S("cat.n.01")), 1.0)
        self.assertAlmostEqual(S("dog.n.01").path_similarity(S("cat.n.01")), 0.2)
        self.assertAlmostEqual(
            S("car.n.01").path_similarity(S("automobile.v.01")),
            S("automobile.v.01").path_similarity(S("car.n.01")),
        )
        self.assertAlmostEqual(
            S("big.a.01").path_similarity(S("dog.n.01")),
            S("dog.n.01").path_similarity(S("big.a.01")),
        )
        self.assertAlmostEqual(
            S("big.a.01").path_similarity(S("long.a.01")),
            S("long.a.01").path_similarity(S("big.a.01")),
        )
        self.assertAlmostEqual(
            S("dog.n.01").lch_similarity(S("cat.n.01")), 2.028, places=3
        )
        self.assertAlmostEqual(
            S("dog.n.01").wup_similarity(S("cat.n.01")), 0.8571, places=3
        )
        self.assertAlmostEqual(
            S("car.n.01").wup_similarity(S("automobile.v.01")),
            S("automobile.v.01").wup_similarity(S("car.n.01")),
        )
        self.assertAlmostEqual(
            S("big.a.01").wup_similarity(S("dog.n.01")),
            S("dog.n.01").wup_similarity(S("big.a.01")),
        )
        self.assertAlmostEqual(
            S("big.a.01").wup_similarity(S("long.a.01")),
            S("long.a.01").wup_similarity(S("big.a.01")),
        )
        self.assertAlmostEqual(
            S("big.a.01").lch_similarity(S("long.a.01")),
            S("long.a.01").lch_similarity(S("big.a.01")),
        )
        # Information Content similarities.
        brown_ic = wnic.ic("ic-brown.dat")
        self.assertAlmostEqual(
            S("dog.n.01").jcn_similarity(S("cat.n.01"), brown_ic), 0.4497, places=3
        )
        semcor_ic = wnic.ic("ic-semcor.dat")
        self.assertAlmostEqual(
            S("dog.n.01").lin_similarity(S("cat.n.01"), semcor_ic), 0.8863, places=3
        )

    def test_omw_lemma_no_trailing_underscore(self):
        expected = sorted(
            [
                "popolna_sprememba_v_mišljenju",
                "popoln_obrat",
                "preobrat",
                "preobrat_v_mišljenju",
            ]
        )
        self.assertEqual(sorted(S("about-face.n.02").lemma_names(lang="slv")), expected)

    def test_iterable_type_for_all_lemma_names(self):
        # Duck-test for iterables.
        # See https://stackoverflow.com/a/36230057/610569
        cat_lemmas = wn.all_lemma_names(lang="cat")
        eng_lemmas = wn.all_lemma_names(lang="eng")

        self.assertTrue(hasattr(eng_lemmas, "__iter__"))
        self.assertTrue(hasattr(eng_lemmas, "__next__") or hasattr(eng_lemmas, "next"))
        self.assertTrue(eng_lemmas.__iter__() is eng_lemmas)

        self.assertTrue(hasattr(cat_lemmas, "__iter__"))
        self.assertTrue(hasattr(cat_lemmas, "__next__") or hasattr(eng_lemmas, "next"))
        self.assertTrue(cat_lemmas.__iter__() is cat_lemmas)
