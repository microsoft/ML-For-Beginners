import unittest

import pytest

from nltk.corpus import (  # mwa_ppdb
    cess_cat,
    cess_esp,
    conll2007,
    floresta,
    indian,
    ptb,
    sinica_treebank,
    udhr,
)
from nltk.tree import Tree


class TestUdhr(unittest.TestCase):
    def test_words(self):
        for name in udhr.fileids():
            words = list(udhr.words(name))
            self.assertTrue(words)

    def test_raw_unicode(self):
        for name in udhr.fileids():
            txt = udhr.raw(name)
            assert not isinstance(txt, bytes), name

    def test_polish_encoding(self):
        text_pl = udhr.raw("Polish-Latin2")[:164]
        text_ppl = udhr.raw("Polish_Polski-Latin2")[:164]
        expected = """POWSZECHNA DEKLARACJA PRAW CZŁOWIEKA
[Preamble]
Trzecia Sesja Ogólnego Zgromadzenia ONZ, obradująca w Paryżu, \
uchwaliła 10 grudnia 1948 roku jednomyślnie Powszechną"""
        assert text_pl == expected, "Polish-Latin2"
        assert text_ppl == expected, "Polish_Polski-Latin2"


class TestIndian(unittest.TestCase):
    def test_words(self):
        words = indian.words()[:3]
        self.assertEqual(words, ["মহিষের", "সন্তান", ":"])

    def test_tagged_words(self):
        tagged_words = indian.tagged_words()[:3]
        self.assertEqual(
            tagged_words, [("মহিষের", "NN"), ("সন্তান", "NN"), (":", "SYM")]
        )


class TestCess(unittest.TestCase):
    def test_catalan(self):
        words = cess_cat.words()[:15]
        txt = "El Tribunal_Suprem -Fpa- TS -Fpt- ha confirmat la condemna a quatre anys d' inhabilitació especial"
        self.assertEqual(words, txt.split())
        self.assertEqual(cess_cat.tagged_sents()[0][34][0], "càrrecs")

    def test_esp(self):
        words = cess_esp.words()[:15]
        txt = "El grupo estatal Electricité_de_France -Fpa- EDF -Fpt- anunció hoy , jueves , la compra del"
        self.assertEqual(words, txt.split())
        self.assertEqual(cess_esp.words()[115], "años")


class TestFloresta(unittest.TestCase):
    def test_words(self):
        words = floresta.words()[:10]
        txt = "Um revivalismo refrescante O 7_e_Meio é um ex-libris de a"
        self.assertEqual(words, txt.split())


class TestSinicaTreebank(unittest.TestCase):
    def test_sents(self):
        first_3_sents = sinica_treebank.sents()[:3]
        self.assertEqual(
            first_3_sents, [["一"], ["友情"], ["嘉珍", "和", "我", "住在", "同一條", "巷子"]]
        )

    def test_parsed_sents(self):
        parsed_sents = sinica_treebank.parsed_sents()[25]
        self.assertEqual(
            parsed_sents,
            Tree(
                "S",
                [
                    Tree("NP", [Tree("Nba", ["嘉珍"])]),
                    Tree("V‧地", [Tree("VA11", ["不停"]), Tree("DE", ["的"])]),
                    Tree("VA4", ["哭泣"]),
                ],
            ),
        )


class TestCoNLL2007(unittest.TestCase):
    # Reading the CoNLL 2007 Dependency Treebanks

    def test_sents(self):
        sents = conll2007.sents("esp.train")[0]
        self.assertEqual(
            sents[:6], ["El", "aumento", "del", "índice", "de", "desempleo"]
        )

    def test_parsed_sents(self):

        parsed_sents = conll2007.parsed_sents("esp.train")[0]

        self.assertEqual(
            parsed_sents.tree(),
            Tree(
                "fortaleció",
                [
                    Tree(
                        "aumento",
                        [
                            "El",
                            Tree(
                                "del",
                                [
                                    Tree(
                                        "índice",
                                        [
                                            Tree(
                                                "de",
                                                [Tree("desempleo", ["estadounidense"])],
                                            )
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    "hoy",
                    "considerablemente",
                    Tree(
                        "al",
                        [
                            Tree(
                                "euro",
                                [
                                    Tree(
                                        "cotizaba",
                                        [
                                            ",",
                                            "que",
                                            Tree("a", [Tree("15.35", ["las", "GMT"])]),
                                            "se",
                                            Tree(
                                                "en",
                                                [
                                                    Tree(
                                                        "mercado",
                                                        [
                                                            "el",
                                                            Tree("de", ["divisas"]),
                                                            Tree("de", ["Fráncfort"]),
                                                        ],
                                                    )
                                                ],
                                            ),
                                            Tree("a", ["0,9452_dólares"]),
                                            Tree(
                                                "frente_a",
                                                [
                                                    ",",
                                                    Tree(
                                                        "0,9349_dólares",
                                                        [
                                                            "los",
                                                            Tree(
                                                                "de",
                                                                [
                                                                    Tree(
                                                                        "mañana",
                                                                        ["esta"],
                                                                    )
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    )
                                ],
                            )
                        ],
                    ),
                    ".",
                ],
            ),
        )


@pytest.mark.skipif(
    not ptb.fileids(),
    reason="A full installation of the Penn Treebank is not available",
)
class TestPTB(unittest.TestCase):
    def test_fileids(self):
        self.assertEqual(
            ptb.fileids()[:4],
            [
                "BROWN/CF/CF01.MRG",
                "BROWN/CF/CF02.MRG",
                "BROWN/CF/CF03.MRG",
                "BROWN/CF/CF04.MRG",
            ],
        )

    def test_words(self):
        self.assertEqual(
            ptb.words("WSJ/00/WSJ_0003.MRG")[:7],
            ["A", "form", "of", "asbestos", "once", "used", "*"],
        )

    def test_tagged_words(self):
        self.assertEqual(
            ptb.tagged_words("WSJ/00/WSJ_0003.MRG")[:3],
            [("A", "DT"), ("form", "NN"), ("of", "IN")],
        )

    def test_categories(self):
        self.assertEqual(
            ptb.categories(),
            [
                "adventure",
                "belles_lettres",
                "fiction",
                "humor",
                "lore",
                "mystery",
                "news",
                "romance",
                "science_fiction",
            ],
        )

    def test_news_fileids(self):
        self.assertEqual(
            ptb.fileids("news")[:3],
            ["WSJ/00/WSJ_0001.MRG", "WSJ/00/WSJ_0002.MRG", "WSJ/00/WSJ_0003.MRG"],
        )

    def test_category_words(self):
        self.assertEqual(
            ptb.words(categories=["humor", "fiction"])[:6],
            ["Thirty-three", "Scotty", "did", "not", "go", "back"],
        )


@pytest.mark.skip("Skipping test for mwa_ppdb.")
class TestMWAPPDB(unittest.TestCase):
    def test_fileids(self):
        self.assertEqual(
            mwa_ppdb.fileids(), ["ppdb-1.0-xxxl-lexical.extended.synonyms.uniquepairs"]
        )

    def test_entries(self):
        self.assertEqual(
            mwa_ppdb.entries()[:10],
            [
                ("10/17/01", "17/10/2001"),
                ("102,70", "102.70"),
                ("13,53", "13.53"),
                ("3.2.5.3.2.1", "3.2.5.3.2.1."),
                ("53,76", "53.76"),
                ("6.9.5", "6.9.5."),
                ("7.7.6.3", "7.7.6.3."),
                ("76,20", "76.20"),
                ("79,85", "79.85"),
                ("93,65", "93.65"),
            ],
        )
