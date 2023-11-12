import contextlib
import sys
import unittest
from io import StringIO

from nltk.corpus import gutenberg
from nltk.text import Text


@contextlib.contextmanager
def stdout_redirect(where):
    sys.stdout = where
    try:
        yield where
    finally:
        sys.stdout = sys.__stdout__


class TestConcordance(unittest.TestCase):
    """Text constructed using: https://www.nltk.org/book/ch01.html"""

    @classmethod
    def setUpClass(cls):
        cls.corpus = gutenberg.words("melville-moby_dick.txt")

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.text = Text(TestConcordance.corpus)
        self.query = "monstrous"
        self.maxDiff = None
        self.list_out = [
            "ong the former , one was of a most monstrous size . ... This came towards us , ",
            'ON OF THE PSALMS . " Touching that monstrous bulk of the whale or ork we have r',
            "ll over with a heathenish array of monstrous clubs and spears . Some were thick",
            "d as you gazed , and wondered what monstrous cannibal and savage could ever hav",
            "that has survived the flood ; most monstrous and most mountainous ! That Himmal",
            "they might scout at Moby Dick as a monstrous fable , or still worse and more de",
            "th of Radney .'\" CHAPTER 55 Of the Monstrous Pictures of Whales . I shall ere l",
            "ing Scenes . In connexion with the monstrous pictures of whales , I am strongly",
            "ere to enter upon those still more monstrous stories of them which are to be fo",
            "ght have been rummaged out of this monstrous cabinet there is no telling . But ",
            "of Whale - Bones ; for Whales of a monstrous size are oftentimes cast up dead u",
        ]

    def tearDown(self):
        pass

    def test_concordance_list(self):
        concordance_out = self.text.concordance_list(self.query)
        self.assertEqual(self.list_out, [c.line for c in concordance_out])

    def test_concordance_width(self):
        list_out = [
            "monstrous",
            "monstrous",
            "monstrous",
            "monstrous",
            "monstrous",
            "monstrous",
            "Monstrous",
            "monstrous",
            "monstrous",
            "monstrous",
            "monstrous",
        ]

        concordance_out = self.text.concordance_list(self.query, width=0)
        self.assertEqual(list_out, [c.query for c in concordance_out])

    def test_concordance_lines(self):
        concordance_out = self.text.concordance_list(self.query, lines=3)
        self.assertEqual(self.list_out[:3], [c.line for c in concordance_out])

    def test_concordance_print(self):
        print_out = """Displaying 11 of 11 matches:
        ong the former , one was of a most monstrous size . ... This came towards us ,
        ON OF THE PSALMS . " Touching that monstrous bulk of the whale or ork we have r
        ll over with a heathenish array of monstrous clubs and spears . Some were thick
        d as you gazed , and wondered what monstrous cannibal and savage could ever hav
        that has survived the flood ; most monstrous and most mountainous ! That Himmal
        they might scout at Moby Dick as a monstrous fable , or still worse and more de
        th of Radney .'" CHAPTER 55 Of the Monstrous Pictures of Whales . I shall ere l
        ing Scenes . In connexion with the monstrous pictures of whales , I am strongly
        ere to enter upon those still more monstrous stories of them which are to be fo
        ght have been rummaged out of this monstrous cabinet there is no telling . But
        of Whale - Bones ; for Whales of a monstrous size are oftentimes cast up dead u
        """

        with stdout_redirect(StringIO()) as stdout:
            self.text.concordance(self.query)

        def strip_space(raw_str):
            return raw_str.replace(" ", "")

        self.assertEqual(strip_space(print_out), strip_space(stdout.getvalue()))
