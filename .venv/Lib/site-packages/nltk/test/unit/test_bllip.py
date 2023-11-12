import pytest

from nltk.data import find
from nltk.parse.bllip import BllipParser
from nltk.tree import Tree


@pytest.fixture(scope="module")
def parser():
    model_dir = find("models/bllip_wsj_no_aux").path
    return BllipParser.from_unified_model_dir(model_dir)


def setup_module():
    pytest.importorskip("bllipparser")


class TestBllipParser:
    def test_parser_loads_a_valid_tree(self, parser):
        parsed = parser.parse("I saw the man with the telescope")
        tree = next(parsed)

        assert isinstance(tree, Tree)
        assert (
            tree.pformat()
            == """
(S1
  (S
    (NP (PRP I))
    (VP
      (VBD saw)
      (NP (DT the) (NN man))
      (PP (IN with) (NP (DT the) (NN telescope))))))
""".strip()
        )

    def test_tagged_parse_finds_matching_element(self, parser):
        parsed = parser.parse("I saw the man with the telescope")
        tagged_tree = next(parser.tagged_parse([("telescope", "NN")]))

        assert isinstance(tagged_tree, Tree)
        assert tagged_tree.pformat() == "(S1 (NP (NN telescope)))"
