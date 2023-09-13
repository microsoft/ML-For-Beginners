from typing import List

import pytest
import pygments.lexers
import pygments.lexer

from IPython.lib.lexers import IPythonConsoleLexer, IPythonLexer, IPython3Lexer

#: the human-readable names of the IPython lexers with ``entry_points``
EXPECTED_LEXER_NAMES = [
    cls.name for cls in [IPythonConsoleLexer, IPythonLexer, IPython3Lexer]
]


@pytest.fixture
def all_pygments_lexer_names() -> List[str]:
    """Get all lexer names registered in pygments."""
    return {l[0] for l in pygments.lexers.get_all_lexers()}


@pytest.mark.parametrize("expected_lexer", EXPECTED_LEXER_NAMES)
def test_pygments_entry_points(
    expected_lexer: str, all_pygments_lexer_names: List[str]
) -> None:
    """Check whether the ``entry_points`` for ``pygments.lexers`` are correct."""
    assert expected_lexer in all_pygments_lexer_names
