def setup_module():
    import pytest

    pytest.importorskip("gensim")
