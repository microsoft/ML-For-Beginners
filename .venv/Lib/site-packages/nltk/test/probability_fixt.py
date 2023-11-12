# probability.doctest uses HMM which requires numpy;
# skip probability.doctest if numpy is not available


def setup_module():
    import pytest

    pytest.importorskip("numpy")
