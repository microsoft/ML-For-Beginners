# most of classify.doctest requires numpy
def setup_module():
    import pytest

    pytest.importorskip("numpy")
