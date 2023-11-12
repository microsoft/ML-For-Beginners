def setup_module():
    import pytest

    from nltk.parse.malt import MaltParser

    try:
        depparser = MaltParser()
    except (AssertionError, LookupError) as e:
        pytest.skip("MaltParser is not available")
