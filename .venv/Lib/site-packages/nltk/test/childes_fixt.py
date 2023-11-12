def setup_module():
    import pytest

    import nltk.data

    try:
        nltk.data.find("corpora/childes/data-xml/Eng-USA-MOR/")
    except LookupError as e:
        pytest.skip(
            "The CHILDES corpus is not found. "
            "It should be manually downloaded and saved/unpacked "
            "to [NLTK_Data_Dir]/corpora/childes/"
        )
