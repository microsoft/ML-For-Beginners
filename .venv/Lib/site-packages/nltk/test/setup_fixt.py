from nltk.internals import find_binary, find_jar


def check_binary(binary: str, **args):
    """Skip a test via `pytest.skip` if the `binary` executable is not found.
    Keyword arguments are passed to `nltk.internals.find_binary`."""
    import pytest

    try:
        find_binary(binary, **args)
    except LookupError:
        pytest.skip(f"Skipping test because the {binary} binary was not found.")


def check_jar(name_pattern: str, **args):
    """Skip a test via `pytest.skip` if the `name_pattern` jar is not found.
    Keyword arguments are passed to `nltk.internals.find_jar`.

    TODO: Investigate why the CoreNLP tests that rely on this check_jar failed
    on the CI. https://github.com/nltk/nltk/pull/3060#issuecomment-1268355108
    """
    import pytest

    pytest.skip(
        "Skipping test because the doctests requiring jars are inconsistent on the CI."
    )
