import os
from io import BytesIO

import pytest

from nltk.corpus.reader import SeekableUnicodeStreamReader


def check_reader(unicode_string, encoding):
    bytestr = unicode_string.encode(encoding)
    stream = BytesIO(bytestr)
    reader = SeekableUnicodeStreamReader(stream, encoding)

    # Should open at the start of the file
    assert reader.tell() == 0

    # Compare original string to contents from `.readlines()`
    assert unicode_string == "".join(reader.readlines())

    # Should be at the end of the file now
    stream.seek(0, os.SEEK_END)
    assert reader.tell() == stream.tell()

    reader.seek(0)  # go back to start

    # Compare original string to contents from `.read()`
    contents = ""
    char = None
    while char != "":
        char = reader.read(1)
        contents += char
    assert unicode_string == contents


# Call `check_reader` with a variety of input strings and encodings.
ENCODINGS = ["ascii", "latin1", "greek", "hebrew", "utf-16", "utf-8"]

STRINGS = [
    """
    This is a test file.
    It is fairly short.
    """,
    "This file can be encoded with latin1. \x83",
    """\
    This is a test file.
    Here's a blank line:

    And here's some unicode: \xee \u0123 \uffe3
    """,
    """\
    This is a test file.
    Unicode characters: \xf3 \u2222 \u3333\u4444 \u5555
    """,
    """\
    This is a larger file.  It has some lines that are longer \
    than 72 characters.  It's got lots of repetition.  Here's \
    some unicode chars: \xee \u0123 \uffe3 \ueeee \u2345

    How fun!  Let's repeat it twenty times.
    """
    * 20,
]


@pytest.mark.parametrize("string", STRINGS)
def test_reader(string):
    for encoding in ENCODINGS:
        # skip strings that can't be encoded with the current encoding
        try:
            string.encode(encoding)
        except UnicodeEncodeError:
            continue
        check_reader(string, encoding)


def test_reader_stream_closes_when_deleted():
    reader = SeekableUnicodeStreamReader(BytesIO(b""), "ascii")
    assert not reader.stream.closed
    reader.__del__()
    assert reader.stream.closed


def teardown_module(module=None):
    import gc

    gc.collect()
