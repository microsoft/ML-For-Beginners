import io
import os.path

from IPython.utils import openpy

mydir = os.path.dirname(__file__)
nonascii_path = os.path.join(mydir, "../../core/tests/nonascii.py")


def test_detect_encoding():
    with open(nonascii_path, "rb") as f:
        enc, lines = openpy.detect_encoding(f.readline)
    assert enc == "iso-8859-5"


def test_read_file():
    with io.open(nonascii_path, encoding="iso-8859-5") as f:
        read_specified_enc = f.read()
    read_detected_enc = openpy.read_py_file(nonascii_path, skip_encoding_cookie=False)
    assert read_detected_enc == read_specified_enc
    assert "coding: iso-8859-5" in read_detected_enc

    read_strip_enc_cookie = openpy.read_py_file(
        nonascii_path, skip_encoding_cookie=True
    )
    assert "coding: iso-8859-5" not in read_strip_enc_cookie


def test_source_to_unicode():
    with io.open(nonascii_path, "rb") as f:
        source_bytes = f.read()
    assert (
        openpy.source_to_unicode(source_bytes, skip_encoding_cookie=False).splitlines()
        == source_bytes.decode("iso-8859-5").splitlines()
    )

    source_no_cookie = openpy.source_to_unicode(source_bytes, skip_encoding_cookie=True)
    assert "coding: iso-8859-5" not in source_no_cookie
