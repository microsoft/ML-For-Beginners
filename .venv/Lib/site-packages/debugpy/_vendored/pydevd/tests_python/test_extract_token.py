# coding: utf-8
from __future__ import unicode_literals
from _pydev_bundle._pydev_completer import (isidentifier, extract_token_and_qualifier,
    TokenAndQualifier)


def test_isidentifier():
    assert isidentifier('abc')
    assert not isidentifier('<')
    assert not isidentifier('')
    assert isidentifier('áéíóú')


def test_extract_token_and_qualifier():

    assert extract_token_and_qualifier('tok', 0, 0) == TokenAndQualifier('', '')
    assert extract_token_and_qualifier('tok', 0, 1) == TokenAndQualifier('', 't')
    assert extract_token_and_qualifier('tok', 0, 2) == TokenAndQualifier('', 'to')
    assert extract_token_and_qualifier('tok', 0, 3) == TokenAndQualifier('', 'tok')
    assert extract_token_and_qualifier('tok', 0, 4) == TokenAndQualifier('', 'tok')

    assert extract_token_and_qualifier('tok.qual', 0, 0) == TokenAndQualifier('', '')
    assert extract_token_and_qualifier('tok.qual', 0, 1) == TokenAndQualifier('', 't')
    assert extract_token_and_qualifier('tok.qual', 0, 2) == TokenAndQualifier('', 'to')
    assert extract_token_and_qualifier('tok.qual', 0, 3) == TokenAndQualifier('', 'tok')

    assert extract_token_and_qualifier('tok.qual', 0, 4) == TokenAndQualifier('tok', '')
    assert extract_token_and_qualifier('tok.qual', 0, 5) == TokenAndQualifier('tok', 'q')
    assert extract_token_and_qualifier('tok.qual', 0, 6) == TokenAndQualifier('tok', 'qu')
    assert extract_token_and_qualifier('tok.qual', 0, 7) == TokenAndQualifier('tok', 'qua')
    assert extract_token_and_qualifier('tok.qual', 0, 8) == TokenAndQualifier('tok', 'qual')

    # out of range (column)
    assert extract_token_and_qualifier('tok.qual.qual2', 0, 100) == TokenAndQualifier('tok.qual', 'qual2')

    assert extract_token_and_qualifier('t<ok', 0, 0) == TokenAndQualifier('', '')
    assert extract_token_and_qualifier('t<ok', 0, 1) == TokenAndQualifier('', 't')
    assert extract_token_and_qualifier('t<ok', 0, 2) == TokenAndQualifier('', '')
    assert extract_token_and_qualifier('t<ok', 0, 3) == TokenAndQualifier('', 'o')
    assert extract_token_and_qualifier('t<ok', 0, 4) == TokenAndQualifier('', 'ok')

    assert extract_token_and_qualifier('a\nt<ok', 1, 0) == TokenAndQualifier('', '')
    assert extract_token_and_qualifier('a\nt<ok', 1, 1) == TokenAndQualifier('', 't')
    assert extract_token_and_qualifier('a\nt<ok', 1, 2) == TokenAndQualifier('', '')
    assert extract_token_and_qualifier('a\nt<ok', 1, 3) == TokenAndQualifier('', 'o')
    assert extract_token_and_qualifier('a\nt<ok', 1, 4) == TokenAndQualifier('', 'ok')

    # out of range (line)
    assert extract_token_and_qualifier('a\nt<ok', 5, 4) == TokenAndQualifier('', '')
