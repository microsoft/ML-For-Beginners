from ..bunch import Bunch


def test_bunch():
    b = Bunch(x=5, y=10)
    assert "y" in b
    assert "x" in b
    assert b.x == 5
    b["a"] = "hi"
    assert b.a == "hi"


def test_bunch_dir():
    b = Bunch(x=5, y=10)
    assert "x" in dir(b)
    assert "keys" in dir(b)
