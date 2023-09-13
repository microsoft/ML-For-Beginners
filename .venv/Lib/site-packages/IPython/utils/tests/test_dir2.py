from IPython.utils.dir2 import dir2

import pytest


class Base(object):
    x = 1
    z = 23


def test_base():
    res = dir2(Base())
    assert "x" in res
    assert "z" in res
    assert "y" not in res
    assert "__class__" in res
    assert res.count("x") == 1
    assert res.count("__class__") == 1


def test_SubClass():
    class SubClass(Base):
        y = 2

    res = dir2(SubClass())
    assert "y" in res
    assert res.count("y") == 1
    assert res.count("x") == 1


def test_SubClass_with_trait_names_attr():
    # usecase: trait_names is used in a class describing psychological classification

    class SubClass(Base):
        y = 2
        trait_names = 44

    res = dir2(SubClass())
    assert "trait_names" in res


def test_misbehaving_object_without_trait_names():
    # dir2 shouldn't raise even when objects are dumb and raise
    # something other than AttribteErrors on bad getattr.

    class MisbehavingGetattr:
        def __getattr__(self, attr):
            raise KeyError("I should be caught")

        def some_method(self):
            return True

    class SillierWithDir(MisbehavingGetattr):
        def __dir__(self):
            return ["some_method"]

    for bad_klass in (MisbehavingGetattr, SillierWithDir):
        obj = bad_klass()

        assert obj.some_method()

        with pytest.raises(KeyError):
            obj.other_method()

        res = dir2(obj)
        assert "some_method" in res
