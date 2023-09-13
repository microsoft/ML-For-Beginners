import pickle
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ipykernel.pickleutil import can, uncan


def interactive(f):
    f.__module__ = "__main__"
    return f


def dumps(obj):
    return pickle.dumps(can(obj))


def loads(obj):
    return uncan(pickle.loads(obj))  # noqa


def test_no_closure():
    @interactive
    def foo():
        a = 5
        return a

    pfoo = dumps(foo)
    bar = loads(pfoo)
    assert foo() == bar()


def test_generator_closure():
    # this only creates a closure on Python 3
    @interactive
    def foo():
        i = "i"
        r = [i for j in (1, 2)]
        return r

    pfoo = dumps(foo)
    bar = loads(pfoo)
    assert foo() == bar()


def test_nested_closure():
    @interactive
    def foo():
        i = "i"

        def g():
            return i

        return g()

    pfoo = dumps(foo)
    bar = loads(pfoo)
    assert foo() == bar()


def test_closure():
    i = "i"

    @interactive
    def foo():
        return i

    pfoo = dumps(foo)
    bar = loads(pfoo)
    assert foo() == bar()


def test_uncan_bytes_buffer():
    data = b"data"
    canned = can(data)
    canned.buffers = [memoryview(buf) for buf in canned.buffers]
    out = uncan(canned)
    assert out == data
