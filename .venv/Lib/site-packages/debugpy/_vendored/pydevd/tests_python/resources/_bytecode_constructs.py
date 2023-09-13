from contextlib import contextmanager


def method1():

    _a = 0
    while _a < 2:  # break while
        _a += 1


def method2():
    try:
        raise AssertionError()
    except:  # break except
        pass


@contextmanager
def ctx():
    yield ''


def method3():
    with ctx() as a:  # break with
        return a


def method4():
    _a = 0
    for i in range(2):  # break for
        _a = i


def method5():
    try:  # break try 1
        _a = 10
    finally:
        _b = 10


def method6():
    try:
        _a = 10  # break try 2
    finally:
        _b = 10


def method7():
    try:
        _a = 10
    finally:
        _b = 10  # break finally 1


def method8():
    try:
        raise AssertionError()
    except:  # break except 2
        _b = 10
    finally:
        _c = 20


def method9():
    # As a note, Python 3.10 is eager to optimize this case and it duplicates the _c = 20
    # in a codepath where the exception is raised and another where it's not raised.
    # The frame eval mode must modify the bytecode so that both paths have the
    # programmatic breakpoint added!
    try:
        _a = 10
    except:
        _b = 10
    finally:_c = 20  # break finally 2


def method9a():
    # Same as method9, but with exception raised (but handled).
    try:
        raise AssertionError()
    except:
        _b = 10
    finally:_c = 20  # break finally 3


def method9b():
    # Same as method9, but with exception raised (unhandled).
    try:
        try:
            raise RuntimeError()
        except AssertionError:
            _b = 10
        finally:_c = 20  # break finally 4
    except:
        pass


def method10():
    _a = {
        0: 0,
        1: 1,  # break in dict
        2: 2,
    }


def method11():
    a = 11
    if a == 10:
        a = 20
    else: a = 30  # break else


if __name__ == '__main__':
    method1()
    method2()
    method3()
    method4()
    method5()
    method6()
    method7()
    method8()
    method9()
    method9a()
    method9b()
    method10()
    method11()
    print('TEST SUCEEDED')
