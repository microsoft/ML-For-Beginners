def method2():
    {}['foo']


def method():
    try:
        method2()
    except Exception as e:
        raise Exception('TEST SUCEEDED') from e


method()
