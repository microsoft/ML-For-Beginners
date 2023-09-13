def method2():
    raise RuntimeError('TEST SUCEEDED')


def method():
    method2()


def handle(e):
    raise Exception('another while handling')


def foobar():
    try:
        try:
            method()
        except Exception as e:
            handle(e)
    except Exception as e:
        raise RuntimeError from e


foobar()
