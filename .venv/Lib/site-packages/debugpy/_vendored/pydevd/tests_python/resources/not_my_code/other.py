def call_me_back2(callback):
    a = 'other'
    callback()
    return a


def call_me_back1(callback):
    a = 'other'
    callback()
    return a


def raise_exception():
    raise RuntimeError('TEST SUCEEDED')

