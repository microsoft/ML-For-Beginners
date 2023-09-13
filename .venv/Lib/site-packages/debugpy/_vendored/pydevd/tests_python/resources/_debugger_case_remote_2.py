if __name__ == '__main__':
    print('Run as main: %s' % (__file__,))
    import sys
    sys.stdout.flush()
    import pydevd
    # Just check that we're already connected
    assert pydevd.GetGlobalDebugger() is not None
    print('finish')
    sys.stdout.flush()
    print('TEST SUCEEDED!')
    sys.stdout.flush()