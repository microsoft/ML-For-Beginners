if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    def call():
        return 1

    call()
    # Check that we're not setting return values here.
    assert '__pydevd_ret_val_dict' not in sys._getframe().f_locals

    try:
        import enum
    except ImportError:
        pass
    else:

        # i.e.: this could fail if a return value was traced
        class MyCode(enum.IntEnum):
            A = 10

    import empty_file
    print('TEST SUCEEDED')
