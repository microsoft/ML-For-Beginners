if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from not_my_code import other_noop
    other_noop.call_noop()  # break here
    print('TEST SUCEEDED!')
