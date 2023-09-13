def main():
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from not_my_code import other

    def callback2():
        try:raise AssertionError()
        except:return False

        return True

    def callback1():
        other.call_me_back2(callback2)

    other.call_me_back1(callback1)
    print('TEST SUCEEDED!')


if __name__ == '__main__':
    main()
