

def foo(arg):
    print('on foo mark', arg)
    return arg + 1


def main():
    # Note that we have multiple foo calls and we have to differentiate and stop at the
    # proper one.
    foo(foo(foo(foo(1))))  # break here


if __name__ == '__main__':
    main()

    print('TEST SUCEEDED')
