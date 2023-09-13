def bar():
    print('on bar mark')


def call_outer(*args):
    print('on outer mark')


def foo(*args):
    print('on foo mark')


def main():
    call_outer(foo(bar()))  # break here


if __name__ == '__main__':
    main()

    print('TEST SUCEEDED')
