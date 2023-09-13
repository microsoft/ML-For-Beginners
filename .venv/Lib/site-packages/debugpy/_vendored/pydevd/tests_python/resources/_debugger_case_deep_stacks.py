def method1(n):
    if n <= 0:
        return 0  # Break here
    method2(n - 1)


def method2(n):
    method1(n - 1)


if __name__ == '__main__':
    try:
        method1(100)
    except:
        pass  # Don't let it print the exception (just deal with caught exceptions).
    print('TEST SUCEEDED!')
