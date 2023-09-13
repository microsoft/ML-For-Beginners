def m1():
    _a = 'm1'  # break 1 here


def m2():  # @DontTrace
    m1()
    _a = 'm2'


def m3():
    m2()
    _a = 'm3'  # break 2 here


if __name__ == '__main__':
    m3()

    print('TEST SUCEEDED')
