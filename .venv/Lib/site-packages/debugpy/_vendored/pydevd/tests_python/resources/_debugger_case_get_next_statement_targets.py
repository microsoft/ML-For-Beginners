def method():
    a = 1
    print('call %s' % (a,))

    def method2():
        print('call %s' % (a,))

    while a < 10:
        a += 1
        print('call %s' % (a,))

    try:
        if a < 0:
            print('call %s' % (a,))
            raise ValueError
        else:
            method2()
    except ValueError:
        pass

    print('call %s' % (a,))


if __name__ == '__main__':
    method()
    print('TEST SUCEEDED!')