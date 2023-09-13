import sys

def m2(a):
    a = 10
    b = 20 #Break here and set a = 40
    c = 30
    
    def function2():
        print(a)

    return a


def m1(a):
    return m2(a)


if __name__ == '__main__':
    found = m1(10)
    if found == 40:
        print('TEST SUCEEDED')
    else:
        raise AssertionError('Expected variable to be changed to 40. Found: %s' % (found,))
