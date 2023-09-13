def method():
    a = 1 # Step here
    print('call %s' % (a,))
    a = 2
    print('call %s' % (a,))
    a = 3 # Break here

if __name__ == '__main__':
    method()
    print('TEST SUCEEDED!')
