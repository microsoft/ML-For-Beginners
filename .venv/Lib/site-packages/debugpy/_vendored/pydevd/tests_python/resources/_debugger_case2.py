
def Call4():
    print('Start Call4')
    print('End Call4')

def Call3():
    print('Start Call3')
    Call4()
    print('End Call3')

def Call2():
    print('Start Call2')
    Call3()
    print('End Call2 - a')
    print('End Call2 - b')

def Call1():
    print('Start Call1')
    Call2()
    print('End Call1')

if __name__ == '__main__':
    Call1()
    print('TEST SUCEEDED!')
