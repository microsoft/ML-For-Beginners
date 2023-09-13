def Method1():
    print('m1')

def Method2():
    print('m2 before')
    Method1()
    print('m2 after')

def Method3():
    print('m3 before')
    Method2()
    print('m3 after')
   
if __name__ == '__main__': 
    Method3()
    print('TEST SUCEEDED!')
