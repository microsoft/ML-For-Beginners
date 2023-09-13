def Method1():
    print('m1')
    print('m1')
    
def Method1a():
    print('m1a')
    print('m1a')

def Method2():
    print('m2 before')
    Method1()
    Method1a()
    print('m2 after')

   
if __name__ == '__main__': 
    Method2()
    print('TEST SUCEEDED!')
