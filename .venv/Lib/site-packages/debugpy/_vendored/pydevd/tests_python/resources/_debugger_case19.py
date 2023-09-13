class A:
    
    def __init__(self):
        self.__var = 10

if __name__ == '__main__':
    a = A()
    print(a._A__var)
    # Evaluate 'a.__var' should give a._A__var_
    print('TEST SUCEEDED')
