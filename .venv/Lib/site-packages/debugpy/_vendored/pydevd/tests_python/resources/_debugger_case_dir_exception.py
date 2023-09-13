class A():

    def __init__(self):
        self.var1 = 10
        self.attr = {}  # Break here

    def __dir__(self):
        return list(self.attr)


if __name__ == '__main__':
    a = A()
    print('TEST SUCEEDED!')
