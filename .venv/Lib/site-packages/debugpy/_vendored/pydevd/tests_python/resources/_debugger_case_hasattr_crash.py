class MyClass(object):

    def __getattribute__(self, attr):
        raise RuntimeError()


obj = MyClass()

if __name__ == '__main__':
    print('TEST SUCEEDED')  # break here
