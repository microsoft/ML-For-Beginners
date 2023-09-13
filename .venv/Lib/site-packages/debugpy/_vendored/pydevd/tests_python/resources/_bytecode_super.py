class A(object):

    def __init__(self):
        self.a = 10


class B(A):

    def __init__(self):
        super().__init__()  # break here
        assert self.a == 10

        def method():
            self.b = self.a

        method()
        assert self.b == 10


if __name__ == '__main__':
    B()
    print('TEST SUCEEDED')
