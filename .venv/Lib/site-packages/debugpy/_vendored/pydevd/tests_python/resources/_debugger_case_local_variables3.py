class MyDictSubclass(dict):

    def __init__(self):
        dict.__init__(self)
        self.var1 = 10
        self['in_dct'] = 20

    def __str__(self):
        ret = []
        for key, val in sorted(self.items()):
            ret.append('%s: %s' % (key, val))
        ret.append('self.var1: %s' % (self.var1,))
        return '{' + '; '.join(ret) + '}'

    __repr__ = __str__


class MyListSubclass(list):

    def __init__(self):
        list.__init__(self)
        self.var1 = 11
        self.append('a')
        self.append('b')

    def __str__(self):
        ret = []
        for obj in self:
            ret.append(repr(obj))
        ret.append('self.var1: %s' % (self.var1,))
        return '[' + ', '.join(ret) + ']'

    __repr__ = __str__


class MySetSubclass(set):

    def __init__(self):
        set.__init__(self)
        self.var1 = 12
        self.add('a')

    def __str__(self):
        ret = []
        for obj in sorted(self):
            ret.append(repr(obj))
        ret.append('self.var1: %s' % (self.var1,))
        return 'set([' + ', '.join(ret) + '])'

    __repr__ = __str__


class MyTupleSubclass(tuple):

    def __new__ (cls):
        return super(MyTupleSubclass, cls).__new__(cls, tuple(['a', 1]))

    def __init__(self):
        self.var1 = 13

    def __str__(self):
        ret = []
        for obj in self:
            ret.append(repr(obj))
        ret.append('self.var1: %s' % (self.var1,))
        return 'tuple(' + ', '.join(ret) + ')'

    __repr__ = __str__


def Call():
    variable_for_test_1 = MyListSubclass()
    variable_for_test_2 = MySetSubclass()
    variable_for_test_3 = MyDictSubclass()
    variable_for_test_4 = MyTupleSubclass()

    all_vars_set = True  # Break here


if __name__ == '__main__':
    Call()
    print('TEST SUCEEDED!')
