in_global_scope = 'in_global_scope_value'


class SomeClass(object):

    def method(self):
        print('breakpoint here')


if __name__ == '__main__':
    SomeClass().method()
    print('TEST SUCEEDED')
