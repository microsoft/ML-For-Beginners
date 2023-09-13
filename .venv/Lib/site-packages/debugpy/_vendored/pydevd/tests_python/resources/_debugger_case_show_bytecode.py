def foo():
    return 1  # breakpoint


def main():

    code = '''

class MyClass(object):
    def method(self):
        if True:
            foo()

MyClass().method()
'''

    co = compile(code, '<something>', 'exec')

    # Intermediate <something> stack frame will have source.
    eval(co)


if __name__ == '__main__':
    main()
    print('TEST SUCEEDED!')
