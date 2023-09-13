def full_function():
    # Note that this function is not called, it's there just to make the mapping explicit.
    a = 1  # map to cEll1, line 2
    b = 2  # map to cEll1, line 3

    c = 3  # map to cEll2, line 2
    d = 4  # map to cEll2, line 3


def create_code():
    cell1_code = compile(''' # line 1
a = 1  # line 2
b = 2  # line 3
''', '<cEll1>', 'exec')

    cell2_code = compile('''# line 1
c = 3  # line 2
d = 4  # line 3
''', '<cEll2>', 'exec')

    return {'cEll1': cell1_code, 'cEll2': cell2_code}


if __name__ == '__main__':
    code = create_code()
    exec(code['cEll1'])
    exec(code['cEll1'])

    exec(code['cEll2'])
    exec(code['cEll2'])
    print('TEST SUCEEDED')
