# Some comment lines to move the function below
# Some comment lines to move the function below
# Some comment lines to move the function below
# Some comment lines to move the function below


def full_function():
    # Note that this function is not called, it's there just to make the mapping explicit.
    # The test case should stop at `a = 1` and then skip the `print('Skip this print')`.
    # map to Cell1, line 1
    a = 1  # map to Cell1, line 2
    print('Skip this print')  # map to Cell1, line 3
    print('TEST SUCEEDED')  # map to Cell1, line 4
    b = 2  # map to Cell1, line 5


if __name__ == '__main__':
    code = compile('''# line 1
a = 1  # line 2
print('Skip this print')  # line 3
print('TEST SUCEEDED')  # line 4
b = 2  # line 5
''', '<Cell1>', 'exec')
    exec(code)
