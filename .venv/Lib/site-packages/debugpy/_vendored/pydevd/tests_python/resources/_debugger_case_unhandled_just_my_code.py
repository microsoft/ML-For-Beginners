import sys

filename = sys.argv[1]

obj = compile('''
def call_exception_in_exec():
    a = 10
    b = 20
    raise Exception('TEST SUCEEDED')
''', filename, 'exec')

exec(obj)
call_exception_in_exec()  # @UndefinedVariable
