
def break_in_method():
    breakpoint()  # Builtin on Py3, but we provide a backport on Py2.


break_in_method()
print('TEST SUCEEDED')
