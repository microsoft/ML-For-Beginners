def full_function():
    # Note that this function is not called, it's there just to make the mapping explicit.  # map to cEll1, line 1
    import sys  # map to cEll1, line 2
    frame = sys._getframe()  # map to cEll1, line 3
    if py_db.in_project_scope(frame, '<cEll1>') != expect_in_project_scope:  # map to cEll1, line 4
        raise AssertionError('Expected <cEll1> to be in project scope: %s' % (expect_in_project_scope,))  # map to cEll1, line 5
    a = 1  # map to cEll1, line 6
    b = 2  # map to cEll1, line 7


def create_code():
    cEll1_code = compile(''' # line 1
import sys # line 2
frame = sys._getframe() # line 3
if py_db.in_project_scope(frame, '<cEll1>') != expect_in_project_scope: # line 4
    raise AssertionError('Expected <cEll1> to be in project scope: %s' % (expect_in_project_scope,)) # line 5
a = 1  # line 6
b = 2  # line 7
''', '<cEll1>', 'exec')

    return {'cEll1': cEll1_code}


if __name__ == '__main__':
    code = create_code()
    import pydevd
    py_db = pydevd.get_global_debugger()

    expect_in_project_scope = True
    exec(code['cEll1'])  # When executing, stop at breakpoint and then remove the source mapping.

    expect_in_project_scope = False
    exec(code['cEll1'])  # Should no longer stop.

    print('TEST SUCEEDED')
