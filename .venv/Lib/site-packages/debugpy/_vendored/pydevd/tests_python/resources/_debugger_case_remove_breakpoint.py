

def check():
    import pydevd
    py_db = pydevd.get_global_debugger()
    assert len(py_db.api_received_breakpoints) == 1
    _a = 10  # break here
    # should remove the breakpoints when stopped on the previous line
    assert len(py_db.api_received_breakpoints) == 0


if __name__ == '__main__':
    check()
    print('TEST SUCEEDED!')
