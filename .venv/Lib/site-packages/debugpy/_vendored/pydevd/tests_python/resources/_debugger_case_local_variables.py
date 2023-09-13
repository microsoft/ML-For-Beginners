

def Call():
    variable_for_test_1 = 10  # Break here
    variable_for_test_2 = 20
    variable_for_test_3 = {'a':30, 'b':20}
    locals()[u'\u16A0'] = u'\u16A1'  # unicode variable (would be syntax error on py2).

    all_vars_set = True  # Break 2 here


if __name__ == '__main__':
    Call()
    print('TEST SUCEEDED!')
