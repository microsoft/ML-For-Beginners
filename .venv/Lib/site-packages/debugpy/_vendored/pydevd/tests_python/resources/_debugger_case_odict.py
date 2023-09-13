def check():
    from collections import OrderedDict
    import sys
    # On 3.6 onwards, use a regular dict.
    odict = OrderedDict() if sys.version_info[:2] < (3, 6) else {}
    odict[4] = 'first'
    odict[3] = 'second'
    odict[2] = 'last'
    print('break here')


if __name__ == '__main__':
    check()
print('TEST SUCEEDED')
