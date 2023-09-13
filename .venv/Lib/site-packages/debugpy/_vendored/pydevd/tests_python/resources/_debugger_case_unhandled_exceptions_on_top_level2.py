try:
    raise ValueError('foobar')
except ValueError:
    pass

raise ValueError('TEST SUCEEDED')