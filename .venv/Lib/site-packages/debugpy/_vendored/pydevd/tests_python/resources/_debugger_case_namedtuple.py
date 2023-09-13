from collections import namedtuple
MyTup = namedtuple('MyTup', 'a, b, c')
tup = MyTup(1, 2, 3)  # break here
assert tup.a == 1
assert tup.b == 2
print('TEST SUCEEDED!')
