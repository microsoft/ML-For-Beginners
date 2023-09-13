def generator2():
    yield from range(4)


def generator():
    a = 42  # break here
    yield from generator2()


sum = 0
for i in generator():
    sum += i

print('TEST SUCEEDED!')
