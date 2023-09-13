def generator2():
    for i in range(4):
        yield i


def generator():
    a = 42  # break here
    for x in generator2():
        yield x


sum = 0
for i in generator():
    sum += i

print('TEST SUCEEDED!')
