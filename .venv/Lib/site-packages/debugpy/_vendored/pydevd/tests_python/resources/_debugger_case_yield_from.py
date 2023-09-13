def method():
    a = [1, 2]

    def b():  # break1
        yield from [j for j in a if j % 2 == 0]  # break2

    for j in b():
        print(j)


method()
print('TEST SUCEEDED')
