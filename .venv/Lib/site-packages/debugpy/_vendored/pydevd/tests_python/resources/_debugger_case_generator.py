

def generator2():
    for i in range(5):
        yield i


def generator():
    print('start')  # break here
    yield from generator2()  # step 1
    print('end')  # step 2


if __name__ == '__main__':
    for i in generator():  # generator return
        print(i)

    print('TEST SUCEEDED!')  # step 3
