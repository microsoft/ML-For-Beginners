def generator():
    yield 1  # stop 1
    yield 2  # stop 4


def main():
    for i in generator():  # stop 3
        print(i)  # stop 2


if __name__ == '__main__':
    main()
    print('TEST SUCEEDED!')
