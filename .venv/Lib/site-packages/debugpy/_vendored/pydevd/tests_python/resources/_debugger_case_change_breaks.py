

def method():
    _a = 1  # break 1
    _a = 2
    _a = 3  # break 2


if __name__ == '__main__':
    for i in range(2):
        method()
    print('TEST SUCEEDED')  # break 3
