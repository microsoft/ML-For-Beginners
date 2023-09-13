def method1():
    yield
    print('here')  # Break here


if __name__ == '__main__':
    # i.e.: make sure we create 2 frames with different frameIds.
    it1 = iter(method1())
    it2 = iter(method1())

    next(it1)  # resume first
    next(it2)  # resume second

    try:
        next(it1)  # finish first
    except StopIteration:
        pass
    try:
        next(it2)  # finish second
    except StopIteration:
        pass

    print('TEST SUCEEDED')
