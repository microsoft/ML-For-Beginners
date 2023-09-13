def foo(arg):
    print('on foo mark', arg)
    return arg + 1


def main():
    items = [1]  # break here
    gen = (foo(arg) for arg in items)
    list(gen)

# import dis
# print('-------- main ------------')
# dis.dis(main)
# print('-------- foo ------------')
# dis.dis(foo)


if __name__ == '__main__':
    main()

    print('TEST SUCEEDED')
