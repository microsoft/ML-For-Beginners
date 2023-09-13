import trio


async def count():
    print('enter count')
    await trio.sleep(.001)  # break count 1
    await trio.sleep(.001)  # break count 2


async def count2():
    print('enter count 2')
    await trio.sleep(.001)
    await trio.sleep(.001)


async def count3():
    print('enter count 3')
    await trio.sleep(.001)
    await trio.sleep(.001)


async def main():
    await count()  # break main
    await count2()  # step main
    await count3()


if __name__ == '__main__':
    trio.run(main)
    print('TEST SUCEEDED!')
