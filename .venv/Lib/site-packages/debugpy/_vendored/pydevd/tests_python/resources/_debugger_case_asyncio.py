import asyncio


async def count():
    print('enter count')
    await asyncio.sleep(.001)  # break count 1
    await asyncio.sleep(.001)  # break count 2


async def count2():
    print('enter count 2')
    await asyncio.sleep(.001)
    await asyncio.sleep(.001)


async def count3():
    print('enter count 3')
    await asyncio.sleep(.001)
    await asyncio.sleep(.001)


async def main():
    await count()  # break main
    await count2()  # step main
    await count3()


if __name__ == "__main__":
    if hasattr(asyncio, 'run'):
        print('using asyncio.run')
        asyncio.run(main())
    else:
        print('using loop.run_until_complete')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        loop.close()
    print('TEST SUCEEDED!')
