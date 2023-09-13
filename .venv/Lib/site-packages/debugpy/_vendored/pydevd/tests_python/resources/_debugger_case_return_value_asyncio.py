import asyncio


async def method1():
    return 1


async def method2():
    return 2


async def main():
    await method1()  # break here
    await method2()
    print('TEST SUCEEDED!')


if __name__ == '__main__':
    if hasattr(asyncio, 'run'):
        print('using asyncio.run')
        asyncio.run(main())
    else:
        print('using loop.run_until_complete')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        loop.close()
