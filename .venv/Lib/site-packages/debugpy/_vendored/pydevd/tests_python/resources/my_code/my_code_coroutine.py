import asyncio


async def main():
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from not_my_code import not_my_coroutine
    await not_my_coroutine.call1(call2)
    print('finish main')


async def call2():
    print('on call2')  # break here


if __name__ == '__main__':
    if hasattr(asyncio, 'run'):
        print('using asyncio.run')
        asyncio.run(main())
    else:
        print('using loop.run_until_complete')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        loop.close()
    print('TEST SUCEEDED!')
