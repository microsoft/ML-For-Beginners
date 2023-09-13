import asyncio


async def main():
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from not_my_code import not_my_coroutine
    try:
        await not_my_coroutine.call1(call2)  # stop here 2
    finally:
        print('finish main')


async def call2():
    try:
        raise RuntimeError()  # stop here 1
    except AssertionError:
        pass


if __name__ == '__main__':
    try:
        if hasattr(asyncio, 'run'):
            print('using asyncio.run')
            asyncio.run(main())  # stop here 3a
        else:
            print('using loop.run_until_complete')
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())  # stop here 3b
            loop.close()
    finally:
        print('TEST SUCEEDED!')
