import asyncio

async def gen():
    for i in range(10):
        yield i

async def run():
    async for p in gen():
        print(p)

if __name__ == "__main__":
    loop = asyncio.get_event_loop_policy().get_event_loop()
    loop.run_until_complete(run())
    print('TEST SUCEEDED')