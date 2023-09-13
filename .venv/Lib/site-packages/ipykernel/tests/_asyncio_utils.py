"""test utilities that use async/await syntax

a separate file to avoid syntax errors on Python 2
"""

import asyncio


def async_func():
    """Simple async function to schedule a task on the current eventloop"""
    loop = asyncio.get_event_loop()
    assert loop.is_running()

    async def task():
        await asyncio.sleep(1)

    loop.create_task(task())
