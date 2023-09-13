"""
Inputhook for running the original asyncio event loop while we're waiting for
input.

By default, in IPython, we run the prompt with a different asyncio event loop,
because otherwise we risk that people are freezing the prompt by scheduling bad
coroutines. E.g., a coroutine that does a while/true and never yield back
control to the loop. We can't cancel that.

However, sometimes we want the asyncio loop to keep running while waiting for
a prompt.

The following example will print the numbers from 1 to 10 above the prompt,
while we are waiting for input. (This works also because we use
prompt_toolkit`s `patch_stdout`)::

    In [1]: import asyncio

    In [2]: %gui asyncio

    In [3]: async def f():
       ...:     for i in range(10):
       ...:         await asyncio.sleep(1)
       ...:         print(i)


    In [4]: asyncio.ensure_future(f())

"""
from prompt_toolkit import __version__ as ptk_version

from IPython.core.async_helpers import get_asyncio_loop

PTK3 = ptk_version.startswith("3.")


def inputhook(context):
    """
    Inputhook for asyncio event loop integration.
    """
    # For prompt_toolkit 3.0, this input hook literally doesn't do anything.
    # The event loop integration here is implemented in `interactiveshell.py`
    # by running the prompt itself in the current asyncio loop. The main reason
    # for this is that nesting asyncio event loops is unreliable.
    if PTK3:
        return

    # For prompt_toolkit 2.0, we can run the current asyncio event loop,
    # because prompt_toolkit 2.0 uses a different event loop internally.

    # get the persistent asyncio event loop
    loop = get_asyncio_loop()

    def stop():
        loop.stop()

    fileno = context.fileno()
    loop.add_reader(fileno, stop)
    try:
        loop.run_forever()
    finally:
        loop.remove_reader(fileno)
