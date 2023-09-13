"""Tests for async helper functions"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import asyncio

from jupyter_core.utils import ensure_async, run_sync


async def afunc():
    return "afunc"


def func():
    return "func"


sync_afunc = run_sync(afunc)


def test_ensure_async():
    async def main():
        assert await ensure_async(afunc()) == "afunc"
        assert await ensure_async(func()) == "func"

    asyncio.run(main())


def test_run_sync():
    async def main():
        assert sync_afunc() == "afunc"

    asyncio.run(main())
