"""
Small utilities for testing.
"""
import os
import gc
import sys

from joblib._multiprocessing_helpers import mp
from joblib.testing import SkipTest, skipif

try:
    import lz4
except ImportError:
    lz4 = None

IS_PYPY = hasattr(sys, "pypy_version_info")

# A decorator to run tests only when numpy is available
try:
    import numpy as np

    def with_numpy(func):
        """A decorator to skip tests requiring numpy."""
        return func

except ImportError:
    def with_numpy(func):
        """A decorator to skip tests requiring numpy."""
        def my_func():
            raise SkipTest('Test requires numpy')
        return my_func
    np = None

# TODO: Turn this back on after refactoring yield based tests in test_hashing
# with_numpy = skipif(not np, reason='Test requires numpy.')

# we use memory_profiler library for memory consumption checks
try:
    from memory_profiler import memory_usage

    def with_memory_profiler(func):
        """A decorator to skip tests requiring memory_profiler."""
        return func

    def memory_used(func, *args, **kwargs):
        """Compute memory usage when executing func."""
        gc.collect()
        mem_use = memory_usage((func, args, kwargs), interval=.001)
        return max(mem_use) - min(mem_use)

except ImportError:
    def with_memory_profiler(func):
        """A decorator to skip tests requiring memory_profiler."""
        def dummy_func():
            raise SkipTest('Test requires memory_profiler.')
        return dummy_func

    memory_usage = memory_used = None


def force_gc_pypy():
    # The gc in pypy can be delayed. Force it to test the behavior when it
    # will eventually be collected.
    if IS_PYPY:
        # Run gc.collect() twice to make sure the weakref is collected, as
        # mentionned in the pypy doc:
        # https://doc.pypy.org/en/latest/config/objspace.usemodules._weakref.html
        import gc
        gc.collect()
        gc.collect()


with_multiprocessing = skipif(
    mp is None, reason='Needs multiprocessing to run.')


with_dev_shm = skipif(
    not os.path.exists('/dev/shm'),
    reason='This test requires a large /dev/shm shared memory fs.')

with_lz4 = skipif(lz4 is None, reason='Needs lz4 compression to run')

without_lz4 = skipif(
    lz4 is not None, reason='Needs lz4 not being installed to run')
