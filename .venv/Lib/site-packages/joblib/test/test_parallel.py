"""
Test the parallel module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2010-2011 Gael Varoquaux
# License: BSD Style, 3 clauses.

import os
import sys
import time
import mmap
import weakref
import warnings
import threading
from traceback import format_exception
from math import sqrt
from time import sleep
from pickle import PicklingError
from contextlib import nullcontext
from multiprocessing import TimeoutError
import pytest

import joblib
from joblib import parallel
from joblib import dump, load

from joblib._multiprocessing_helpers import mp

from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.test.common import IS_PYPY, force_gc_pypy
from joblib.testing import (parametrize, raises, check_subprocess_call,
                            skipif, warns)

if mp is not None:
    # Loky is not available if multiprocessing is not
    from joblib.externals.loky import get_reusable_executor

from queue import Queue

try:
    import posix
except ImportError:
    posix = None

try:
    from ._openmp_test_helper.parallel_sum import parallel_sum
except ImportError:
    parallel_sum = None

try:
    import distributed
except ImportError:
    distributed = None

from joblib._parallel_backends import SequentialBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib._parallel_backends import ParallelBackendBase
from joblib._parallel_backends import LokyBackend

from joblib.parallel import Parallel, delayed
from joblib.parallel import parallel_config
from joblib.parallel import parallel_backend
from joblib.parallel import register_parallel_backend
from joblib.parallel import effective_n_jobs, cpu_count

from joblib.parallel import mp, BACKENDS, DEFAULT_BACKEND


RETURN_GENERATOR_BACKENDS = BACKENDS.copy()
RETURN_GENERATOR_BACKENDS.pop("multiprocessing", None)

ALL_VALID_BACKENDS = [None] + sorted(BACKENDS.keys())
# Add instances of backend classes deriving from ParallelBackendBase
ALL_VALID_BACKENDS += [BACKENDS[backend_str]() for backend_str in BACKENDS]
if mp is None:
    PROCESS_BACKENDS = []
else:
    PROCESS_BACKENDS = ['multiprocessing', 'loky']
PARALLEL_BACKENDS = PROCESS_BACKENDS + ['threading']

if hasattr(mp, 'get_context'):
    # Custom multiprocessing context in Python 3.4+
    ALL_VALID_BACKENDS.append(mp.get_context('spawn'))

DefaultBackend = BACKENDS[DEFAULT_BACKEND]


def get_workers(backend):
    return getattr(backend, '_pool', getattr(backend, '_workers', None))


def division(x, y):
    return x / y


def square(x):
    return x ** 2


class MyExceptionWithFinickyInit(Exception):
    """An exception class with non trivial __init__
    """
    def __init__(self, a, b, c, d):
        pass


def exception_raiser(x, custom_exception=False):
    if x == 7:
        raise (MyExceptionWithFinickyInit('a', 'b', 'c', 'd')
               if custom_exception else ValueError)
    return x


def interrupt_raiser(x):
    time.sleep(.05)
    raise KeyboardInterrupt


def f(x, y=0, z=0):
    """ A module-level function so that it can be spawn with
    multiprocessing.
    """
    return x ** 2 + y + z


def _active_backend_type():
    return type(parallel.get_active_backend()[0])


def parallel_func(inner_n_jobs, backend):
    return Parallel(n_jobs=inner_n_jobs, backend=backend)(
        delayed(square)(i) for i in range(3))


###############################################################################
def test_cpu_count():
    assert cpu_count() > 0


def test_effective_n_jobs():
    assert effective_n_jobs() > 0


@parametrize("context", [parallel_config, parallel_backend])
@pytest.mark.parametrize(
    "backend_n_jobs, expected_n_jobs",
    [(3, 3), (-1, effective_n_jobs(n_jobs=-1)), (None, 1)],
    ids=["positive-int", "negative-int", "None"]
)
@with_multiprocessing
def test_effective_n_jobs_None(context, backend_n_jobs, expected_n_jobs):
    # check the number of effective jobs when `n_jobs=None`
    # non-regression test for https://github.com/joblib/joblib/issues/984
    with context("threading", n_jobs=backend_n_jobs):
        # when using a backend, the default of number jobs will be the one set
        # in the backend
        assert effective_n_jobs(n_jobs=None) == expected_n_jobs
    # without any backend, None will default to a single job
    assert effective_n_jobs(n_jobs=None) == 1


###############################################################################
# Test parallel

@parametrize('backend', ALL_VALID_BACKENDS)
@parametrize('n_jobs', [1, 2, -1, -2])
@parametrize('verbose', [2, 11, 100])
def test_simple_parallel(backend, n_jobs, verbose):
    assert ([square(x) for x in range(5)] ==
            Parallel(n_jobs=n_jobs, backend=backend,
                     verbose=verbose)(
                delayed(square)(x) for x in range(5)))


@parametrize('backend', ALL_VALID_BACKENDS)
def test_main_thread_renamed_no_warning(backend, monkeypatch):
    # Check that no default backend relies on the name of the main thread:
    # https://github.com/joblib/joblib/issues/180#issuecomment-253266247
    # Some programs use a different name for the main thread. This is the case
    # for uWSGI apps for instance.
    monkeypatch.setattr(target=threading.current_thread(), name='name',
                        value='some_new_name_for_the_main_thread')

    with warnings.catch_warnings(record=True) as warninfo:
        results = Parallel(n_jobs=2, backend=backend)(
            delayed(square)(x) for x in range(3))
        assert results == [0, 1, 4]

    # Due to the default parameters of LokyBackend, there is a chance that
    # warninfo catches Warnings from worker timeouts. We remove it if it exists
    warninfo = [w for w in warninfo if "worker timeout" not in str(w.message)]

    # The multiprocessing backend will raise a warning when detecting that is
    # started from the non-main thread. Let's check that there is no false
    # positive because of the name change.
    assert len(warninfo) == 0


def _assert_warning_nested(backend, inner_n_jobs, expected):
    with warnings.catch_warnings(record=True) as warninfo:
        warnings.simplefilter("always")
        parallel_func(backend=backend, inner_n_jobs=inner_n_jobs)

    warninfo = [w.message for w in warninfo]
    if expected:
        if warninfo:
            warnings_are_correct = all(
                'backed parallel loops cannot' in each.args[0]
                for each in warninfo
            )
            # With Python nogil, when the outer backend is threading, we might
            # see more that one warning
            warnings_have_the_right_length = (
                len(warninfo) >= 1 if getattr(sys.flags, 'nogil', False)
                else len(warninfo) == 1)
            return warnings_are_correct and warnings_have_the_right_length

        return False
    else:
        assert not warninfo
        return True


@with_multiprocessing
@parametrize('parent_backend,child_backend,expected', [
    ('loky', 'multiprocessing', True),
    ('loky', 'loky', False),
    ('multiprocessing', 'multiprocessing', True),
    ('multiprocessing', 'loky', True),
    ('threading', 'multiprocessing', True),
    ('threading', 'loky', True),
])
def test_nested_parallel_warnings(parent_backend, child_backend, expected):

    # no warnings if inner_n_jobs=1
    Parallel(n_jobs=2, backend=parent_backend)(
        delayed(_assert_warning_nested)(
            backend=child_backend, inner_n_jobs=1,
            expected=False)
        for _ in range(5))

    #  warnings if inner_n_jobs != 1 and expected
    res = Parallel(n_jobs=2, backend=parent_backend)(
        delayed(_assert_warning_nested)(
            backend=child_backend, inner_n_jobs=2,
            expected=expected)
        for _ in range(5))

    # warning handling is not thread safe. One thread might see multiple
    # warning or no warning at all.
    if parent_backend == "threading":
        if IS_PYPY and not any(res):
            # Related to joblib#1426, should be removed once it is solved.
            pytest.xfail(reason="This test often fails in PyPy.")
        assert any(res)
    else:
        assert all(res)


@with_multiprocessing
@parametrize('backend', ['loky', 'multiprocessing', 'threading'])
def test_background_thread_parallelism(backend):
    is_run_parallel = [False]

    def background_thread(is_run_parallel):
        with warnings.catch_warnings(record=True) as warninfo:
            Parallel(n_jobs=2)(
                delayed(sleep)(.1) for _ in range(4))
        print(len(warninfo))
        is_run_parallel[0] = len(warninfo) == 0

    t = threading.Thread(target=background_thread, args=(is_run_parallel,))
    t.start()
    t.join()
    assert is_run_parallel[0]


def nested_loop(backend):
    Parallel(n_jobs=2, backend=backend)(
        delayed(square)(.01) for _ in range(2))


@parametrize('child_backend', BACKENDS)
@parametrize('parent_backend', BACKENDS)
def test_nested_loop(parent_backend, child_backend):
    Parallel(n_jobs=2, backend=parent_backend)(
        delayed(nested_loop)(child_backend) for _ in range(2))


def raise_exception(backend):
    raise ValueError


@with_multiprocessing
def test_nested_loop_with_exception_with_loky():
    with raises(ValueError):
        with Parallel(n_jobs=2, backend="loky") as parallel:
            parallel([delayed(nested_loop)("loky"),
                      delayed(raise_exception)("loky")])


def test_mutate_input_with_threads():
    """Input is mutable when using the threading backend"""
    q = Queue(maxsize=5)
    Parallel(n_jobs=2, backend="threading")(
        delayed(q.put)(1) for _ in range(5))
    assert q.full()


@parametrize('n_jobs', [1, 2, 3])
def test_parallel_kwargs(n_jobs):
    """Check the keyword argument processing of pmap."""
    lst = range(10)
    assert ([f(x, y=1) for x in lst] ==
            Parallel(n_jobs=n_jobs)(delayed(f)(x, y=1) for x in lst))


@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_as_context_manager(backend):
    lst = range(10)
    expected = [f(x, y=1) for x in lst]

    with Parallel(n_jobs=4, backend=backend) as p:
        # Internally a pool instance has been eagerly created and is managed
        # via the context manager protocol
        managed_backend = p._backend

        # We make call with the managed parallel object several times inside
        # the managed block:
        assert expected == p(delayed(f)(x, y=1) for x in lst)
        assert expected == p(delayed(f)(x, y=1) for x in lst)

        # Those calls have all used the same pool instance:
        if mp is not None:
            assert get_workers(managed_backend) is get_workers(p._backend)

    # As soon as we exit the context manager block, the pool is terminated and
    # no longer referenced from the parallel object:
    if mp is not None:
        assert get_workers(p._backend) is None

    # It's still possible to use the parallel instance in non-managed mode:
    assert expected == p(delayed(f)(x, y=1) for x in lst)
    if mp is not None:
        assert get_workers(p._backend) is None


@with_multiprocessing
def test_parallel_pickling():
    """ Check that pmap captures the errors when it is passed an object
        that cannot be pickled.
    """
    class UnpicklableObject(object):
        def __reduce__(self):
            raise RuntimeError('123')

    with raises(PicklingError, match=r"the task to send"):
        Parallel(n_jobs=2, backend='loky')(delayed(id)(
            UnpicklableObject()) for _ in range(10))


@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_timeout_success(backend):
    # Check that timeout isn't thrown when function is fast enough
    assert len(Parallel(n_jobs=2, backend=backend, timeout=30)(
        delayed(sleep)(0.001) for x in range(10))) == 10


@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_timeout_fail(backend):
    # Check that timeout properly fails when function is too slow
    with raises(TimeoutError):
        Parallel(n_jobs=2, backend=backend, timeout=0.01)(
            delayed(sleep)(10) for x in range(10))


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_error_capture(backend):
    # Check that error are captured, and that correct exceptions
    # are raised.
    if mp is not None:
        with raises(ZeroDivisionError):
            Parallel(n_jobs=2, backend=backend)(
                [delayed(division)(x, y)
                    for x, y in zip((0, 1), (1, 0))])

        with raises(KeyboardInterrupt):
            Parallel(n_jobs=2, backend=backend)(
                [delayed(interrupt_raiser)(x) for x in (1, 0)])

        # Try again with the context manager API
        with Parallel(n_jobs=2, backend=backend) as parallel:
            assert get_workers(parallel._backend) is not None
            original_workers = get_workers(parallel._backend)

            with raises(ZeroDivisionError):
                parallel([delayed(division)(x, y)
                          for x, y in zip((0, 1), (1, 0))])

            # The managed pool should still be available and be in a working
            # state despite the previously raised (and caught) exception
            assert get_workers(parallel._backend) is not None

            # The pool should have been interrupted and restarted:
            assert get_workers(parallel._backend) is not original_workers

            assert ([f(x, y=1) for x in range(10)] ==
                    parallel(delayed(f)(x, y=1) for x in range(10)))

            original_workers = get_workers(parallel._backend)
            with raises(KeyboardInterrupt):
                parallel([delayed(interrupt_raiser)(x) for x in (1, 0)])

            # The pool should still be available despite the exception
            assert get_workers(parallel._backend) is not None

            # The pool should have been interrupted and restarted:
            assert get_workers(parallel._backend) is not original_workers

            assert ([f(x, y=1) for x in range(10)] ==
                    parallel(delayed(f)(x, y=1) for x in range(10))), (
                parallel._iterating, parallel.n_completed_tasks,
                parallel.n_dispatched_tasks, parallel._aborting
            )

        # Check that the inner pool has been terminated when exiting the
        # context manager
        assert get_workers(parallel._backend) is None
    else:
        with raises(KeyboardInterrupt):
            Parallel(n_jobs=2)(
                [delayed(interrupt_raiser)(x) for x in (1, 0)])

    # wrapped exceptions should inherit from the class of the original
    # exception to make it easy to catch them
    with raises(ZeroDivisionError):
        Parallel(n_jobs=2)(
            [delayed(division)(x, y) for x, y in zip((0, 1), (1, 0))])

    with raises(MyExceptionWithFinickyInit):
        Parallel(n_jobs=2, verbose=0)(
            (delayed(exception_raiser)(i, custom_exception=True)
             for i in range(30)))


def consumer(queue, item):
    queue.append('Consumed %s' % item)


@parametrize('backend', BACKENDS)
@parametrize('batch_size, expected_queue',
             [(1, ['Produced 0', 'Consumed 0',
                   'Produced 1', 'Consumed 1',
                   'Produced 2', 'Consumed 2',
                   'Produced 3', 'Consumed 3',
                   'Produced 4', 'Consumed 4',
                   'Produced 5', 'Consumed 5']),
              (4, [  # First Batch
                  'Produced 0', 'Produced 1', 'Produced 2', 'Produced 3',
                  'Consumed 0', 'Consumed 1', 'Consumed 2', 'Consumed 3',
                     # Second batch
                  'Produced 4', 'Produced 5', 'Consumed 4', 'Consumed 5'])])
def test_dispatch_one_job(backend, batch_size, expected_queue):
    """ Test that with only one job, Parallel does act as a iterator.
    """
    queue = list()

    def producer():
        for i in range(6):
            queue.append('Produced %i' % i)
            yield i

    Parallel(n_jobs=1, batch_size=batch_size, backend=backend)(
        delayed(consumer)(queue, x) for x in producer())
    assert queue == expected_queue
    assert len(queue) == 12


@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
def test_dispatch_multiprocessing(backend):
    """ Check that using pre_dispatch Parallel does indeed dispatch items
        lazily.
    """
    manager = mp.Manager()
    queue = manager.list()

    def producer():
        for i in range(6):
            queue.append('Produced %i' % i)
            yield i

    Parallel(n_jobs=2, batch_size=1, pre_dispatch=3, backend=backend)(
        delayed(consumer)(queue, 'any') for _ in producer())

    queue_contents = list(queue)
    assert queue_contents[0] == 'Produced 0'

    # Only 3 tasks are pre-dispatched out of 6. The 4th task is dispatched only
    # after any of the first 3 jobs have completed.
    first_consumption_index = queue_contents[:4].index('Consumed any')
    assert first_consumption_index > -1

    produced_3_index = queue_contents.index('Produced 3')  # 4th task produced
    assert produced_3_index > first_consumption_index

    assert len(queue) == 12


def test_batching_auto_threading():
    # batching='auto' with the threading backend leaves the effective batch
    # size to 1 (no batching) as it has been found to never be beneficial with
    # this low-overhead backend.

    with Parallel(n_jobs=2, batch_size='auto', backend='threading') as p:
        p(delayed(id)(i) for i in range(5000))  # many very fast tasks
        assert p._backend.compute_batch_size() == 1


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_batching_auto_subprocesses(backend):
    with Parallel(n_jobs=2, batch_size='auto', backend=backend) as p:
        p(delayed(id)(i) for i in range(5000))  # many very fast tasks

        # It should be strictly larger than 1 but as we don't want heisen
        # failures on clogged CI worker environment be safe and only check that
        # it's a strictly positive number.
        assert p._backend.compute_batch_size() > 0


def test_exception_dispatch():
    """Make sure that exception raised during dispatch are indeed captured"""
    with raises(ValueError):
        Parallel(n_jobs=2, pre_dispatch=16, verbose=0)(
            delayed(exception_raiser)(i) for i in range(30))


def nested_function_inner(i):
    Parallel(n_jobs=2)(
        delayed(exception_raiser)(j) for j in range(30))


def nested_function_outer(i):
    Parallel(n_jobs=2)(
        delayed(nested_function_inner)(j) for j in range(30))


@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
@pytest.mark.xfail(reason="https://github.com/joblib/loky/pull/255")
def test_nested_exception_dispatch(backend):
    """Ensure errors for nested joblib cases gets propagated

    We rely on the Python 3 built-in __cause__ system that already
    report this kind of information to the user.
    """
    with raises(ValueError) as excinfo:
        Parallel(n_jobs=2, backend=backend)(
            delayed(nested_function_outer)(i) for i in range(30))

    # Check that important information such as function names are visible
    # in the final error message reported to the user
    report_lines = format_exception(excinfo.type, excinfo.value, excinfo.tb)
    report = "".join(report_lines)
    assert 'nested_function_outer' in report
    assert 'nested_function_inner' in report
    assert 'exception_raiser' in report

    assert type(excinfo.value) is ValueError


class FakeParallelBackend(SequentialBackend):
    """Pretends to run concurrently while running sequentially."""

    def configure(self, n_jobs=1, parallel=None, **backend_args):
        self.n_jobs = self.effective_n_jobs(n_jobs)
        self.parallel = parallel
        return n_jobs

    def effective_n_jobs(self, n_jobs=1):
        if n_jobs < 0:
            n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
        return n_jobs


def test_invalid_backend():
    with raises(ValueError, match="Invalid backend:"):
        Parallel(backend='unit-testing')

    with raises(ValueError, match="Invalid backend:"):
        with parallel_config(backend='unit-testing'):
            pass

    with raises(ValueError, match="Invalid backend:"):
        with parallel_config(backend='unit-testing'):
            pass


@parametrize('backend', ALL_VALID_BACKENDS)
def test_invalid_njobs(backend):
    with raises(ValueError) as excinfo:
        Parallel(n_jobs=0, backend=backend)._initialize_backend()
    assert "n_jobs == 0 in Parallel has no meaning" in str(excinfo.value)


def test_register_parallel_backend():
    try:
        register_parallel_backend("test_backend", FakeParallelBackend)
        assert "test_backend" in BACKENDS
        assert BACKENDS["test_backend"] == FakeParallelBackend
    finally:
        del BACKENDS["test_backend"]


def test_overwrite_default_backend():
    assert _active_backend_type() == DefaultBackend
    try:
        register_parallel_backend("threading", BACKENDS["threading"],
                                  make_default=True)
        assert _active_backend_type() == ThreadingBackend
    finally:
        # Restore the global default manually
        parallel.DEFAULT_BACKEND = DEFAULT_BACKEND
    assert _active_backend_type() == DefaultBackend


@skipif(mp is not None, reason="Only without multiprocessing")
def test_backend_no_multiprocessing():
    with warns(UserWarning,
               match="joblib backend '.*' is not available on.*"):
        Parallel(backend='loky')(delayed(square)(i) for i in range(3))

    # The below should now work without problems
    with parallel_config(backend='loky'):
        Parallel()(delayed(square)(i) for i in range(3))


def check_backend_context_manager(context, backend_name):
    with context(backend_name, n_jobs=3):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert active_n_jobs == 3
        assert effective_n_jobs(3) == 3
        p = Parallel()
        assert p.n_jobs == 3
        if backend_name == 'multiprocessing':
            assert type(active_backend) is MultiprocessingBackend
            assert type(p._backend) is MultiprocessingBackend
        elif backend_name == 'loky':
            assert type(active_backend) is LokyBackend
            assert type(p._backend) is LokyBackend
        elif backend_name == 'threading':
            assert type(active_backend) is ThreadingBackend
            assert type(p._backend) is ThreadingBackend
        elif backend_name.startswith('test_'):
            assert type(active_backend) is FakeParallelBackend
            assert type(p._backend) is FakeParallelBackend


all_backends_for_context_manager = PARALLEL_BACKENDS[:]
all_backends_for_context_manager.extend(
    ['test_backend_%d' % i for i in range(3)]
)


@with_multiprocessing
@parametrize('backend', all_backends_for_context_manager)
@parametrize('context', [parallel_backend, parallel_config])
def test_backend_context_manager(monkeypatch, backend, context):
    if backend not in BACKENDS:
        monkeypatch.setitem(BACKENDS, backend, FakeParallelBackend)

    assert _active_backend_type() == DefaultBackend
    # check that this possible to switch parallel backends sequentially
    check_backend_context_manager(context, backend)

    # The default backend is restored
    assert _active_backend_type() == DefaultBackend

    # Check that context manager switching is thread safe:
    Parallel(n_jobs=2, backend='threading')(
        delayed(check_backend_context_manager)(context, b)
        for b in all_backends_for_context_manager if not b)

    # The default backend is again restored
    assert _active_backend_type() == DefaultBackend


class ParameterizedParallelBackend(SequentialBackend):
    """Pretends to run conncurrently while running sequentially."""

    def __init__(self, param=None):
        if param is None:
            raise ValueError('param should not be None')
        self.param = param


@parametrize("context", [parallel_config, parallel_backend])
def test_parameterized_backend_context_manager(monkeypatch, context):
    monkeypatch.setitem(BACKENDS, 'param_backend',
                        ParameterizedParallelBackend)
    assert _active_backend_type() == DefaultBackend

    with context('param_backend', param=42, n_jobs=3):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert type(active_backend) is ParameterizedParallelBackend
        assert active_backend.param == 42
        assert active_n_jobs == 3
        p = Parallel()
        assert p.n_jobs == 3
        assert p._backend is active_backend
        results = p(delayed(sqrt)(i) for i in range(5))
    assert results == [sqrt(i) for i in range(5)]

    # The default backend is again restored
    assert _active_backend_type() == DefaultBackend


@parametrize("context", [parallel_config, parallel_backend])
def test_directly_parameterized_backend_context_manager(context):
    assert _active_backend_type() == DefaultBackend

    # Check that it's possible to pass a backend instance directly,
    # without registration
    with context(ParameterizedParallelBackend(param=43), n_jobs=5):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert type(active_backend) is ParameterizedParallelBackend
        assert active_backend.param == 43
        assert active_n_jobs == 5
        p = Parallel()
        assert p.n_jobs == 5
        assert p._backend is active_backend
        results = p(delayed(sqrt)(i) for i in range(5))
    assert results == [sqrt(i) for i in range(5)]

    # The default backend is again restored
    assert _active_backend_type() == DefaultBackend


def sleep_and_return_pid():
    sleep(.1)
    return os.getpid()


def get_nested_pids():
    assert _active_backend_type() == ThreadingBackend
    # Assert that the nested backend does not change the default number of
    # jobs used in Parallel
    assert Parallel()._effective_n_jobs() == 1

    # Assert that the tasks are running only on one process
    return Parallel(n_jobs=2)(delayed(sleep_and_return_pid)()
                              for _ in range(2))


class MyBackend(joblib._parallel_backends.LokyBackend):
    """Backend to test backward compatibility with older backends"""
    def get_nested_backend(self, ):
        # Older backends only return a backend, without n_jobs indications.
        return super(MyBackend, self).get_nested_backend()[0]


register_parallel_backend('back_compat_backend', MyBackend)


@with_multiprocessing
@parametrize('backend', ['threading', 'loky', 'multiprocessing',
                         'back_compat_backend'])
@parametrize("context", [parallel_config, parallel_backend])
def test_nested_backend_context_manager(context, backend):
    # Check that by default, nested parallel calls will always use the
    # ThreadingBackend

    with context(backend):
        pid_groups = Parallel(n_jobs=2)(
            delayed(get_nested_pids)()
            for _ in range(10)
        )
        for pid_group in pid_groups:
            assert len(set(pid_group)) == 1


@with_multiprocessing
@parametrize('n_jobs', [2, -1, None])
@parametrize('backend', PARALLEL_BACKENDS)
@parametrize("context", [parallel_config, parallel_backend])
def test_nested_backend_in_sequential(backend, n_jobs, context):
    # Check that by default, nested parallel calls will always use the
    # ThreadingBackend

    def check_nested_backend(expected_backend_type, expected_n_job):
        # Assert that the sequential backend at top level, does not change the
        # backend for nested calls.
        assert _active_backend_type() == BACKENDS[expected_backend_type]

        # Assert that the nested backend in SequentialBackend does not change
        # the default number of jobs used in Parallel
        expected_n_job = effective_n_jobs(expected_n_job)
        assert Parallel()._effective_n_jobs() == expected_n_job

    Parallel(n_jobs=1)(
        delayed(check_nested_backend)(DEFAULT_BACKEND, 1)
        for _ in range(10)
    )

    with context(backend, n_jobs=n_jobs):
        Parallel(n_jobs=1)(
            delayed(check_nested_backend)(backend, n_jobs)
            for _ in range(10)
        )


def check_nesting_level(context, inner_backend, expected_level):
    with context(inner_backend) as ctx:
        if context is parallel_config:
            backend = ctx["backend"]
        if context is parallel_backend:
            backend = ctx[0]
        assert backend.nesting_level == expected_level


@with_multiprocessing
@parametrize('outer_backend', PARALLEL_BACKENDS)
@parametrize('inner_backend', PARALLEL_BACKENDS)
@parametrize("context", [parallel_config, parallel_backend])
def test_backend_nesting_level(context, outer_backend, inner_backend):
    # Check that the nesting level for the backend is correctly set
    check_nesting_level(context, outer_backend, 0)

    Parallel(n_jobs=2, backend=outer_backend)(
        delayed(check_nesting_level)(context, inner_backend, 1)
        for _ in range(10)
    )

    with context(inner_backend, n_jobs=2):
        Parallel()(delayed(check_nesting_level)(context, inner_backend, 1)
                   for _ in range(10))


@with_multiprocessing
@parametrize("context", [parallel_config, parallel_backend])
@parametrize('with_retrieve_callback', [True, False])
def test_retrieval_context(context, with_retrieve_callback):
    import contextlib

    class MyBackend(ThreadingBackend):
        i = 0
        supports_retrieve_callback = with_retrieve_callback

        @contextlib.contextmanager
        def retrieval_context(self):
            self.i += 1
            yield

    register_parallel_backend("retrieval", MyBackend)

    def nested_call(n):
        return Parallel(n_jobs=2)(delayed(id)(i) for i in range(n))

    with context("retrieval") as ctx:
        Parallel(n_jobs=2)(
            delayed(nested_call)(i)
            for i in range(5)
        )
        if context is parallel_config:
            assert ctx["backend"].i == 1
        if context is parallel_backend:
            assert ctx[0].i == 1


###############################################################################
# Test helpers

@parametrize('batch_size', [0, -1, 1.42])
def test_invalid_batch_size(batch_size):
    with raises(ValueError):
        Parallel(batch_size=batch_size)


@parametrize('n_tasks, n_jobs, pre_dispatch, batch_size',
             [(2, 2, 'all', 'auto'),
              (2, 2, 'n_jobs', 'auto'),
              (10, 2, 'n_jobs', 'auto'),
              (517, 2, 'n_jobs', 'auto'),
              (10, 2, 'n_jobs', 'auto'),
              (10, 4, 'n_jobs', 'auto'),
              (200, 12, 'n_jobs', 'auto'),
              (25, 12, '2 * n_jobs', 1),
              (250, 12, 'all', 1),
              (250, 12, '2 * n_jobs', 7),
              (200, 12, '2 * n_jobs', 'auto')])
def test_dispatch_race_condition(n_tasks, n_jobs, pre_dispatch, batch_size):
    # Check that using (async-)dispatch does not yield a race condition on the
    # iterable generator that is not thread-safe natively.
    # This is a non-regression test for the "Pool seems closed" class of error
    params = {'n_jobs': n_jobs, 'pre_dispatch': pre_dispatch,
              'batch_size': batch_size}
    expected = [square(i) for i in range(n_tasks)]
    results = Parallel(**params)(delayed(square)(i) for i in range(n_tasks))
    assert results == expected


@with_multiprocessing
def test_default_mp_context():
    mp_start_method = mp.get_start_method()
    p = Parallel(n_jobs=2, backend='multiprocessing')
    context = p._backend_args.get('context')
    start_method = context.get_start_method()
    assert start_method == mp_start_method


@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_no_blas_crash_or_freeze_with_subprocesses(backend):
    if backend == 'multiprocessing':
        # Use the spawn backend that is both robust and available on all
        # platforms
        backend = mp.get_context('spawn')

    # Check that on recent Python version, the 'spawn' start method can make
    # it possible to use multiprocessing in conjunction of any BLAS
    # implementation that happens to be used by numpy with causing a freeze or
    # a crash
    rng = np.random.RandomState(42)

    # call BLAS DGEMM to force the initialization of the internal thread-pool
    # in the main process
    a = rng.randn(1000, 1000)
    np.dot(a, a.T)

    # check that the internal BLAS thread-pool is not in an inconsistent state
    # in the worker processes managed by multiprocessing
    Parallel(n_jobs=2, backend=backend)(
        delayed(np.dot)(a, a.T) for i in range(2))


UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_NO_MAIN = """\
from joblib import Parallel, delayed

def square(x):
    return x ** 2

backend = "{}"
if backend == "spawn":
    from multiprocessing import get_context
    backend = get_context(backend)

print(Parallel(n_jobs=2, backend=backend)(
      delayed(square)(i) for i in range(5)))
"""


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_parallel_with_interactively_defined_functions(backend):
    # When using the "-c" flag, interactive functions defined in __main__
    # should work with any backend.
    if backend == "multiprocessing" and mp.get_start_method() != "fork":
        pytest.skip("Require fork start method to use interactively defined "
                    "functions with multiprocessing.")
    code = UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_NO_MAIN.format(backend)
    check_subprocess_call(
        [sys.executable, '-c', code], timeout=10,
        stdout_regex=r'\[0, 1, 4, 9, 16\]')


UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_MAIN = """\
import sys
# Make sure that joblib is importable in the subprocess launching this
# script. This is needed in case we run the tests from the joblib root
# folder without having installed joblib
sys.path.insert(0, {joblib_root_folder!r})

from joblib import Parallel, delayed

def run(f, x):
    return f(x)

{define_func}

if __name__ == "__main__":
    backend = "{backend}"
    if backend == "spawn":
        from multiprocessing import get_context
        backend = get_context(backend)

    callable_position = "{callable_position}"
    if callable_position == "delayed":
        print(Parallel(n_jobs=2, backend=backend)(
                delayed(square)(i) for i in range(5)))
    elif callable_position == "args":
        print(Parallel(n_jobs=2, backend=backend)(
                delayed(run)(square, i) for i in range(5)))
    else:
        print(Parallel(n_jobs=2, backend=backend)(
                delayed(run)(f=square, x=i) for i in range(5)))
"""

SQUARE_MAIN = """\
def square(x):
    return x ** 2
"""
SQUARE_LOCAL = """\
def gen_square():
    def square(x):
        return x ** 2
    return square
square = gen_square()
"""
SQUARE_LAMBDA = """\
square = lambda x: x ** 2
"""


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS + ([] if mp is None else ['spawn']))
@parametrize('define_func', [SQUARE_MAIN, SQUARE_LOCAL, SQUARE_LAMBDA])
@parametrize('callable_position', ['delayed', 'args', 'kwargs'])
def test_parallel_with_unpicklable_functions_in_args(
        backend, define_func, callable_position, tmpdir):
    if backend in ['multiprocessing', 'spawn'] and (
            define_func != SQUARE_MAIN or sys.platform == "win32"):
        pytest.skip("Not picklable with pickle")
    code = UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_MAIN.format(
        define_func=define_func, backend=backend,
        callable_position=callable_position,
        joblib_root_folder=os.path.dirname(os.path.dirname(joblib.__file__)))
    code_file = tmpdir.join("unpicklable_func_script.py")
    code_file.write(code)
    check_subprocess_call(
        [sys.executable, code_file.strpath], timeout=10,
        stdout_regex=r'\[0, 1, 4, 9, 16\]')


INTERACTIVE_DEFINED_FUNCTION_AND_CLASS_SCRIPT_CONTENT = """\
import sys
import faulthandler
# Make sure that joblib is importable in the subprocess launching this
# script. This is needed in case we run the tests from the joblib root
# folder without having installed joblib
sys.path.insert(0, {joblib_root_folder!r})

from joblib import Parallel, delayed
from functools import partial

class MyClass:
    '''Class defined in the __main__ namespace'''
    def __init__(self, value):
        self.value = value


def square(x, ignored=None, ignored2=None):
    '''Function defined in the __main__ namespace'''
    return x.value ** 2


square2 = partial(square, ignored2='something')

# Here, we do not need the `if __name__ == "__main__":` safeguard when
# using the default `loky` backend (even on Windows).

# To make debugging easier
faulthandler.dump_traceback_later(30, exit=True)

# The following baroque function call is meant to check that joblib
# introspection rightfully uses cloudpickle instead of the (faster) pickle
# module of the standard library when necessary. In particular cloudpickle is
# necessary for functions and instances of classes interactively defined in the
# __main__ module.

print(Parallel(backend="loky", n_jobs=2)(
    delayed(square2)(MyClass(i), ignored=[dict(a=MyClass(1))])
    for i in range(5)
))
""".format(joblib_root_folder=os.path.dirname(
    os.path.dirname(joblib.__file__)))


@with_multiprocessing
def test_parallel_with_interactively_defined_functions_loky(tmpdir):
    # loky accepts interactive functions defined in __main__ and does not
    # require if __name__ == '__main__' even when the __main__ module is
    # defined by the result of the execution of a filesystem script.
    script = tmpdir.join('joblib_interactively_defined_function.py')
    script.write(INTERACTIVE_DEFINED_FUNCTION_AND_CLASS_SCRIPT_CONTENT)
    check_subprocess_call(
        [sys.executable, script.strpath],
        stdout_regex=r'\[0, 1, 4, 9, 16\]',
        timeout=None,  # rely on faulthandler to kill the process
    )


INTERACTIVELY_DEFINED_SUBCLASS_WITH_METHOD_SCRIPT_CONTENT = """\
import sys
# Make sure that joblib is importable in the subprocess launching this
# script. This is needed in case we run the tests from the joblib root
# folder without having installed joblib
sys.path.insert(0, {joblib_root_folder!r})

from joblib import Parallel, delayed, hash
import multiprocessing as mp
mp.util.log_to_stderr(5)

class MyList(list):
    '''MyList is interactively defined by MyList.append is a built-in'''
    def __hash__(self):
        # XXX: workaround limitation in cloudpickle
        return hash(self).__hash__()

l = MyList()

print(Parallel(backend="loky", n_jobs=2)(
    delayed(l.append)(i) for i in range(3)
))
""".format(joblib_root_folder=os.path.dirname(
    os.path.dirname(joblib.__file__)))


@with_multiprocessing
def test_parallel_with_interactively_defined_bound_method_loky(tmpdir):
    script = tmpdir.join('joblib_interactive_bound_method_script.py')
    script.write(INTERACTIVELY_DEFINED_SUBCLASS_WITH_METHOD_SCRIPT_CONTENT)
    check_subprocess_call([sys.executable, script.strpath],
                          stdout_regex=r'\[None, None, None\]',
                          stderr_regex=r'LokyProcess',
                          timeout=15)


def test_parallel_with_exhausted_iterator():
    exhausted_iterator = iter([])
    assert Parallel(n_jobs=2)(exhausted_iterator) == []


def _cleanup_worker():
    """Helper function to force gc in each worker."""
    force_gc_pypy()
    time.sleep(.1)


def check_memmap(a):
    if not isinstance(a, np.memmap):
        raise TypeError('Expected np.memmap instance, got %r',
                        type(a))
    return a.copy()  # return a regular array instead of a memmap


@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_auto_memmap_on_arrays_from_generator(backend):
    # Non-regression test for a problem with a bad interaction between the
    # GC collecting arrays recently created during iteration inside the
    # parallel dispatch loop and the auto-memmap feature of Parallel.
    # See: https://github.com/joblib/joblib/pull/294
    def generate_arrays(n):
        for i in range(n):
            yield np.ones(10, dtype=np.float32) * i
    # Use max_nbytes=1 to force the use of memory-mapping even for small
    # arrays
    results = Parallel(n_jobs=2, max_nbytes=1, backend=backend)(
        delayed(check_memmap)(a) for a in generate_arrays(100))
    for result, expected in zip(results, generate_arrays(len(results))):
        np.testing.assert_array_equal(expected, result)

    # Second call to force loky to adapt the executor by growing the number
    # of worker processes. This is a non-regression test for:
    # https://github.com/joblib/joblib/issues/629.
    results = Parallel(n_jobs=4, max_nbytes=1, backend=backend)(
        delayed(check_memmap)(a) for a in generate_arrays(100))
    for result, expected in zip(results, generate_arrays(len(results))):
        np.testing.assert_array_equal(expected, result)


def identity(arg):
    return arg


@with_numpy
@with_multiprocessing
def test_memmap_with_big_offset(tmpdir):
    fname = tmpdir.join('test.mmap').strpath
    size = mmap.ALLOCATIONGRANULARITY
    obj = [np.zeros(size, dtype='uint8'), np.ones(size, dtype='uint8')]
    dump(obj, fname)
    memmap = load(fname, mmap_mode='r')
    result, = Parallel(n_jobs=2)(delayed(identity)(memmap) for _ in [0])
    assert isinstance(memmap[1], np.memmap)
    assert memmap[1].offset > size
    np.testing.assert_array_equal(obj, result)


def test_warning_about_timeout_not_supported_by_backend():
    with warnings.catch_warnings(record=True) as warninfo:
        Parallel(n_jobs=1, timeout=1)(delayed(square)(i) for i in range(50))
    assert len(warninfo) == 1
    w = warninfo[0]
    assert isinstance(w.message, UserWarning)
    assert str(w.message) == (
        "The backend class 'SequentialBackend' does not support timeout. "
        "You have set 'timeout=1' in Parallel but the 'timeout' parameter "
        "will not be used.")


def set_list_value(input_list, index, value):
    input_list[index] = value
    return value


@pytest.mark.parametrize('n_jobs', [1, 2, 4])
def test_parallel_return_order_with_return_as_generator_parameter(n_jobs):
    # This test inserts values in a list in some expected order
    # in sequential computing, and then checks that this order has been
    # respected by Parallel output generator.
    input_list = [0] * 5
    result = Parallel(n_jobs=n_jobs, return_as="generator",
                      backend='threading')(
        delayed(set_list_value)(input_list, i, i) for i in range(5))

    # Ensure that all the tasks are completed before checking the result
    result = list(result)

    assert all(v == r for v, r in zip(input_list, result))


@parametrize('backend', ALL_VALID_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_abort_backend(n_jobs, backend):
    delays = ["a"] + [10] * 100
    with raises(TypeError):
        t_start = time.time()
        Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(time.sleep)(i) for i in delays)
    dt = time.time() - t_start
    assert dt < 20


def get_large_object(arg):
    result = np.ones(int(5 * 1e5), dtype=bool)
    result[0] = False
    return result


@with_numpy
@parametrize('backend', RETURN_GENERATOR_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_deadlock_with_generator(backend, n_jobs):
    # Non-regression test for a race condition in the backends when the pickler
    # is delayed by a large object.
    with Parallel(n_jobs=n_jobs, backend=backend,
                  return_as="generator") as parallel:
        result = parallel(delayed(get_large_object)(i) for i in range(10))
        next(result)
        next(result)
        del result
        # The gc in pypy can be delayed. Force it to make sure this test does
        # not cause timeout on the CI.
        force_gc_pypy()


@parametrize('backend', RETURN_GENERATOR_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_multiple_generator_call(backend, n_jobs):
    # Non-regression test that ensures the dispatch of the tasks starts
    # immediately when Parallel.__call__ is called. This test relies on the
    # assumption that only one generator can be submitted at a time.
    with raises(RuntimeError,
                match="This Parallel instance is already running"):
        parallel = Parallel(n_jobs, backend=backend, return_as="generator")
        g = parallel(delayed(sleep)(1) for _ in range(10))  # noqa: F841
        t_start = time.time()
        gen2 = parallel(delayed(id)(i) for i in range(100))  # noqa: F841

    # Make sure that the error is raised quickly
    assert time.time() - t_start < 2, (
        "The error should be raised immediatly when submitting a new task "
        "but it took more than 2s."
    )

    del g
    # The gc in pypy can be delayed. Force it to make sure this test does not
    # cause timeout on the CI.
    force_gc_pypy()


@parametrize('backend', RETURN_GENERATOR_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_multiple_generator_call_managed(backend, n_jobs):
    # Non-regression test that ensures the dispatch of the tasks starts
    # immediately when Parallel.__call__ is called. This test relies on the
    # assumption that only one generator can be submitted at a time.
    with Parallel(n_jobs, backend=backend,
                  return_as="generator") as parallel:
        g = parallel(delayed(sleep)(10) for _ in range(10))  # noqa: F841
        t_start = time.time()
        with raises(RuntimeError,
                    match="This Parallel instance is already running"):
            g2 = parallel(delayed(id)(i) for i in range(100))  # noqa: F841

        # Make sure that the error is raised quickly
        assert time.time() - t_start < 2, (
            "The error should be raised immediatly when submitting a new task "
            "but it took more than 2s."
        )

    # The gc in pypy can be delayed. Force it to make sure this test does not
    # cause timeout on the CI.
    del g
    force_gc_pypy()


@parametrize('backend', RETURN_GENERATOR_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_multiple_generator_call_separated(backend, n_jobs):
    # Check that for separated Parallel, both tasks are correctly returned.
    g = Parallel(n_jobs, backend=backend, return_as="generator")(
        delayed(sqrt)(i ** 2) for i in range(10)
    )
    g2 = Parallel(n_jobs, backend=backend, return_as="generator")(
        delayed(sqrt)(i ** 2) for i in range(10, 20)
    )

    assert all(res == i for res, i in zip(g, range(10)))
    assert all(res == i for res, i in zip(g2, range(10, 20)))


@parametrize('backend, error', [
    ('loky', True),
    ('threading', False),
    ('sequential', False),
])
def test_multiple_generator_call_separated_gc(backend, error):

    if (backend == 'loky') and (mp is None):
        pytest.skip("Requires multiprocessing")

    # Check that in loky, only one call can be run at a time with
    # a single executor.
    parallel = Parallel(2, backend=backend, return_as="generator")
    g = parallel(delayed(sleep)(10) for i in range(10))
    g_wr = weakref.finalize(g, lambda: print("Generator collected"))
    ctx = (
        raises(RuntimeError, match="The executor underlying Parallel")
        if error else nullcontext()
    )
    with ctx:
        # For loky, this call will raise an error as the gc of the previous
        # generator will shutdown the shared executor.
        # For the other backends, as the worker pools are not shared between
        # the two calls, this should proceed correctly.
        t_start = time.time()
        g = Parallel(2, backend=backend, return_as="generator")(
            delayed(sqrt)(i ** 2) for i in range(10, 20)
        )

        # The gc in pypy can be delayed. Force it to test the behavior when it
        # will eventually be collected.
        force_gc_pypy()
        assert all(res == i for res, i in zip(g, range(10, 20)))

    assert time.time() - t_start < 5

    # Make sure that the computation are stopped for the gc'ed generator
    retry = 0
    while g_wr.alive and retry < 3:
        retry += 1
        time.sleep(.5)
    assert time.time() - t_start < 5

    if parallel._effective_n_jobs() != 1:
        # check that the first parallel object is aborting (the final _aborted
        # state might be delayed).
        assert parallel._aborting


@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_memmapping_leaks(backend, tmpdir):
    # Non-regression test for memmapping backends. Ensure that the data
    # does not stay too long in memory
    tmpdir = tmpdir.strpath

    # Use max_nbytes=1 to force the use of memory-mapping even for small
    # arrays
    with Parallel(n_jobs=2, max_nbytes=1, backend=backend,
                  temp_folder=tmpdir) as p:
        p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)

        # The memmap folder should not be clean in the context scope
        assert len(os.listdir(tmpdir)) > 0

        # Cleaning of the memmap folder is triggered by the garbage
        # collection. With pypy the garbage collection has been observed to be
        # delayed, sometimes up until the shutdown of the interpreter. This
        # cleanup job executed in the worker ensures that it's triggered
        # immediately.
        p(delayed(_cleanup_worker)() for _ in range(2))

    # Make sure that the shared memory is cleaned at the end when we exit
    # the context
    for _ in range(100):
        if not os.listdir(tmpdir):
            break
        sleep(.1)
    else:
        raise AssertionError('temporary directory of Parallel was not removed')

    # Make sure that the shared memory is cleaned at the end of a call
    p = Parallel(n_jobs=2, max_nbytes=1, backend=backend)
    p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)
    p(delayed(_cleanup_worker)() for _ in range(2))

    for _ in range(100):
        if not os.listdir(tmpdir):
            break
        sleep(.1)
    else:
        raise AssertionError('temporary directory of Parallel was not removed')


@parametrize('backend',
             ([None, 'threading'] if mp is None
              else [None, 'loky', 'threading'])
             )
def test_lambda_expression(backend):
    # cloudpickle is used to pickle delayed callables
    results = Parallel(n_jobs=2, backend=backend)(
        delayed(lambda x: x ** 2)(i) for i in range(10))
    assert results == [i ** 2 for i in range(10)]


@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_backend_batch_statistics_reset(backend):
    """Test that a parallel backend correctly resets its batch statistics."""
    n_jobs = 2
    n_inputs = 500
    task_time = 2. / n_inputs

    p = Parallel(verbose=10, n_jobs=n_jobs, backend=backend)
    p(delayed(time.sleep)(task_time) for i in range(n_inputs))
    assert (p._backend._effective_batch_size ==
            p._backend._DEFAULT_EFFECTIVE_BATCH_SIZE)
    assert (p._backend._smoothed_batch_duration ==
            p._backend._DEFAULT_SMOOTHED_BATCH_DURATION)

    p(delayed(time.sleep)(task_time) for i in range(n_inputs))
    assert (p._backend._effective_batch_size ==
            p._backend._DEFAULT_EFFECTIVE_BATCH_SIZE)
    assert (p._backend._smoothed_batch_duration ==
            p._backend._DEFAULT_SMOOTHED_BATCH_DURATION)


@with_multiprocessing
@parametrize("context", [parallel_config, parallel_backend])
def test_backend_hinting_and_constraints(context):
    for n_jobs in [1, 2, -1]:
        assert type(Parallel(n_jobs=n_jobs)._backend) == DefaultBackend

        p = Parallel(n_jobs=n_jobs, prefer='threads')
        assert type(p._backend) is ThreadingBackend

        p = Parallel(n_jobs=n_jobs, prefer='processes')
        assert type(p._backend) is DefaultBackend

        p = Parallel(n_jobs=n_jobs, require='sharedmem')
        assert type(p._backend) is ThreadingBackend

    # Explicit backend selection can override backend hinting although it
    # is useless to pass a hint when selecting a backend.
    p = Parallel(n_jobs=2, backend='loky', prefer='threads')
    assert type(p._backend) is LokyBackend

    with context('loky', n_jobs=2):
        # Explicit backend selection by the user with the context manager
        # should be respected when combined with backend hints only.
        p = Parallel(prefer='threads')
        assert type(p._backend) is LokyBackend
        assert p.n_jobs == 2

    with context('loky', n_jobs=2):
        # Locally hard-coded n_jobs value is respected.
        p = Parallel(n_jobs=3, prefer='threads')
        assert type(p._backend) is LokyBackend
        assert p.n_jobs == 3

    with context('loky', n_jobs=2):
        # Explicit backend selection by the user with the context manager
        # should be ignored when the Parallel call has hard constraints.
        # In this case, the default backend that supports shared mem is
        # used an the default number of processes is used.
        p = Parallel(require='sharedmem')
        assert type(p._backend) is ThreadingBackend
        assert p.n_jobs == 1

    with context('loky', n_jobs=2):
        p = Parallel(n_jobs=3, require='sharedmem')
        assert type(p._backend) is ThreadingBackend
        assert p.n_jobs == 3


@parametrize("context", [parallel_config, parallel_backend])
def test_backend_hinting_and_constraints_with_custom_backends(
    capsys, context
):
    # Custom backends can declare that they use threads and have shared memory
    # semantics:
    class MyCustomThreadingBackend(ParallelBackendBase):
        supports_sharedmem = True
        use_threads = True

        def apply_async(self):
            pass

        def effective_n_jobs(self, n_jobs):
            return n_jobs

    with context(MyCustomThreadingBackend()):
        p = Parallel(n_jobs=2, prefer='processes')  # ignored
        assert type(p._backend) is MyCustomThreadingBackend

        p = Parallel(n_jobs=2, require='sharedmem')
        assert type(p._backend) is MyCustomThreadingBackend

    class MyCustomProcessingBackend(ParallelBackendBase):
        supports_sharedmem = False
        use_threads = False

        def apply_async(self):
            pass

        def effective_n_jobs(self, n_jobs):
            return n_jobs

    with context(MyCustomProcessingBackend()):
        p = Parallel(n_jobs=2, prefer='processes')
        assert type(p._backend) is MyCustomProcessingBackend

        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""

        p = Parallel(n_jobs=2, require='sharedmem', verbose=10)
        assert type(p._backend) is ThreadingBackend

        out, err = capsys.readouterr()
        expected = ("Using ThreadingBackend as joblib backend "
                    "instead of MyCustomProcessingBackend as the latter "
                    "does not provide shared memory semantics.")
        assert out.strip() == expected
        assert err == ""

    with raises(ValueError):
        Parallel(backend=MyCustomProcessingBackend(), require='sharedmem')


def test_invalid_backend_hinting_and_constraints():
    with raises(ValueError):
        Parallel(prefer='invalid')

    with raises(ValueError):
        Parallel(require='invalid')

    with raises(ValueError):
        # It is inconsistent to prefer process-based parallelism while
        # requiring shared memory semantics.
        Parallel(prefer='processes', require='sharedmem')

    if mp is not None:
        # It is inconsistent to ask explicitly for a process-based
        # parallelism while requiring shared memory semantics.
        with raises(ValueError):
            Parallel(backend='loky', require='sharedmem')
        with raises(ValueError):
            Parallel(backend='multiprocessing', require='sharedmem')


def _recursive_backend_info(limit=3, **kwargs):
    """Perform nested parallel calls and introspect the backend on the way"""

    with Parallel(n_jobs=2) as p:
        this_level = [(type(p._backend).__name__, p._backend.nesting_level)]
        if limit == 0:
            return this_level
        results = p(delayed(_recursive_backend_info)(limit=limit - 1, **kwargs)
                    for i in range(1))
        return this_level + results[0]


@with_multiprocessing
@parametrize('backend', ['loky', 'threading'])
@parametrize("context", [parallel_config, parallel_backend])
def test_nested_parallelism_limit(context, backend):
    with context(backend, n_jobs=2):
        backend_types_and_levels = _recursive_backend_info()

    if cpu_count() == 1:
        second_level_backend_type = 'SequentialBackend'
        max_level = 1
    else:
        second_level_backend_type = 'ThreadingBackend'
        max_level = 2

    top_level_backend_type = backend.title() + 'Backend'
    expected_types_and_levels = [
        (top_level_backend_type, 0),
        (second_level_backend_type, 1),
        ('SequentialBackend', max_level),
        ('SequentialBackend', max_level)
    ]
    assert backend_types_and_levels == expected_types_and_levels


@with_numpy
@skipif(distributed is None, reason='This test requires dask')
@parametrize("context", [parallel_config, parallel_backend])
def test_nested_parallelism_with_dask(context):
    client = distributed.Client(n_workers=2, threads_per_worker=2)  # noqa

    # 10 MB of data as argument to trigger implicit scattering
    data = np.ones(int(1e7), dtype=np.uint8)
    for i in range(2):
        with context('dask'):
            backend_types_and_levels = _recursive_backend_info(data=data)
        assert len(backend_types_and_levels) == 4
        assert all(name == 'DaskDistributedBackend'
                   for name, _ in backend_types_and_levels)

    # No argument
    with context('dask'):
        backend_types_and_levels = _recursive_backend_info()
    assert len(backend_types_and_levels) == 4
    assert all(name == 'DaskDistributedBackend'
               for name, _ in backend_types_and_levels)


def _recursive_parallel(nesting_limit=None):
    """A horrible function that does recursive parallel calls"""
    return Parallel()(delayed(_recursive_parallel)() for i in range(2))


@pytest.mark.no_cover
@parametrize("context", [parallel_config, parallel_backend])
@parametrize(
    'backend', (['threading'] if mp is None else ['loky', 'threading'])
)
def test_thread_bomb_mitigation(context, backend):
    # Test that recursive parallelism raises a recursion rather than
    # saturating the operating system resources by creating a unbounded number
    # of threads.
    with context(backend, n_jobs=2):
        with raises(BaseException) as excinfo:
            _recursive_parallel()
    exc = excinfo.value
    if backend == "loky":
        # Local import because loky may not be importable for lack of
        # multiprocessing
        from joblib.externals.loky.process_executor import TerminatedWorkerError # noqa
        if isinstance(exc, (TerminatedWorkerError, PicklingError)):
            # The recursion exception can itself cause an error when
            # pickling it to be send back to the parent process. In this
            # case the worker crashes but the original traceback is still
            # printed on stderr. This could be improved but does not seem
            # simple to do and this is not critical for users (as long
            # as there is no process or thread bomb happening).
            pytest.xfail("Loky worker crash when serializing RecursionError")

    assert isinstance(exc, RecursionError)


def _run_parallel_sum():
    env_vars = {}
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'NUMBA_NUM_THREADS', 'ENABLE_IPC']:
        env_vars[var] = os.environ.get(var)
    return env_vars, parallel_sum(100)


@parametrize("backend", ([None, 'loky'] if mp is not None else [None]))
@skipif(parallel_sum is None, reason="Need OpenMP helper compiled")
def test_parallel_thread_limit(backend):
    results = Parallel(n_jobs=2, backend=backend)(
        delayed(_run_parallel_sum)() for _ in range(2)
    )
    expected_num_threads = max(cpu_count() // 2, 1)
    for worker_env_vars, omp_num_threads in results:
        assert omp_num_threads == expected_num_threads
        for name, value in worker_env_vars.items():
            if name.endswith("_THREADS"):
                assert value == str(expected_num_threads)
            else:
                assert name == "ENABLE_IPC"
                assert value == "1"


@skipif(distributed is not None,
        reason='This test requires dask NOT installed')
@parametrize("context", [parallel_config, parallel_backend])
def test_dask_backend_when_dask_not_installed(context):
    with raises(ValueError, match='Please install dask'):
        context('dask')


@parametrize("context", [parallel_config, parallel_backend])
def test_zero_worker_backend(context):
    # joblib.Parallel should reject with an explicit error message parallel
    # backends that have no worker.
    class ZeroWorkerBackend(ThreadingBackend):
        def configure(self, *args, **kwargs):
            return 0

        def apply_async(self, func, callback=None):   # pragma: no cover
            raise TimeoutError("No worker available")

        def effective_n_jobs(self, n_jobs):   # pragma: no cover
            return 0

    expected_msg = "ZeroWorkerBackend has no active worker"
    with context(ZeroWorkerBackend()):
        with pytest.raises(RuntimeError, match=expected_msg):
            Parallel(n_jobs=2)(delayed(id)(i) for i in range(2))


def test_globals_update_at_each_parallel_call():
    # This is a non-regression test related to joblib issues #836 and #833.
    # Cloudpickle versions between 0.5.4 and 0.7 introduced a bug where global
    # variables changes in a parent process between two calls to
    # joblib.Parallel would not be propagated into the workers.
    global MY_GLOBAL_VARIABLE
    MY_GLOBAL_VARIABLE = "original value"

    def check_globals():
        global MY_GLOBAL_VARIABLE
        return MY_GLOBAL_VARIABLE

    assert check_globals() == "original value"

    workers_global_variable = Parallel(n_jobs=2)(
        delayed(check_globals)() for i in range(2))
    assert set(workers_global_variable) == {"original value"}

    # Change the value of MY_GLOBAL_VARIABLE, and make sure this change gets
    # propagated into the workers environment
    MY_GLOBAL_VARIABLE = "changed value"
    assert check_globals() == "changed value"

    workers_global_variable = Parallel(n_jobs=2)(
        delayed(check_globals)() for i in range(2))
    assert set(workers_global_variable) == {"changed value"}


##############################################################################
# Test environment variable in child env, in particular for limiting
# the maximal number of threads in C-library threadpools.
#

def _check_numpy_threadpool_limits():
    import numpy as np
    # Let's call BLAS on a Matrix Matrix multiplication with dimensions large
    # enough to ensure that the threadpool managed by the underlying BLAS
    # implementation is actually used so as to force its initialization.
    a = np.random.randn(100, 100)
    np.dot(a, a)
    from threadpoolctl import threadpool_info
    return threadpool_info()


def _parent_max_num_threads_for(child_module, parent_info):
    for parent_module in parent_info:
        if parent_module['filepath'] == child_module['filepath']:
            return parent_module['num_threads']
    raise ValueError("An unexpected module was loaded in child:\n{}"
                     .format(child_module))


def check_child_num_threads(workers_info, parent_info, num_threads):
    # Check that the number of threads reported in workers_info is consistent
    # with the expectation. We need to be careful to handle the cases where
    # the requested number of threads is below max_num_thread for the library.
    for child_threadpool_info in workers_info:
        for child_module in child_threadpool_info:
            parent_max_num_threads = _parent_max_num_threads_for(
                child_module, parent_info)
            expected = {min(num_threads, parent_max_num_threads), num_threads}
            assert child_module['num_threads'] in expected


@with_numpy
@with_multiprocessing
@parametrize('n_jobs', [2, 4, -2, -1])
def test_threadpool_limitation_in_child_loky(n_jobs):
    # Check that the protection against oversubscription in workers is working
    # using threadpoolctl functionalities.

    # Skip this test if numpy is not linked to a BLAS library
    parent_info = _check_numpy_threadpool_limits()
    if len(parent_info) == 0:
        pytest.skip(msg="Need a version of numpy linked to BLAS")

    workers_threadpool_infos = Parallel(backend="loky", n_jobs=n_jobs)(
        delayed(_check_numpy_threadpool_limits)() for i in range(2))

    n_jobs = effective_n_jobs(n_jobs)
    expected_child_num_threads = max(cpu_count() // n_jobs, 1)

    check_child_num_threads(workers_threadpool_infos, parent_info,
                            expected_child_num_threads)


@with_numpy
@with_multiprocessing
@parametrize('inner_max_num_threads', [1, 2, 4, None])
@parametrize('n_jobs', [2, -1])
@parametrize("context", [parallel_config, parallel_backend])
def test_threadpool_limitation_in_child_context(
    context, n_jobs, inner_max_num_threads
):
    # Check that the protection against oversubscription in workers is working
    # using threadpoolctl functionalities.

    # Skip this test if numpy is not linked to a BLAS library
    parent_info = _check_numpy_threadpool_limits()
    if len(parent_info) == 0:
        pytest.skip(msg="Need a version of numpy linked to BLAS")

    with context('loky', inner_max_num_threads=inner_max_num_threads):
        workers_threadpool_infos = Parallel(n_jobs=n_jobs)(
            delayed(_check_numpy_threadpool_limits)() for i in range(2))

    n_jobs = effective_n_jobs(n_jobs)
    if inner_max_num_threads is None:
        expected_child_num_threads = max(cpu_count() // n_jobs, 1)
    else:
        expected_child_num_threads = inner_max_num_threads

    check_child_num_threads(workers_threadpool_infos, parent_info,
                            expected_child_num_threads)


@with_multiprocessing
@parametrize('n_jobs', [2, -1])
@parametrize('var_name', ["OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS",
                          "OMP_NUM_THREADS"])
@parametrize("context", [parallel_config, parallel_backend])
def test_threadpool_limitation_in_child_override(context, n_jobs, var_name):
    # Check that environment variables set by the user on the main process
    # always have the priority.

    # Clean up the existing executor because we change the environment of the
    # parent at runtime and it is not detected in loky intentionally.
    get_reusable_executor(reuse=True).shutdown()

    def _get_env(var_name):
        return os.environ.get(var_name)

    original_var_value = os.environ.get(var_name)
    try:
        os.environ[var_name] = "4"
        # Skip this test if numpy is not linked to a BLAS library
        results = Parallel(n_jobs=n_jobs)(
            delayed(_get_env)(var_name) for i in range(2))
        assert results == ["4", "4"]

        with context('loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_get_env)(var_name) for i in range(2))
        assert results == ["1", "1"]

    finally:
        if original_var_value is None:
            del os.environ[var_name]
        else:
            os.environ[var_name] = original_var_value


@with_multiprocessing
@parametrize('n_jobs', [2, 4, -1])
def test_loky_reuse_workers(n_jobs):
    # Non-regression test for issue #967 where the workers are not reused when
    # calling multiple Parallel loops.

    def parallel_call(n_jobs):
        x = range(10)
        Parallel(n_jobs=n_jobs)(delayed(sum)(x) for i in range(10))

    # Run a parallel loop and get the workers used for computations
    parallel_call(n_jobs)
    first_executor = get_reusable_executor(reuse=True)

    # Ensure that the workers are reused for the next calls, as the executor is
    # not restarted.
    for _ in range(10):
        parallel_call(n_jobs)
        executor = get_reusable_executor(reuse=True)
        assert executor == first_executor
