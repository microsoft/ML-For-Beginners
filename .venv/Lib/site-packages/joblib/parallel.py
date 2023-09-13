"""
Helpers for embarrassingly parallel code.
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause

from __future__ import division

import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext

from multiprocessing import TimeoutError

from ._multiprocessing_helpers import mp

from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
                                 ThreadingBackend, SequentialBackend,
                                 LokyBackend)
from ._utils import eval_expr, _Sentinel

# Make sure that those two classes are part of the public joblib.parallel API
# so that 3rd party backend implementers can import them from here.
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa


IS_PYPY = hasattr(sys, "pypy_version_info")


BACKENDS = {
    'threading': ThreadingBackend,
    'sequential': SequentialBackend,
}
# name of the backend used by default by Parallel outside of any context
# managed by ``parallel_config`` or ``parallel_backend``.

# threading is the only backend that is always everywhere
DEFAULT_BACKEND = 'threading'

MAYBE_AVAILABLE_BACKENDS = {'multiprocessing', 'loky'}

# if multiprocessing is available, so is loky, we set it as the default
# backend
if mp is not None:
    BACKENDS['multiprocessing'] = MultiprocessingBackend
    from .externals import loky
    BACKENDS['loky'] = LokyBackend
    DEFAULT_BACKEND = 'loky'


DEFAULT_THREAD_BACKEND = 'threading'


# Thread local value that can be overridden by the ``parallel_config`` context
# manager
_backend = threading.local()


def _register_dask():
    """Register Dask Backend if called with parallel_config(backend="dask")"""
    try:
        from ._dask import DaskDistributedBackend
        register_parallel_backend('dask', DaskDistributedBackend)
    except ImportError as e:
        msg = ("To use the dask.distributed backend you must install both "
               "the `dask` and distributed modules.\n\n"
               "See https://dask.pydata.org/en/latest/install.html for more "
               "information.")
        raise ImportError(msg) from e


EXTERNAL_BACKENDS = {
    'dask': _register_dask,
}


# Sentinels for the default values of the Parallel constructor and
# the parallel_config and parallel_backend context managers
default_parallel_config = {
    "backend": _Sentinel(default_value=None),
    "n_jobs": _Sentinel(default_value=None),
    "verbose": _Sentinel(default_value=0),
    "temp_folder": _Sentinel(default_value=None),
    "max_nbytes": _Sentinel(default_value="1M"),
    "mmap_mode": _Sentinel(default_value="r"),
    "prefer": _Sentinel(default_value=None),
    "require": _Sentinel(default_value=None),
}


VALID_BACKEND_HINTS = ('processes', 'threads', None)
VALID_BACKEND_CONSTRAINTS = ('sharedmem', None)


def _get_config_param(param, context_config, key):
    """Return the value of a parallel config parameter

    Explicitly setting it in Parallel has priority over setting in a
    parallel_(config/backend) context manager.
    """
    if param is not default_parallel_config[key]:
        # param is explicitely set, return it
        return param

    if context_config[key] is not default_parallel_config[key]:
        # there's a context manager and the key is set, return it
        return context_config[key]

    # Otherwise, we are in the default_parallel_config,
    # return the default value
    return param.default_value


def get_active_backend(
    prefer=default_parallel_config["prefer"],
    require=default_parallel_config["require"],
    verbose=default_parallel_config["verbose"],
):
    """Return the active default backend"""
    backend, config = _get_active_backend(prefer, require, verbose)
    n_jobs = _get_config_param(
        default_parallel_config['n_jobs'], config, "n_jobs"
    )
    return backend, n_jobs


def _get_active_backend(
    prefer=default_parallel_config["prefer"],
    require=default_parallel_config["require"],
    verbose=default_parallel_config["verbose"],
):
    """Return the active default backend"""

    backend_config = getattr(_backend, "config", default_parallel_config)

    backend = _get_config_param(
        default_parallel_config['backend'], backend_config, "backend"
    )
    prefer = _get_config_param(prefer, backend_config, "prefer")
    require = _get_config_param(require, backend_config, "require")
    verbose = _get_config_param(verbose, backend_config, "verbose")

    if prefer not in VALID_BACKEND_HINTS:
        raise ValueError(
            f"prefer={prefer} is not a valid backend hint, "
            f"expected one of {VALID_BACKEND_HINTS}"
        )
    if require not in VALID_BACKEND_CONSTRAINTS:
        raise ValueError(
            f"require={require} is not a valid backend constraint, "
            f"expected one of {VALID_BACKEND_CONSTRAINTS}"
        )
    if prefer == 'processes' and require == 'sharedmem':
        raise ValueError(
            "prefer == 'processes' and require == 'sharedmem'"
            " are inconsistent settings"
        )

    explicit_backend = True
    if backend is None:

        # We are either outside of the scope of any parallel_(config/backend)
        # context manager or the context manager did not set a backend.
        # create the default backend instance now.
        backend = BACKENDS[DEFAULT_BACKEND](nesting_level=0)
        explicit_backend = False

    # Try to use the backend set by the user with the context manager.

    nesting_level = backend.nesting_level
    uses_threads = getattr(backend, 'uses_threads', False)
    supports_sharedmem = getattr(backend, 'supports_sharedmem', False)
    # Force to use thread-based backend if the provided backend does not
    # match the shared memory constraint or if the backend is not explicitely
    # given and threads are prefered.
    force_threads = (require == 'sharedmem' and not supports_sharedmem)
    force_threads |= (
        not explicit_backend and prefer == 'threads' and not uses_threads
    )
    if force_threads:
        # This backend does not match the shared memory constraint:
        # fallback to the default thead-based backend.
        sharedmem_backend = BACKENDS[DEFAULT_THREAD_BACKEND](
            nesting_level=nesting_level
        )
        # Warn the user if we forced the backend to thread-based, while the
        # user explicitely specified a non-thread-based backend.
        if verbose >= 10 and explicit_backend:
            print(
                f"Using {sharedmem_backend.__class__.__name__} as "
                f"joblib backend instead of {backend.__class__.__name__} "
                "as the latter does not provide shared memory semantics."
            )
        # Force to n_jobs=1 by default
        thread_config = backend_config.copy()
        thread_config['n_jobs'] = 1
        return sharedmem_backend, thread_config

    return backend, backend_config


class parallel_config:
    """Set the default backend or configuration for :class:`~joblib.Parallel`.

    This is an alternative to directly passing keyword arguments to the
    :class:`~joblib.Parallel` class constructor. It is particularly useful when
    calling into library code that uses joblib internally but does not expose
    the various parallel configuration arguments in its own API.

    Parameters
    ----------
    backend : str or ParallelBackendBase instance, default=None
        If ``backend`` is a string it must match a previously registered
        implementation using the :func:`~register_parallel_backend` function.

        By default the following backends are available:

        - 'loky': single-host, process-based parallelism (used by default),
        - 'threading': single-host, thread-based parallelism,
        - 'multiprocessing': legacy single-host, process-based parallelism.

        'loky' is recommended to run functions that manipulate Python objects.
        'threading' is a low-overhead alternative that is most efficient for
        functions that release the Global Interpreter Lock: e.g. I/O-bound
        code or CPU-bound code in a few calls to native code that explicitly
        releases the GIL. Note that on some rare systems (such as pyodide),
        multiprocessing and loky may not be available, in which case joblib
        defaults to threading.

        In addition, if the ``dask`` and ``distributed`` Python packages are
        installed, it is possible to use the 'dask' backend for better
        scheduling of nested parallel calls without over-subscription and
        potentially distribute parallel calls over a networked cluster of
        several hosts.

        It is also possible to use the distributed 'ray' backend for
        distributing the workload to a cluster of nodes. See more details
        in the Examples section below.

        Alternatively the backend can be passed directly as an instance.

    n_jobs : int, default=None
        The maximum number of concurrently running jobs, such as the number
        of Python worker processes when ``backend="loky"`` or the size of the
        thread-pool when ``backend="threading"``.
        If -1 all CPUs are used. If 1 is given, no parallel computing code
        is used at all, which is useful for debugging. For ``n_jobs`` below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for ``n_jobs=-2``, all
        CPUs but one are used.
        ``None`` is a marker for 'unset' that will be interpreted as
        ``n_jobs=1`` in most backends.

    verbose : int, default=0
        The verbosity level: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    temp_folder : str, default=None
        Folder to be used by the pool for memmapping large arrays
        for sharing memory with worker processes. If None, this will try in
        order:

        - a folder pointed by the ``JOBLIB_TEMP_FOLDER`` environment
          variable,
        - ``/dev/shm`` if the folder exists and is writable: this is a
          RAM disk filesystem available by default on modern Linux
          distributions,
        - the default system temporary folder that can be
          overridden with ``TMP``, ``TMPDIR`` or ``TEMP`` environment
          variables, typically ``/tmp`` under Unix operating systems.

    max_nbytes int, str, or None, optional, default='1M'
        Threshold on the size of arrays passed to the workers that
        triggers automated memory mapping in temp_folder. Can be an int
        in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
        Use None to disable memmapping of large arrays.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, default='r'
        Memmapping mode for numpy arrays passed to workers. None will
        disable memmapping, other modes defined in the numpy.memmap doc:
        https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        Also, see 'max_nbytes' parameter documentation for more details.

    prefer: str in {'processes', 'threads'} or None, default=None
        Soft hint to choose the default backend.
        The default process-based backend is 'loky' and the default
        thread-based backend is 'threading'. Ignored if the ``backend``
        parameter is specified.

    require: 'sharedmem' or None, default=None
        Hard constraint to select the backend. If set to 'sharedmem',
        the selected backend will be single-host and thread-based.

    inner_max_num_threads : int, default=None
        If not None, overwrites the limit set on the number of threads
        usable in some third-party library threadpools like OpenBLAS,
        MKL or OpenMP. This is only used with the ``loky`` backend.

    backend_params : dict
        Additional parameters to pass to the backend constructor when
        backend is a string.

    Notes
    -----
    Joblib tries to limit the oversubscription by limiting the number of
    threads usable in some third-party library threadpools like OpenBLAS, MKL
    or OpenMP. The default limit in each worker is set to
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
    overwritten with the ``inner_max_num_threads`` argument which will be used
    to set this limit in the child processes.

    .. versionadded:: 1.3

    Examples
    --------
    >>> from operator import neg
    >>> with parallel_config(backend='threading'):
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    ...
    [-1, -2, -3, -4, -5]

    To use the 'ray' joblib backend add the following lines:

    >>> from ray.util.joblib import register_ray  # doctest: +SKIP
    >>> register_ray()  # doctest: +SKIP
    >>> with parallel_config(backend="ray"):  # doctest: +SKIP
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    [-1, -2, -3, -4, -5]

    """
    def __init__(
        self,
        backend=default_parallel_config["backend"],
        *,
        n_jobs=default_parallel_config["n_jobs"],
        verbose=default_parallel_config["verbose"],
        temp_folder=default_parallel_config["temp_folder"],
        max_nbytes=default_parallel_config["max_nbytes"],
        mmap_mode=default_parallel_config["mmap_mode"],
        prefer=default_parallel_config["prefer"],
        require=default_parallel_config["require"],
        inner_max_num_threads=None,
        **backend_params
    ):
        # Save the parallel info and set the active parallel config
        self.old_parallel_config = getattr(
            _backend, "config", default_parallel_config
        )

        backend = self._check_backend(
            backend, inner_max_num_threads, **backend_params
        )

        new_config = {
            "n_jobs": n_jobs,
            "verbose": verbose,
            "temp_folder": temp_folder,
            "max_nbytes": max_nbytes,
            "mmap_mode": mmap_mode,
            "prefer": prefer,
            "require": require,
            "backend": backend
        }
        self.parallel_config = self.old_parallel_config.copy()
        self.parallel_config.update({
            k: v for k, v in new_config.items()
            if not isinstance(v, _Sentinel)
        })

        setattr(_backend, "config", self.parallel_config)

    def _check_backend(self, backend, inner_max_num_threads, **backend_params):
        if backend is default_parallel_config['backend']:
            if inner_max_num_threads is not None or len(backend_params) > 0:
                raise ValueError(
                    "inner_max_num_threads and other constructor "
                    "parameters backend_params are only supported "
                    "when backend is not None."
                )
            return backend

        if isinstance(backend, str):
            # Handle non-registered or missing backends
            if backend not in BACKENDS:
                if backend in EXTERNAL_BACKENDS:
                    register = EXTERNAL_BACKENDS[backend]
                    register()
                elif backend in MAYBE_AVAILABLE_BACKENDS:
                    warnings.warn(
                        f"joblib backend '{backend}' is not available on "
                        f"your system, falling back to {DEFAULT_BACKEND}.",
                        UserWarning,
                        stacklevel=2
                    )
                    BACKENDS[backend] = BACKENDS[DEFAULT_BACKEND]
                else:
                    raise ValueError(
                        f"Invalid backend: {backend}, expected one of "
                        f"{sorted(BACKENDS.keys())}"
                    )

            backend = BACKENDS[backend](**backend_params)

        if inner_max_num_threads is not None:
            msg = (
                f"{backend.__class__.__name__} does not accept setting the "
                "inner_max_num_threads argument."
            )
            assert backend.supports_inner_max_num_threads, msg
            backend.inner_max_num_threads = inner_max_num_threads

        # If the nesting_level of the backend is not set previously, use the
        # nesting level from the previous active_backend to set it
        if backend.nesting_level is None:
            parent_backend = self.old_parallel_config['backend']
            if parent_backend is default_parallel_config['backend']:
                nesting_level = 0
            else:
                nesting_level = parent_backend.nesting_level
            backend.nesting_level = nesting_level

        return backend

    def __enter__(self):
        return self.parallel_config

    def __exit__(self, type, value, traceback):
        self.unregister()

    def unregister(self):
        setattr(_backend, "config", self.old_parallel_config)


class parallel_backend(parallel_config):
    """Change the default backend used by Parallel inside a with block.

    .. warning::
        It is advised to use the :class:`~joblib.parallel_config` context
        manager instead, which allows more fine-grained control over the
        backend configuration.

    If ``backend`` is a string it must match a previously registered
    implementation using the :func:`~register_parallel_backend` function.

    By default the following backends are available:

    - 'loky': single-host, process-based parallelism (used by default),
    - 'threading': single-host, thread-based parallelism,
    - 'multiprocessing': legacy single-host, process-based parallelism.

    'loky' is recommended to run functions that manipulate Python objects.
    'threading' is a low-overhead alternative that is most efficient for
    functions that release the Global Interpreter Lock: e.g. I/O-bound code or
    CPU-bound code in a few calls to native code that explicitly releases the
    GIL. Note that on some rare systems (such as Pyodide),
    multiprocessing and loky may not be available, in which case joblib
    defaults to threading.

    You can also use the `Dask <https://docs.dask.org/en/stable/>`_ joblib
    backend to distribute work across machines. This works well with
    scikit-learn estimators with the ``n_jobs`` parameter, for example::

    >>> import joblib  # doctest: +SKIP
    >>> from sklearn.model_selection import GridSearchCV  # doctest: +SKIP
    >>> from dask.distributed import Client, LocalCluster # doctest: +SKIP

    >>> # create a local Dask cluster
    >>> cluster = LocalCluster()  # doctest: +SKIP
    >>> client = Client(cluster)  # doctest: +SKIP
    >>> grid_search = GridSearchCV(estimator, param_grid, n_jobs=-1)
    ... # doctest: +SKIP
    >>> with joblib.parallel_backend("dask", scatter=[X, y]):  # doctest: +SKIP
    ...     grid_search.fit(X, y)

    It is also possible to use the distributed 'ray' backend for distributing
    the workload to a cluster of nodes. To use the 'ray' joblib backend add
    the following lines::

     >>> from ray.util.joblib import register_ray  # doctest: +SKIP
     >>> register_ray()  # doctest: +SKIP
     >>> with parallel_backend("ray"):  # doctest: +SKIP
     ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
     [-1, -2, -3, -4, -5]

    Alternatively the backend can be passed directly as an instance.

    By default all available workers will be used (``n_jobs=-1``) unless the
    caller passes an explicit value for the ``n_jobs`` parameter.

    This is an alternative to passing a ``backend='backend_name'`` argument to
    the :class:`~Parallel` class constructor. It is particularly useful when
    calling into library code that uses joblib internally but does not expose
    the backend argument in its own API.

    >>> from operator import neg
    >>> with parallel_backend('threading'):
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    ...
    [-1, -2, -3, -4, -5]

    Joblib also tries to limit the oversubscription by limiting the number of
    threads usable in some third-party library threadpools like OpenBLAS, MKL
    or OpenMP. The default limit in each worker is set to
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
    overwritten with the ``inner_max_num_threads`` argument which will be used
    to set this limit in the child processes.

    .. versionadded:: 0.10

    See Also
    --------
    joblib.parallel_config : context manager to change the backend
        configuration.
    """
    def __init__(self, backend, n_jobs=-1, inner_max_num_threads=None,
                 **backend_params):

        super().__init__(
            backend=backend,
            n_jobs=n_jobs,
            inner_max_num_threads=inner_max_num_threads,
            **backend_params
        )

        if self.old_parallel_config is None:
            self.old_backend_and_jobs = None
        else:
            self.old_backend_and_jobs = (
                self.old_parallel_config["backend"],
                self.old_parallel_config["n_jobs"],
            )
        self.new_backend_and_jobs = (
            self.parallel_config["backend"],
            self.parallel_config["n_jobs"],
        )

    def __enter__(self):
        return self.new_backend_and_jobs


# Under Linux or OS X the default start method of multiprocessing
# can cause third party libraries to crash. Under Python 3.4+ it is possible
# to set an environment variable to switch the default start method from
# 'fork' to 'forkserver' or 'spawn' to avoid this issue albeit at the cost
# of causing semantic changes and some additional pool instantiation overhead.
DEFAULT_MP_CONTEXT = None
if hasattr(mp, 'get_context'):
    method = os.environ.get('JOBLIB_START_METHOD', '').strip() or None
    if method is not None:
        DEFAULT_MP_CONTEXT = mp.get_context(method=method)


class BatchedCalls(object):
    """Wrap a sequence of (func, args, kwargs) tuples as a single callable"""

    def __init__(self, iterator_slice, backend_and_jobs, reducer_callback=None,
                 pickle_cache=None):
        self.items = list(iterator_slice)
        self._size = len(self.items)
        self._reducer_callback = reducer_callback
        if isinstance(backend_and_jobs, tuple):
            self._backend, self._n_jobs = backend_and_jobs
        else:
            # this is for backward compatibility purposes. Before 0.12.6,
            # nested backends were returned without n_jobs indications.
            self._backend, self._n_jobs = backend_and_jobs, None
        self._pickle_cache = pickle_cache if pickle_cache is not None else {}

    def __call__(self):
        # Set the default nested backend to self._backend but do not set the
        # change the default number of processes to -1
        with parallel_config(backend=self._backend, n_jobs=self._n_jobs):
            return [func(*args, **kwargs)
                    for func, args, kwargs in self.items]

    def __reduce__(self):
        if self._reducer_callback is not None:
            self._reducer_callback()
        # no need to pickle the callback.
        return (
            BatchedCalls,
            (self.items, (self._backend, self._n_jobs), None,
             self._pickle_cache)
        )

    def __len__(self):
        return self._size


# Possible exit status for a task
TASK_DONE = "Done"
TASK_ERROR = "Error"
TASK_PENDING = "Pending"


###############################################################################
# CPU count that works also when multiprocessing has been disabled via
# the JOBLIB_MULTIPROCESSING environment variable
def cpu_count(only_physical_cores=False):
    """Return the number of CPUs.

    This delegates to loky.cpu_count that takes into account additional
    constraints such as Linux CFS scheduler quotas (typically set by container
    runtimes such as docker) and CPU affinity (for instance using the taskset
    command on Linux).

    If only_physical_cores is True, do not take hyperthreading / SMT logical
    cores into account.
    """
    if mp is None:
        return 1

    return loky.cpu_count(only_physical_cores=only_physical_cores)


###############################################################################
# For verbosity

def _verbosity_filter(index, verbose):
    """ Returns False for indices increasingly apart, the distance
        depending on the value of verbose.

        We use a lag increasing as the square of index
    """
    if not verbose:
        return True
    elif verbose > 10:
        return False
    if index == 0:
        return False
    verbose = .5 * (11 - verbose) ** 2
    scale = sqrt(index / verbose)
    next_scale = sqrt((index + 1) / verbose)
    return (int(next_scale) == int(scale))


###############################################################################
def delayed(function):
    """Decorator used to capture the arguments of a function."""

    def delayed_function(*args, **kwargs):
        return function, args, kwargs
    try:
        delayed_function = functools.wraps(function)(delayed_function)
    except AttributeError:
        " functools.wraps fails on some callable objects "
    return delayed_function


###############################################################################
class BatchCompletionCallBack(object):
    """Callback to keep track of completed results and schedule the next tasks.

    This callable is executed by the parent process whenever a worker process
    has completed a batch of tasks.

    It is used for progress reporting, to update estimate of the batch
    processing duration and to schedule the next batch of tasks to be
    processed.

    It is assumed that this callback will always be triggered by the backend
    right after the end of a task, in case of success as well as in case of
    failure.
    """

    ##########################################################################
    #                   METHODS CALLED BY THE MAIN THREAD                    #
    ##########################################################################
    def __init__(self, dispatch_timestamp, batch_size, parallel):
        self.dispatch_timestamp = dispatch_timestamp
        self.batch_size = batch_size
        self.parallel = parallel
        self.parallel_call_id = parallel._call_id

        # Internals to keep track of the status and outcome of the task.

        # Used to hold a reference to the future-like object returned by the
        # backend after launching this task
        # This will be set later when calling `register_job`, as it is only
        # created once the task has been submitted.
        self.job = None

        if not parallel._backend.supports_retrieve_callback:
            # The status is only used for asynchronous result retrieval in the
            # callback.
            self.status = None
        else:
            # The initial status for the job is TASK_PENDING.
            # Once it is done, it will be either TASK_DONE, or TASK_ERROR.
            self.status = TASK_PENDING

    def register_job(self, job):
        """Register the object returned by `apply_async`."""
        self.job = job

    def get_result(self, timeout):
        """Returns the raw result of the task that was submitted.

        If the task raised an exception rather than returning, this same
        exception will be raised instead.

        If the backend supports the retrieval callback, it is assumed that this
        method is only called after the result has been registered. It is
        ensured by checking that `self.status(timeout)` does not return
        TASK_PENDING. In this case, `get_result` directly returns the
        registered result (or raise the registered exception).

        For other backends, there are no such assumptions, but `get_result`
        still needs to synchronously retrieve the result before it can
        return it or raise. It will block at most `self.timeout` seconds
        waiting for retrieval to complete, after that it raises a TimeoutError.
        """

        backend = self.parallel._backend

        if backend.supports_retrieve_callback:
            # We assume that the result has already been retrieved by the
            # callback thread, and is stored internally. It's just waiting to
            # be returned.
            return self._return_or_raise()

        # For other backends, the main thread needs to run the retrieval step.
        try:
            if backend.supports_timeout:
                result = self.job.get(timeout=timeout)
            else:
                result = self.job.get()
            outcome = dict(result=result, status=TASK_DONE)
        except BaseException as e:
            outcome = dict(result=e, status=TASK_ERROR)
        self._register_outcome(outcome)

        return self._return_or_raise()

    def _return_or_raise(self):
        try:
            if self.status == TASK_ERROR:
                raise self._result
            return self._result
        finally:
            del self._result

    def get_status(self, timeout):
        """Get the status of the task.

        This function also checks if the timeout has been reached and register
        the TimeoutError outcome when it is the case.
        """
        if timeout is None or self.status != TASK_PENDING:
            return self.status

        # The computation are running and the status is pending.
        # Check that we did not wait for this jobs more than `timeout`.
        now = time.time()
        if not hasattr(self, "_completion_timeout_counter"):
            self._completion_timeout_counter = now

        if (now - self._completion_timeout_counter) > timeout:
            outcome = dict(result=TimeoutError(), status=TASK_ERROR)
            self._register_outcome(outcome)

        return self.status

    ##########################################################################
    #                     METHODS CALLED BY CALLBACK THREADS                 #
    ##########################################################################
    def __call__(self, out):
        """Function called by the callback thread after a job is completed."""

        # If the backend doesn't support callback retrievals, the next batch of
        # tasks is dispatched regardless. The result will be retrieved by the
        # main thread when calling `get_result`.
        if not self.parallel._backend.supports_retrieve_callback:
            self._dispatch_new()
            return

        # If the backend supports retrieving the result in the callback, it
        # registers the task outcome (TASK_ERROR or TASK_DONE), and schedules
        # the next batch if needed.
        with self.parallel._lock:
            # Edge case where while the task was processing, the `parallel`
            # instance has been reset and a new call has been issued, but the
            # worker managed to complete the task and trigger this callback
            # call just before being aborted by the reset.
            if self.parallel._call_id != self.parallel_call_id:
                return

            # When aborting, stop as fast as possible and do not retrieve the
            # result as it won't be returned by the Parallel call.
            if self.parallel._aborting:
                return

            # Retrieves the result of the task in the main process and dispatch
            # a new batch if needed.
            job_succeeded = self._retrieve_result(out)

        if job_succeeded:
            self._dispatch_new()

    def _dispatch_new(self):
        """Schedule the next batch of tasks to be processed."""

        # This steps ensure that auto-baching works as expected.
        this_batch_duration = time.time() - self.dispatch_timestamp
        self.parallel._backend.batch_completed(self.batch_size,
                                               this_batch_duration)

        # Schedule the next batch of tasks.
        with self.parallel._lock:
            self.parallel.n_completed_tasks += self.batch_size
            self.parallel.print_progress()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    def _retrieve_result(self, out):
        """Fetch and register the outcome of a task.

        Return True if the task succeeded, False otherwise.
        This function is only called by backends that support retrieving
        the task result in the callback thread.
        """
        try:
            result = self.parallel._backend.retrieve_result_callback(out)
            outcome = dict(status=TASK_DONE, result=result)
        except BaseException as e:
            # Avoid keeping references to parallel in the error.
            e.__traceback__ = None
            outcome = dict(result=e, status=TASK_ERROR)

        self._register_outcome(outcome)
        return outcome['status'] != TASK_ERROR

    ##########################################################################
    #            This method can be called either in the main thread         #
    #                        or in the callback thread.                      #
    ##########################################################################
    def _register_outcome(self, outcome):
        """Register the outcome of a task.

        This method can be called only once, future calls will be ignored.
        """
        # Covers the edge case where the main thread tries to register a
        # `TimeoutError` while the callback thread tries to register a result
        # at the same time.
        with self.parallel._lock:
            if self.status not in (TASK_PENDING, None):
                return
            self.status = outcome["status"]

        self._result = outcome["result"]

        # Once the result and the status are extracted, the last reference to
        # the job can be deleted.
        self.job = None

        # As soon as an error as been spotted, early stopping flags are sent to
        # the `parallel` instance.
        if self.status == TASK_ERROR:
            self.parallel._exception = True
            self.parallel._aborting = True


###############################################################################
def register_parallel_backend(name, factory, make_default=False):
    """Register a new Parallel backend factory.

    The new backend can then be selected by passing its name as the backend
    argument to the :class:`~Parallel` class. Moreover, the default backend can
    be overwritten globally by setting make_default=True.

    The factory can be any callable that takes no argument and return an
    instance of ``ParallelBackendBase``.

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    .. versionadded:: 0.10
    """
    BACKENDS[name] = factory
    if make_default:
        global DEFAULT_BACKEND
        DEFAULT_BACKEND = name


def effective_n_jobs(n_jobs=-1):
    """Determine the number of jobs that can actually run in parallel

    n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
    means requesting all available workers for instance matching the number of
    CPU cores on the worker host(s).

    This method should return a guesstimate of the number of workers that can
    actually perform work concurrently with the currently enabled default
    backend. The primary use case is to make it possible for the caller to know
    in how many chunks to slice the work.

    In general working on larger data chunks is more efficient (less scheduling
    overhead and better use of CPU cache prefetching heuristics) as long as all
    the workers have enough work to do.

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    .. versionadded:: 0.10
    """
    if n_jobs == 1:
        return 1

    backend, backend_n_jobs = get_active_backend()
    if n_jobs is None:
        n_jobs = backend_n_jobs
    return backend.effective_n_jobs(n_jobs=n_jobs)


###############################################################################
class Parallel(Logger):
    ''' Helper class for readable parallel mapping.

        Read more in the :ref:`User Guide <parallel>`.

        Parameters
        ----------
        n_jobs: int, default: None
            The maximum number of concurrently running jobs, such as the number
            of Python worker processes when backend="multiprocessing"
            or the size of the thread-pool when backend="threading".
            If -1 all CPUs are used.
            If 1 is given, no parallel computing code is used at all, and the
            behavior amounts to a simple python `for` loop. This mode is not
            compatible with `timeout`.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used.
            None is a marker for 'unset' that will be interpreted as n_jobs=1
            unless the call is performed under a :func:`~parallel_config`
            context manager that sets another value for ``n_jobs``.
        backend: str, ParallelBackendBase instance or None, default: 'loky'
            Specify the parallelization backend implementation.
            Supported backends are:

            - "loky" used by default, can induce some
              communication and memory overhead when exchanging input and
              output data with the worker Python processes. On some rare
              systems (such as Pyiodide), the loky backend may not be
              available.
            - "multiprocessing" previous process-based backend based on
              `multiprocessing.Pool`. Less robust than `loky`.
            - "threading" is a very low-overhead backend but it suffers
              from the Python Global Interpreter Lock if the called function
              relies a lot on Python objects. "threading" is mostly useful
              when the execution bottleneck is a compiled extension that
              explicitly releases the GIL (for instance a Cython loop wrapped
              in a "with nogil" block or an expensive call to a library such
              as NumPy).
            - finally, you can register backends by calling
              :func:`~register_parallel_backend`. This will allow you to
              implement a backend of your liking.

            It is not recommended to hard-code the backend name in a call to
            :class:`~Parallel` in a library. Instead it is recommended to set
            soft hints (prefer) or hard constraints (require) so as to make it
            possible for library users to change the backend from the outside
            using the :func:`~parallel_config` context manager.
        return_as: str in {'list', 'generator'}, default: 'list'
            If 'list', calls to this instance will return a list, only when
            all results have been processed and retrieved.
            If 'generator', it will return a generator that yields the results
            as soon as they are available, in the order the tasks have been
            submitted with.
            Future releases are planned to also support 'generator_unordered',
            in which case the generator immediately yields available results
            independently of the submission order.
        prefer: str in {'processes', 'threads'} or None, default: None
            Soft hint to choose the default backend if no specific backend
            was selected with the :func:`~parallel_config` context manager.
            The default process-based backend is 'loky' and the default
            thread-based backend is 'threading'. Ignored if the ``backend``
            parameter is specified.
        require: 'sharedmem' or None, default None
            Hard constraint to select the backend. If set to 'sharedmem',
            the selected backend will be single-host and thread-based even
            if the user asked for a non-thread based backend with
            :func:`~joblib.parallel_config`.
        verbose: int, optional
            The verbosity level: if non zero, progress messages are
            printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
        timeout: float, optional
            Timeout limit for each task to complete.  If any task takes longer
            a TimeOutError will be raised. Only applied when n_jobs != 1
        pre_dispatch: {'all', integer, or expression, as in '3*n_jobs'}
            The number of batches (of tasks) to be pre-dispatched.
            Default is '2*n_jobs'. When batch_size="auto" this is reasonable
            default and the workers should never starve. Note that only basic
            arithmetics are allowed here and no modules can be used in this
            expression.
        batch_size: int or 'auto', default: 'auto'
            The number of atomic tasks to dispatch at once to each
            worker. When individual evaluations are very fast, dispatching
            calls to workers can be slower than sequential computation because
            of the overhead. Batching fast computations together can mitigate
            this.
            The ``'auto'`` strategy keeps track of the time it takes for a
            batch to complete, and dynamically adjusts the batch size to keep
            the time on the order of half a second, using a heuristic. The
            initial batch size is 1.
            ``batch_size="auto"`` with ``backend="threading"`` will dispatch
            batches of a single task at a time as the threading backend has
            very little overhead and using larger batch size has not proved to
            bring any gain in that case.
        temp_folder: str, optional
            Folder to be used by the pool for memmapping large arrays
            for sharing memory with worker processes. If None, this will try in
            order:

            - a folder pointed by the JOBLIB_TEMP_FOLDER environment
              variable,
            - /dev/shm if the folder exists and is writable: this is a
              RAM disk filesystem available by default on modern Linux
              distributions,
            - the default system temporary folder that can be
              overridden with TMP, TMPDIR or TEMP environment
              variables, typically /tmp under Unix operating systems.

            Only active when backend="loky" or "multiprocessing".
        max_nbytes int, str, or None, optional, 1M by default
            Threshold on the size of arrays passed to the workers that
            triggers automated memory mapping in temp_folder. Can be an int
            in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
            Use None to disable memmapping of large arrays.
            Only active when backend="loky" or "multiprocessing".
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, default: 'r'
            Memmapping mode for numpy arrays passed to workers. None will
            disable memmapping, other modes defined in the numpy.memmap doc:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
            Also, see 'max_nbytes' parameter documentation for more details.

        Notes
        -----

        This object uses workers to compute in parallel the application of a
        function to many different arguments. The main functionality it brings
        in addition to using the raw multiprocessing or concurrent.futures API
        are (see examples for details):

        * More readable code, in particular since it avoids
          constructing list of arguments.

        * Easier debugging:
            - informative tracebacks even when the error happens on
              the client side
            - using 'n_jobs=1' enables to turn off parallel computing
              for debugging without changing the codepath
            - early capture of pickling errors

        * An optional progress meter.

        * Interruption of multiprocesses jobs with 'Ctrl-C'

        * Flexible pickling control for the communication to and from
          the worker processes.

        * Ability to use shared memory efficiently with worker
          processes for large numpy-based datastructures.

        Note that the intended usage is to run one call at a time. Multiple
        calls to the same Parallel object will result in a ``RuntimeError``

        Examples
        --------

        A simple example:

        >>> from math import sqrt
        >>> from joblib import Parallel, delayed
        >>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        Reshaping the output when the function has several return
        values:

        >>> from math import modf
        >>> from joblib import Parallel, delayed
        >>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))
        >>> res, i = zip(*r)
        >>> res
        (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
        >>> i
        (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)

        The progress meter: the higher the value of `verbose`, the more
        messages:

        >>> from time import sleep
        >>> from joblib import Parallel, delayed
        >>> r = Parallel(n_jobs=2, verbose=10)(
        ...     delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP
        [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s
        [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s
        [Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished

        Traceback example, note how the line of the error is indicated
        as well as the values of the parameter passed to the function that
        triggered the exception, even though the traceback happens in the
        child process:

        >>> from heapq import nlargest
        >>> from joblib import Parallel, delayed
        >>> Parallel(n_jobs=2)(
        ... delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3))
        ... # doctest: +SKIP
        -----------------------------------------------------------------------
        Sub-process traceback:
        -----------------------------------------------------------------------
        TypeError                                      Mon Nov 12 11:37:46 2012
        PID: 12934                                Python 2.7.3: /usr/bin/python
        ........................................................................
        /usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)
            419         if n >= size:
            420             return sorted(iterable, key=key, reverse=True)[:n]
            421
            422     # When key is none, use simpler decoration
            423     if key is None:
        --> 424         it = izip(iterable, count(0,-1))           # decorate
            425         result = _nlargest(n, it)
            426         return map(itemgetter(0), result)          # undecorate
            427
            428     # General case, slowest method
         TypeError: izip argument #1 must support iteration
        _______________________________________________________________________


        Using pre_dispatch in a producer/consumer situation, where the
        data is generated on the fly. Note how the producer is first
        called 3 times before the parallel loop is initiated, and then
        called to generate new data on the fly:

        >>> from math import sqrt
        >>> from joblib import Parallel, delayed
        >>> def producer():
        ...     for i in range(6):
        ...         print('Produced %s' % i)
        ...         yield i
        >>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(
        ...     delayed(sqrt)(i) for i in producer()) #doctest: +SKIP
        Produced 0
        Produced 1
        Produced 2
        [Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s
        Produced 3
        [Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s
        Produced 4
        [Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s
        Produced 5
        [Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s finished

    '''
    def __init__(
        self,
        n_jobs=default_parallel_config["n_jobs"],
        backend=default_parallel_config['backend'],
        return_as="list",
        verbose=default_parallel_config["verbose"],
        timeout=None,
        pre_dispatch='2 * n_jobs',
        batch_size='auto',
        temp_folder=default_parallel_config["temp_folder"],
        max_nbytes=default_parallel_config["max_nbytes"],
        mmap_mode=default_parallel_config["mmap_mode"],
        prefer=default_parallel_config["prefer"],
        require=default_parallel_config["require"],
    ):
        # Initiate parent Logger class state
        super().__init__()

        # Interpret n_jobs=None as 'unset'
        if n_jobs is None:
            n_jobs = default_parallel_config["n_jobs"]

        active_backend, context_config = _get_active_backend(
            prefer=prefer, require=require, verbose=verbose
        )

        nesting_level = active_backend.nesting_level

        self.verbose = _get_config_param(verbose, context_config, "verbose")
        self.timeout = timeout
        self.pre_dispatch = pre_dispatch

        if return_as not in {"list", "generator"}:
            raise ValueError(
                'Expected `return_as` parameter to be a string equal to "list"'
                f' or "generator", but got {return_as} instead'
            )
        self.return_as = return_as
        self.return_generator = return_as != "list"

        # Check if we are under a parallel_config or parallel_backend
        # context manager and use the config from the context manager
        # for arguments that are not explicitly set.
        self._backend_args = {
            k: _get_config_param(param, context_config, k) for param, k in [
                (max_nbytes, "max_nbytes"),
                (temp_folder, "temp_folder"),
                (mmap_mode, "mmap_mode"),
                (prefer, "prefer"),
                (require, "require"),
                (verbose, "verbose"),
            ]
        }

        if isinstance(self._backend_args["max_nbytes"], str):
            self._backend_args["max_nbytes"] = memstr_to_bytes(
                self._backend_args["max_nbytes"]
            )
        self._backend_args["verbose"] = max(
            0, self._backend_args["verbose"] - 50
        )

        if DEFAULT_MP_CONTEXT is not None:
            self._backend_args['context'] = DEFAULT_MP_CONTEXT
        elif hasattr(mp, "get_context"):
            self._backend_args['context'] = mp.get_context()

        if backend is default_parallel_config['backend'] or backend is None:
            backend = active_backend

        elif isinstance(backend, ParallelBackendBase):
            # Use provided backend as is, with the current nesting_level if it
            # is not set yet.
            if backend.nesting_level is None:
                backend.nesting_level = nesting_level

        elif hasattr(backend, 'Pool') and hasattr(backend, 'Lock'):
            # Make it possible to pass a custom multiprocessing context as
            # backend to change the start method to forkserver or spawn or
            # preload modules on the forkserver helper process.
            self._backend_args['context'] = backend
            backend = MultiprocessingBackend(nesting_level=nesting_level)

        elif backend not in BACKENDS and backend in MAYBE_AVAILABLE_BACKENDS:
            warnings.warn(
                f"joblib backend '{backend}' is not available on "
                f"your system, falling back to {DEFAULT_BACKEND}.",
                UserWarning,
                stacklevel=2)
            BACKENDS[backend] = BACKENDS[DEFAULT_BACKEND]
            backend = BACKENDS[DEFAULT_BACKEND](nesting_level=nesting_level)

        else:
            try:
                backend_factory = BACKENDS[backend]
            except KeyError as e:
                raise ValueError("Invalid backend: %s, expected one of %r"
                                 % (backend, sorted(BACKENDS.keys()))) from e
            backend = backend_factory(nesting_level=nesting_level)

        n_jobs = _get_config_param(n_jobs, context_config, "n_jobs")
        if n_jobs is None:
            # No specific context override and no specific value request:
            # default to the default of the backend.
            n_jobs = backend.default_n_jobs
        self.n_jobs = n_jobs

        if (require == 'sharedmem' and
                not getattr(backend, 'supports_sharedmem', False)):
            raise ValueError("Backend %s does not support shared memory"
                             % backend)

        if (batch_size == 'auto' or isinstance(batch_size, Integral) and
                batch_size > 0):
            self.batch_size = batch_size
        else:
            raise ValueError(
                "batch_size must be 'auto' or a positive integer, got: %r"
                % batch_size)

        if not isinstance(backend, SequentialBackend):
            if self.return_generator and not backend.supports_return_generator:
                raise ValueError(
                    "Backend {} does not support "
                    "return_as={}".format(backend, return_as)
                )
            # This lock is used to coordinate the main thread of this process
            # with the async callback thread of our the pool.
            self._lock = threading.RLock()
            self._jobs = collections.deque()
            self._pending_outputs = list()
            self._ready_batches = queue.Queue()
            self._reducer_callback = None

        # Internal variables
        self._backend = backend
        self._running = False
        self._managed_backend = False
        self._id = uuid4().hex
        self._call_ref = None

    def __enter__(self):
        self._managed_backend = True
        self._calling = False
        self._initialize_backend()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._managed_backend = False
        if self.return_generator and self._calling:
            self._abort()
        self._terminate_and_reset()

    def _initialize_backend(self):
        """Build a process or thread pool and return the number of workers"""
        try:
            n_jobs = self._backend.configure(n_jobs=self.n_jobs, parallel=self,
                                             **self._backend_args)
            if self.timeout is not None and not self._backend.supports_timeout:
                warnings.warn(
                    'The backend class {!r} does not support timeout. '
                    "You have set 'timeout={}' in Parallel but "
                    "the 'timeout' parameter will not be used.".format(
                        self._backend.__class__.__name__,
                        self.timeout))

        except FallbackToBackend as e:
            # Recursively initialize the backend in case of requested fallback.
            self._backend = e.backend
            n_jobs = self._initialize_backend()

        return n_jobs

    def _effective_n_jobs(self):
        if self._backend:
            return self._backend.effective_n_jobs(self.n_jobs)
        return 1

    def _terminate_and_reset(self):
        if hasattr(self._backend, 'stop_call') and self._calling:
            self._backend.stop_call()
        self._calling = False
        if not self._managed_backend:
            self._backend.terminate()

    def _dispatch(self, batch):
        """Queue the batch for computing, with or without multiprocessing

        WARNING: this method is not thread-safe: it should be only called
        indirectly via dispatch_one_batch.

        """
        # If job.get() catches an exception, it closes the queue:
        if self._aborting:
            return

        batch_size = len(batch)

        self.n_dispatched_tasks += batch_size
        self.n_dispatched_batches += 1

        dispatch_timestamp = time.time()

        batch_tracker = BatchCompletionCallBack(
            dispatch_timestamp, batch_size, self
        )
        self._jobs.append(batch_tracker)

        job = self._backend.apply_async(batch, callback=batch_tracker)
        batch_tracker.register_job(job)

    def dispatch_next(self):
        """Dispatch more data for parallel processing

        This method is meant to be called concurrently by the multiprocessing
        callback. We rely on the thread-safety of dispatch_one_batch to protect
        against concurrent consumption of the unprotected iterator.

        """
        if not self.dispatch_one_batch(self._original_iterator):
            self._iterating = False
            self._original_iterator = None

    def dispatch_one_batch(self, iterator):
        """Prefetch the tasks for the next batch and dispatch them.

        The effective size of the batch is computed here.
        If there are no more jobs to dispatch, return False, else return True.

        The iterator consumption and dispatching is protected by the same
        lock so calling this function should be thread safe.

        """

        if self._aborting:
            return False

        batch_size = self._get_batch_size()

        with self._lock:
            # to ensure an even distribution of the workload between workers,
            # we look ahead in the original iterators more than batch_size
            # tasks - However, we keep consuming only one batch at each
            # dispatch_one_batch call. The extra tasks are stored in a local
            # queue, _ready_batches, that is looked-up prior to re-consuming
            # tasks from the origal iterator.
            try:
                tasks = self._ready_batches.get(block=False)
            except queue.Empty:
                # slice the iterator n_jobs * batchsize items at a time. If the
                # slice returns less than that, then the current batchsize puts
                # too much weight on a subset of workers, while other may end
                # up starving. So in this case, re-scale the batch size
                # accordingly to distribute evenly the last items between all
                # workers.
                n_jobs = self._cached_effective_n_jobs
                big_batch_size = batch_size * n_jobs

                islice = list(itertools.islice(iterator, big_batch_size))
                if len(islice) == 0:
                    return False
                elif (iterator is self._original_iterator and
                      len(islice) < big_batch_size):
                    # We reached the end of the original iterator (unless
                    # iterator is the ``pre_dispatch``-long initial slice of
                    # the original iterator) -- decrease the batch size to
                    # account for potential variance in the batches running
                    # time.
                    final_batch_size = max(1, len(islice) // (10 * n_jobs))
                else:
                    final_batch_size = max(1, len(islice) // n_jobs)

                # enqueue n_jobs batches in a local queue
                for i in range(0, len(islice), final_batch_size):
                    tasks = BatchedCalls(islice[i:i + final_batch_size],
                                         self._backend.get_nested_backend(),
                                         self._reducer_callback,
                                         self._pickle_cache)
                    self._ready_batches.put(tasks)

                # finally, get one task.
                tasks = self._ready_batches.get(block=False)
            if len(tasks) == 0:
                # No more tasks available in the iterator: tell caller to stop.
                return False
            else:
                self._dispatch(tasks)
                return True

    def _get_batch_size(self):
        """Returns the effective batch size for dispatch"""
        if self.batch_size == 'auto':
            return self._backend.compute_batch_size()
        else:
            # Fixed batch size strategy
            return self.batch_size

    def _print(self, msg):
        """Display the message on stout or stderr depending on verbosity"""
        # XXX: Not using the logger framework: need to
        # learn to use logger better.
        if not self.verbose:
            return
        if self.verbose < 50:
            writer = sys.stderr.write
        else:
            writer = sys.stdout.write
        writer(f"[{self}]: {msg}\n")

    def _is_completed(self):
        """Check if all tasks have been completed"""
        return self.n_completed_tasks == self.n_dispatched_tasks and not (
            self._iterating or self._aborting
        )

    def print_progress(self):
        """Display the process of the parallel execution only a fraction
           of time, controlled by self.verbose.
        """

        if not self.verbose:
            return

        elapsed_time = time.time() - self._start_time

        if self._is_completed():
            # Make sure that we get a last message telling us we are done
            self._print(
                f"Done {self.n_completed_tasks:3d} out of "
                f"{self.n_completed_tasks:3d} | elapsed: "
                f"{short_format_time(elapsed_time)} finished"
            )
            return

        # Original job iterator becomes None once it has been fully
        # consumed : at this point we know the total number of jobs and we are
        # able to display an estimation of the remaining time based on already
        # completed jobs. Otherwise, we simply display the number of completed
        # tasks.
        elif self._original_iterator is not None:
            if _verbosity_filter(self.n_dispatched_batches, self.verbose):
                return
            self._print(
                f"Done {self.n_completed_tasks:3d} tasks      | elapsed: "
                f"{short_format_time(elapsed_time)}"
            )
        else:
            index = self.n_completed_tasks
            # We are finished dispatching
            total_tasks = self.n_dispatched_tasks
            # We always display the first loop
            if not index == 0:
                # Display depending on the number of remaining items
                # A message as soon as we finish dispatching, cursor is 0
                cursor = (total_tasks - index + 1 -
                          self._pre_dispatch_amount)
                frequency = (total_tasks // self.verbose) + 1
                is_last_item = (index + 1 == total_tasks)
                if (is_last_item or cursor % frequency):
                    return
            remaining_time = (elapsed_time / index) * \
                             (self.n_dispatched_tasks - index * 1.0)
            # only display status if remaining time is greater or equal to 0
            self._print(
                f"Done {index:3d} out of {total_tasks:3d} | elapsed: "
                f"{short_format_time(elapsed_time)} remaining: "
                f"{short_format_time(remaining_time)}"
            )

    def _abort(self):
        # Stop dispatching new jobs in the async callback thread
        self._aborting = True

        # If the backend allows it, cancel or kill remaining running
        # tasks without waiting for the results as we will raise
        # the exception we got back to the caller instead of returning
        # any result.
        backend = self._backend
        if (not self._aborted and hasattr(backend, 'abort_everything')):
            # If the backend is managed externally we need to make sure
            # to leave it in a working state to allow for future jobs
            # scheduling.
            ensure_ready = self._managed_backend
            backend.abort_everything(ensure_ready=ensure_ready)
        self._aborted = True

    def _start(self, iterator, pre_dispatch):
        # Only set self._iterating to True if at least a batch
        # was dispatched. In particular this covers the edge
        # case of Parallel used with an exhausted iterator. If
        # self._original_iterator is None, then this means either
        # that pre_dispatch == "all", n_jobs == 1 or that the first batch
        # was very quick and its callback already dispatched all the
        # remaining jobs.
        self._iterating = False
        if self.dispatch_one_batch(iterator):
            self._iterating = self._original_iterator is not None

        while self.dispatch_one_batch(iterator):
            pass

        if pre_dispatch == "all":
            # The iterable was consumed all at once by the above for loop.
            # No need to wait for async callbacks to trigger to
            # consumption.
            self._iterating = False

    def _get_outputs(self, iterator, pre_dispatch):
        """Iterator returning the tasks' output as soon as they are ready."""
        dispatch_thread_id = threading.get_ident()
        detach_generator_exit = False
        try:
            self._start(iterator, pre_dispatch)
            # first yield returns None, for internal use only. This ensures
            # that we enter the try/except block and start dispatching the
            # tasks.
            yield

            with self._backend.retrieval_context():
                yield from self._retrieve()

        except GeneratorExit:
            # The generator has been garbage collected before being fully
            # consumed. This aborts the remaining tasks if possible and warn
            # the user if necessary.
            self._exception = True

            # In some interpreters such as PyPy, GeneratorExit can be raised in
            # a different thread than the one used to start the dispatch of the
            # parallel tasks. This can lead to hang when a thread attempts to
            # join itself. As workaround, we detach the execution of the
            # aborting code to a dedicated thread. We then need to make sure
            # the rest of the function does not call `_terminate_and_reset`
            # in finally.
            if dispatch_thread_id != threading.get_ident():
                if not IS_PYPY:
                    warnings.warn(
                        "A generator produced by joblib.Parallel has been "
                        "gc'ed in an unexpected thread. This behavior should "
                        "not cause major -issues but to make sure, please "
                        "report this warning and your use case at "
                        "https://github.com/joblib/joblib/issues so it can "
                        "be investigated."
                    )

                detach_generator_exit = True
                _parallel = self

                class _GeneratorExitThread(threading.Thread):
                    def run(self):
                        _parallel._abort()
                        if _parallel.return_generator:
                            _parallel._warn_exit_early()
                        _parallel._terminate_and_reset()

                _GeneratorExitThread(
                    name="GeneratorExitThread"
                ).start()
                return

            # Otherwise, we are in the thread that started the dispatch: we can
            # safely abort the execution and warn the user.
            self._abort()
            if self.return_generator:
                self._warn_exit_early()

            raise

        # Note: we catch any BaseException instead of just Exception instances
        # to also include KeyboardInterrupt
        except BaseException:
            self._exception = True
            self._abort()
            raise
        finally:
            # Store the unconsumed tasks and terminate the workers if necessary
            _remaining_outputs = ([] if self._exception else self._jobs)
            self._jobs = collections.deque()
            self._running = False
            if not detach_generator_exit:
                self._terminate_and_reset()

        while len(_remaining_outputs) > 0:
            batched_results = _remaining_outputs.popleft()
            batched_results = batched_results.get_result(self.timeout)
            for result in batched_results:
                yield result

    def _wait_retrieval(self):
        """Return True if we need to continue retriving some tasks."""

        # If the input load is still being iterated over, it means that tasks
        # are still on the dispatch wait list and their results will need to
        # be retrieved later on.
        if self._iterating:
            return True

        # If some of the dispatched tasks are still being processed by the
        # workers, wait for the compute to finish before starting retrieval
        if self.n_completed_tasks < self.n_dispatched_tasks:
            return True

        # For backends that does not support retrieving asynchronously the
        # result to the main process, all results must be carefully retrieved
        # in the _retrieve loop in the main thread while the backend is alive.
        # For other backends, the actual retrieval is done asynchronously in
        # the callback thread, and we can terminate the backend before the
        # `self._jobs` result list has been emptied. The remaining results
        # will be collected in the `finally` step of the generator.
        if not self._backend.supports_retrieve_callback:
            if len(self._jobs) > 0:
                return True

        return False

    def _retrieve(self):
        while self._wait_retrieval():

            # If the callback thread of a worker has signaled that its task
            # triggered an exception, or if the retrieval loop has raised an
            # exception (e.g. `GeneratorExit`), exit the loop and surface the
            # worker traceback.
            if self._aborting:
                self._raise_error_fast()
                break

            # If the next job is not ready for retrieval yet, we just wait for
            # async callbacks to progress.
            if ((len(self._jobs) == 0) or
                (self._jobs[0].get_status(
                    timeout=self.timeout) == TASK_PENDING)):
                time.sleep(0.01)
                continue

            # We need to be careful: the job list can be filling up as
            # we empty it and Python list are not thread-safe by
            # default hence the use of the lock
            with self._lock:
                batched_results = self._jobs.popleft()

            # Flatten the batched results to output one output at a time
            batched_results = batched_results.get_result(self.timeout)
            for result in batched_results:
                self._nb_consumed += 1
                yield result

    def _raise_error_fast(self):
        """If we are aborting, raise if a job caused an error."""

        # Find the first job whose status is TASK_ERROR if it exists.
        with self._lock:
            error_job = next((job for job in self._jobs
                              if job.status == TASK_ERROR), None)

        # If this error job exists, immediatly raise the error by
        # calling get_result. This job might not exists if abort has been
        # called directly or if the generator is gc'ed.
        if error_job is not None:
            error_job.get_result(self.timeout)

    def _warn_exit_early(self):
        """Warn the user if the generator is gc'ed before being consumned."""
        ready_outputs = self.n_completed_tasks - self._nb_consumed
        is_completed = self._is_completed()
        msg = ""
        if ready_outputs:
            msg += (
                f"{ready_outputs} tasks have been successfully executed "
                " but not used."
            )
            if not is_completed:
                msg += " Additionally, "

        if not is_completed:
            msg += (
                f"{self.n_dispatched_tasks - self.n_completed_tasks} tasks "
                "which were still being processed by the workers have been "
                "cancelled."
            )

        if msg:
            msg += (
                " You could benefit from adjusting the input task "
                "iterator to limit unnecessary computation time."
            )

            warnings.warn(msg)

    def _get_sequential_output(self, iterable):
        """Separate loop for sequential output.

        This simplifies the traceback in case of errors and reduces the
        overhead of calling sequential tasks with `joblib`.
        """
        try:
            self._iterating = True
            self._original_iterator = iterable
            batch_size = self._get_batch_size()

            if batch_size != 1:
                it = iter(iterable)
                iterable_batched = iter(
                    lambda: tuple(itertools.islice(it, batch_size)), ()
                )
                iterable = (
                    task for batch in iterable_batched for task in batch
                )

            # first yield returns None, for internal use only. This ensures
            # that we enter the try/except block and setup the generator.
            yield None

            # Sequentially call the tasks and yield the results.
            for func, args, kwargs in iterable:
                self.n_dispatched_batches += 1
                self.n_dispatched_tasks += 1
                res = func(*args, **kwargs)
                self.n_completed_tasks += 1
                self.print_progress()
                yield res
                self._nb_consumed += 1
        except BaseException:
            self._exception = True
            self._aborting = True
            self._aborted = True
            raise
        finally:
            self.print_progress()
            self._running = False
            self._iterating = False
            self._original_iterator = None

    def _reset_run_tracking(self):
        """Reset the counters and flags used to track the execution."""

        # Makes sur the parallel instance was not previously running in a
        # thread-safe way.
        with getattr(self, '_lock', nullcontext()):
            if self._running:
                msg = 'This Parallel instance is already running !'
                if self.return_generator is True:
                    msg += (
                        " Before submitting new tasks, you must wait for the "
                        "completion of all the previous tasks, or clean all "
                        "references to the output generator."
                    )
                raise RuntimeError(msg)
            self._running = True

        # Counter to keep track of the task dispatched and completed.
        self.n_dispatched_batches = 0
        self.n_dispatched_tasks = 0
        self.n_completed_tasks = 0

        # Following count is incremented by one each time the user iterates
        # on the output generator, it is used to prepare an informative
        # warning message in case the generator is deleted before all the
        # dispatched tasks have been consumed.
        self._nb_consumed = 0

        # Following flags are used to synchronize the threads in case one of
        # the tasks error-out to ensure that all workers abort fast and that
        # the backend terminates properly.

        # Set to True as soon as a worker signals that a task errors-out
        self._exception = False
        # Set to True in case of early termination following an incident
        self._aborting = False
        # Set to True after abortion is complete
        self._aborted = False

    def __call__(self, iterable):
        """Main function to dispatch parallel tasks."""

        self._reset_run_tracking()
        self._start_time = time.time()

        if not self._managed_backend:
            n_jobs = self._initialize_backend()
        else:
            n_jobs = self._effective_n_jobs()

        if n_jobs == 1:
            # If n_jobs==1, run the computation sequentially and return
            # immediatly to avoid overheads.
            output = self._get_sequential_output(iterable)
            next(output)
            return output if self.return_generator else list(output)

        # Let's create an ID that uniquely identifies the current call. If the
        # call is interrupted early and that the same instance is immediately
        # re-used, this id will be used to prevent workers that were
        # concurrently finalizing a task from the previous call to run the
        # callback.
        with self._lock:
            self._call_id = uuid4().hex

        # self._effective_n_jobs should be called in the Parallel.__call__
        # thread only -- store its value in an attribute for further queries.
        self._cached_effective_n_jobs = n_jobs

        if isinstance(self._backend, LokyBackend):
            # For the loky backend, we add a callback executed when reducing
            # BatchCalls, that makes the loky executor use a temporary folder
            # specific to this Parallel object when pickling temporary memmaps.
            # This callback is necessary to ensure that several Parallel
            # objects using the same resuable executor don't use the same
            # temporary resources.

            def _batched_calls_reducer_callback():
                # Relevant implementation detail: the following lines, called
                # when reducing BatchedCalls, are called in a thread-safe
                # situation, meaning that the context of the temporary folder
                # manager will not be changed in between the callback execution
                # and the end of the BatchedCalls pickling. The reason is that
                # pickling (the only place where set_current_context is used)
                # is done from a single thread (the queue_feeder_thread).
                self._backend._workers._temp_folder_manager.set_current_context(  # noqa
                    self._id
                )
            self._reducer_callback = _batched_calls_reducer_callback

        # self._effective_n_jobs should be called in the Parallel.__call__
        # thread only -- store its value in an attribute for further queries.
        self._cached_effective_n_jobs = n_jobs

        backend_name = self._backend.__class__.__name__
        if n_jobs == 0:
            raise RuntimeError("%s has no active worker." % backend_name)

        self._print(
            f"Using backend {backend_name} with {n_jobs} concurrent workers."
        )
        if hasattr(self._backend, 'start_call'):
            self._backend.start_call()

        # Following flag prevents double calls to `backend.stop_call`.
        self._calling = True

        iterator = iter(iterable)
        pre_dispatch = self.pre_dispatch

        if pre_dispatch == 'all':
            # prevent further dispatch via multiprocessing callback thread
            self._original_iterator = None
            self._pre_dispatch_amount = 0
        else:
            self._original_iterator = iterator
            if hasattr(pre_dispatch, 'endswith'):
                pre_dispatch = eval_expr(
                    pre_dispatch.replace("n_jobs", str(n_jobs))
                )
            self._pre_dispatch_amount = pre_dispatch = int(pre_dispatch)

            # The main thread will consume the first pre_dispatch items and
            # the remaining items will later be lazily dispatched by async
            # callbacks upon task completions.

            # TODO: this iterator should be batch_size * n_jobs
            iterator = itertools.islice(iterator, self._pre_dispatch_amount)

        # Use a caching dict for callables that are pickled with cloudpickle to
        # improve performances. This cache is used only in the case of
        # functions that are defined in the __main__ module, functions that
        # are defined locally (inside another function) and lambda expressions.
        self._pickle_cache = dict()

        output = self._get_outputs(iterator, pre_dispatch)
        self._call_ref = weakref.ref(output)

        # The first item from the output is blank, but it makes the interpreter
        # progress until it enters the Try/Except block of the generator and
        # reach the first `yield` statement. This starts the aynchronous
        # dispatch of the tasks to the workers.
        next(output)

        return output if self.return_generator else list(output)

    def __repr__(self):
        return '%s(n_jobs=%s)' % (self.__class__.__name__, self.n_jobs)
