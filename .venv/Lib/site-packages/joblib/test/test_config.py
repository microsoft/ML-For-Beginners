import os

from joblib.parallel import parallel_config
from joblib.parallel import parallel_backend
from joblib.parallel import Parallel, delayed

from joblib.parallel import BACKENDS
from joblib.parallel import DEFAULT_BACKEND
from joblib.parallel import EXTERNAL_BACKENDS

from joblib._parallel_backends import LokyBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend

from joblib.testing import parametrize, raises
from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.test.test_parallel import check_memmap


@parametrize("context", [parallel_config, parallel_backend])
def test_global_parallel_backend(context):
    default = Parallel()._backend

    pb = context('threading')
    try:
        assert isinstance(Parallel()._backend, ThreadingBackend)
    finally:
        pb.unregister()
    assert type(Parallel()._backend) is type(default)


@parametrize("context", [parallel_config, parallel_backend])
def test_external_backends(context):
    def register_foo():
        BACKENDS['foo'] = ThreadingBackend

    EXTERNAL_BACKENDS['foo'] = register_foo
    try:
        with context('foo'):
            assert isinstance(Parallel()._backend, ThreadingBackend)
    finally:
        del EXTERNAL_BACKENDS['foo']


@with_numpy
@with_multiprocessing
def test_parallel_config_no_backend(tmpdir):
    # Check that parallel_config allows to change the config
    # even if no backend is set.
    with parallel_config(n_jobs=2, max_nbytes=1, temp_folder=tmpdir):
        with Parallel(prefer="processes") as p:
            assert isinstance(p._backend, LokyBackend)
            assert p.n_jobs == 2

            # Checks that memmapping is enabled
            p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)
            assert len(os.listdir(tmpdir)) > 0


@with_numpy
@with_multiprocessing
def test_parallel_config_params_explicit_set(tmpdir):
    with parallel_config(n_jobs=3, max_nbytes=1, temp_folder=tmpdir):
        with Parallel(n_jobs=2, prefer="processes", max_nbytes='1M') as p:
            assert isinstance(p._backend, LokyBackend)
            assert p.n_jobs == 2

            # Checks that memmapping is disabled
            with raises(TypeError, match="Expected np.memmap instance"):
                p(delayed(check_memmap)(a) for a in [np.random.random(10)] * 2)


@parametrize("param", ["prefer", "require"])
def test_parallel_config_bad_params(param):
    # Check that an error is raised when setting a wrong backend
    # hint or constraint
    with raises(ValueError, match=f"{param}=wrong is not a valid"):
        with parallel_config(**{param: "wrong"}):
            Parallel()


def test_parallel_config_constructor_params():
    # Check that an error is raised when backend is None
    # but backend constructor params are given
    with raises(ValueError, match="only supported when backend is not None"):
        with parallel_config(inner_max_num_threads=1):
            pass

    with raises(ValueError, match="only supported when backend is not None"):
        with parallel_config(backend_param=1):
            pass


def test_parallel_config_nested():
    # Check that nested configuration retrieves the info from the
    # parent config and do not reset them.

    with parallel_config(n_jobs=2):
        p = Parallel()
        assert isinstance(p._backend, BACKENDS[DEFAULT_BACKEND])
        assert p.n_jobs == 2

    with parallel_config(backend='threading'):
        with parallel_config(n_jobs=2):
            p = Parallel()
            assert isinstance(p._backend, ThreadingBackend)
            assert p.n_jobs == 2

    with parallel_config(verbose=100):
        with parallel_config(n_jobs=2):
            p = Parallel()
            assert p.verbose == 100
            assert p.n_jobs == 2


@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'threading',
                         MultiprocessingBackend(), ThreadingBackend()])
@parametrize("context", [parallel_config, parallel_backend])
def test_threadpool_limitation_in_child_context_error(context, backend):

    with raises(AssertionError, match=r"does not acc.*inner_max_num_threads"):
        context(backend, inner_max_num_threads=1)


@parametrize("context", [parallel_config, parallel_backend])
def test_parallel_n_jobs_none(context):
    # Check that n_jobs=None is interpreted as "unset" in Parallel
    # non regression test for #1473
    with context(backend="threading", n_jobs=2):
        with Parallel(n_jobs=None) as p:
            assert p.n_jobs == 2

    with context(backend="threading"):
        default_n_jobs = Parallel().n_jobs
        with Parallel(n_jobs=None) as p:
            assert p.n_jobs == default_n_jobs


@parametrize("context", [parallel_config, parallel_backend])
def test_parallel_config_n_jobs_none(context):
    # Check that n_jobs=None is interpreted as "explicitly set" in
    # parallel_(config/backend)
    # non regression test for #1473
    with context(backend="threading", n_jobs=2):
        with context(backend="threading", n_jobs=None):
            # n_jobs=None resets n_jobs to backend's default
            with Parallel() as p:
                assert p.n_jobs == 1
