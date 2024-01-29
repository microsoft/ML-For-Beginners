import builtins
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import sklearn
from sklearn import config_context, get_config, set_config
from sklearn.utils import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed


def test_config_context():
    assert get_config() == {
        "assume_finite": False,
        "working_memory": 1024,
        "print_changed_only": True,
        "display": "diagram",
        "array_api_dispatch": False,
        "pairwise_dist_chunk_size": 256,
        "enable_cython_pairwise_dist": True,
        "transform_output": "default",
        "enable_metadata_routing": False,
        "skip_parameter_validation": False,
    }

    # Not using as a context manager affects nothing
    config_context(assume_finite=True)
    assert get_config()["assume_finite"] is False

    with config_context(assume_finite=True):
        assert get_config() == {
            "assume_finite": True,
            "working_memory": 1024,
            "print_changed_only": True,
            "display": "diagram",
            "array_api_dispatch": False,
            "pairwise_dist_chunk_size": 256,
            "enable_cython_pairwise_dist": True,
            "transform_output": "default",
            "enable_metadata_routing": False,
            "skip_parameter_validation": False,
        }
    assert get_config()["assume_finite"] is False

    with config_context(assume_finite=True):
        with config_context(assume_finite=None):
            assert get_config()["assume_finite"] is True

        assert get_config()["assume_finite"] is True

        with config_context(assume_finite=False):
            assert get_config()["assume_finite"] is False

            with config_context(assume_finite=None):
                assert get_config()["assume_finite"] is False

                # global setting will not be retained outside of context that
                # did not modify this setting
                set_config(assume_finite=True)
                assert get_config()["assume_finite"] is True

            assert get_config()["assume_finite"] is False

        assert get_config()["assume_finite"] is True

    assert get_config() == {
        "assume_finite": False,
        "working_memory": 1024,
        "print_changed_only": True,
        "display": "diagram",
        "array_api_dispatch": False,
        "pairwise_dist_chunk_size": 256,
        "enable_cython_pairwise_dist": True,
        "transform_output": "default",
        "enable_metadata_routing": False,
        "skip_parameter_validation": False,
    }

    # No positional arguments
    with pytest.raises(TypeError):
        config_context(True)

    # No unknown arguments
    with pytest.raises(TypeError):
        config_context(do_something_else=True).__enter__()


def test_config_context_exception():
    assert get_config()["assume_finite"] is False
    try:
        with config_context(assume_finite=True):
            assert get_config()["assume_finite"] is True
            raise ValueError()
    except ValueError:
        pass
    assert get_config()["assume_finite"] is False


def test_set_config():
    assert get_config()["assume_finite"] is False
    set_config(assume_finite=None)
    assert get_config()["assume_finite"] is False
    set_config(assume_finite=True)
    assert get_config()["assume_finite"] is True
    set_config(assume_finite=None)
    assert get_config()["assume_finite"] is True
    set_config(assume_finite=False)
    assert get_config()["assume_finite"] is False

    # No unknown arguments
    with pytest.raises(TypeError):
        set_config(do_something_else=True)


def set_assume_finite(assume_finite, sleep_duration):
    """Return the value of assume_finite after waiting `sleep_duration`."""
    with config_context(assume_finite=assume_finite):
        time.sleep(sleep_duration)
        return get_config()["assume_finite"]


@pytest.mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_config_threadsafe_joblib(backend):
    """Test that the global config is threadsafe with all joblib backends.
    Two jobs are spawned and sets assume_finite to two different values.
    When the job with a duration 0.1s completes, the assume_finite value
    should be the same as the value passed to the function. In other words,
    it is not influenced by the other job setting assume_finite to True.
    """
    assume_finites = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]

    items = Parallel(backend=backend, n_jobs=2)(
        delayed(set_assume_finite)(assume_finite, sleep_dur)
        for assume_finite, sleep_dur in zip(assume_finites, sleep_durations)
    )

    assert items == [False, True, False, True]


@pytest.mark.xfail(_IS_WASM, reason="cannot start threads")
def test_config_threadsafe():
    """Uses threads directly to test that the global config does not change
    between threads. Same test as `test_config_threadsafe_joblib` but with
    `ThreadPoolExecutor`."""

    assume_finites = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]

    with ThreadPoolExecutor(max_workers=2) as e:
        items = [
            output
            for output in e.map(set_assume_finite, assume_finites, sleep_durations)
        ]

    assert items == [False, True, False, True]


def test_config_array_api_dispatch_error(monkeypatch):
    """Check error is raised when array_api_compat is not installed."""

    # Hide array_api_compat import
    orig_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "array_api_compat":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)

    with pytest.raises(ImportError, match="array_api_compat is required"):
        with config_context(array_api_dispatch=True):
            pass

    with pytest.raises(ImportError, match="array_api_compat is required"):
        set_config(array_api_dispatch=True)


def test_config_array_api_dispatch_error_numpy(monkeypatch):
    """Check error when NumPy is too old"""
    # Pretend that array_api_compat is installed.
    orig_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "array_api_compat":
            return object()
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)
    monkeypatch.setattr(sklearn.utils._array_api.numpy, "__version__", "1.20")

    with pytest.raises(ImportError, match="NumPy must be 1.21 or newer"):
        with config_context(array_api_dispatch=True):
            pass

    with pytest.raises(ImportError, match="NumPy must be 1.21 or newer"):
        set_config(array_api_dispatch=True)
