import builtins
import platform
import sys
from contextlib import suppress
from functools import wraps
from os import environ
from unittest import SkipTest

import joblib
import numpy as np
import pytest
from _pytest.doctest import DoctestItem
from threadpoolctl import threadpool_limits

from sklearn import config_context, set_config
from sklearn._min_dependencies import PYTEST_MIN_VERSION
from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_20newsgroups_vectorized,
    fetch_california_housing,
    fetch_covtype,
    fetch_kddcup99,
    fetch_olivetti_faces,
    fetch_rcv1,
)
from sklearn.tests import random_seed
from sklearn.utils import _IS_32BIT
from sklearn.utils.fixes import np_base_version, parse_version, sp_version

if parse_version(pytest.__version__) < parse_version(PYTEST_MIN_VERSION):
    raise ImportError(
        f"Your version of pytest is too old. Got version {pytest.__version__}, you"
        f" should have pytest >= {PYTEST_MIN_VERSION} installed."
    )

scipy_datasets_require_network = sp_version >= parse_version("1.10")


@pytest.fixture
def enable_slep006():
    """Enable SLEP006 for all tests."""
    with config_context(enable_metadata_routing=True):
        yield


def raccoon_face_or_skip():
    # SciPy >= 1.10 requires network to access to get data
    if scipy_datasets_require_network:
        run_network_tests = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"
        if not run_network_tests:
            raise SkipTest("test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0")

        try:
            import pooch  # noqa
        except ImportError:
            raise SkipTest("test requires pooch to be installed")

        from scipy.datasets import face
    else:
        from scipy.misc import face

    return face(gray=True)


dataset_fetchers = {
    "fetch_20newsgroups_fxt": fetch_20newsgroups,
    "fetch_20newsgroups_vectorized_fxt": fetch_20newsgroups_vectorized,
    "fetch_california_housing_fxt": fetch_california_housing,
    "fetch_covtype_fxt": fetch_covtype,
    "fetch_kddcup99_fxt": fetch_kddcup99,
    "fetch_olivetti_faces_fxt": fetch_olivetti_faces,
    "fetch_rcv1_fxt": fetch_rcv1,
}

if scipy_datasets_require_network:
    dataset_fetchers["raccoon_face_fxt"] = raccoon_face_or_skip

_SKIP32_MARK = pytest.mark.skipif(
    environ.get("SKLEARN_RUN_FLOAT32_TESTS", "0") != "1",
    reason="Set SKLEARN_RUN_FLOAT32_TESTS=1 to run float32 dtype tests",
)


# Global fixtures
@pytest.fixture(params=[pytest.param(np.float32, marks=_SKIP32_MARK), np.float64])
def global_dtype(request):
    yield request.param


def _fetch_fixture(f):
    """Fetch dataset (download if missing and requested by environment)."""
    download_if_missing = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"

    @wraps(f)
    def wrapped(*args, **kwargs):
        kwargs["download_if_missing"] = download_if_missing
        try:
            return f(*args, **kwargs)
        except OSError as e:
            if str(e) != "Data not found and `download_if_missing` is False":
                raise
            pytest.skip("test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0")

    return pytest.fixture(lambda: wrapped)


# Adds fixtures for fetching data
fetch_20newsgroups_fxt = _fetch_fixture(fetch_20newsgroups)
fetch_20newsgroups_vectorized_fxt = _fetch_fixture(fetch_20newsgroups_vectorized)
fetch_california_housing_fxt = _fetch_fixture(fetch_california_housing)
fetch_covtype_fxt = _fetch_fixture(fetch_covtype)
fetch_kddcup99_fxt = _fetch_fixture(fetch_kddcup99)
fetch_olivetti_faces_fxt = _fetch_fixture(fetch_olivetti_faces)
fetch_rcv1_fxt = _fetch_fixture(fetch_rcv1)
raccoon_face_fxt = pytest.fixture(raccoon_face_or_skip)


def pytest_collection_modifyitems(config, items):
    """Called after collect is completed.

    Parameters
    ----------
    config : pytest config
    items : list of collected items
    """
    run_network_tests = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"
    skip_network = pytest.mark.skip(
        reason="test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0"
    )

    # download datasets during collection to avoid thread unsafe behavior
    # when running pytest in parallel with pytest-xdist
    dataset_features_set = set(dataset_fetchers)
    datasets_to_download = set()

    for item in items:
        if isinstance(item, DoctestItem) and "fetch_" in item.name:
            fetcher_function_name = item.name.split(".")[-1]
            dataset_fetchers_key = f"{fetcher_function_name}_fxt"
            dataset_to_fetch = set([dataset_fetchers_key]) & dataset_features_set
        elif not hasattr(item, "fixturenames"):
            continue
        else:
            item_fixtures = set(item.fixturenames)
            dataset_to_fetch = item_fixtures & dataset_features_set

        if not dataset_to_fetch:
            continue

        if run_network_tests:
            datasets_to_download |= dataset_to_fetch
        else:
            # network tests are skipped
            item.add_marker(skip_network)

    # Only download datasets on the first worker spawned by pytest-xdist
    # to avoid thread unsafe behavior. If pytest-xdist is not used, we still
    # download before tests run.
    worker_id = environ.get("PYTEST_XDIST_WORKER", "gw0")
    if worker_id == "gw0" and run_network_tests:
        for name in datasets_to_download:
            with suppress(SkipTest):
                dataset_fetchers[name]()

    for item in items:
        # Known failure on with GradientBoostingClassifier on ARM64
        if (
            item.name.endswith("GradientBoostingClassifier")
            and platform.machine() == "aarch64"
        ):
            marker = pytest.mark.xfail(
                reason=(
                    "know failure. See "
                    "https://github.com/scikit-learn/scikit-learn/issues/17797"  # noqa
                )
            )
            item.add_marker(marker)

    skip_doctests = False
    try:
        import matplotlib  # noqa
    except ImportError:
        skip_doctests = True
        reason = "matplotlib is required to run the doctests"

    if _IS_32BIT:
        reason = "doctest are only run when the default numpy int is 64 bits."
        skip_doctests = True
    elif sys.platform.startswith("win32"):
        reason = (
            "doctests are not run for Windows because numpy arrays "
            "repr is inconsistent across platforms."
        )
        skip_doctests = True

    if np_base_version >= parse_version("2"):
        reason = "Due to NEP 51 numpy scalar repr has changed in numpy 2"
        skip_doctests = True

    # Normally doctest has the entire module's scope. Here we set globs to an empty dict
    # to remove the module's scope:
    # https://docs.python.org/3/library/doctest.html#what-s-the-execution-context
    for item in items:
        if isinstance(item, DoctestItem):
            item.dtest.globs = {}

    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)

        for item in items:
            if isinstance(item, DoctestItem):
                # work-around an internal error with pytest if adding a skip
                # mark to a doctest in a contextmanager, see
                # https://github.com/pytest-dev/pytest/issues/8796 for more
                # details.
                if item.name != "sklearn._config.config_context":
                    item.add_marker(skip_marker)
    try:
        import PIL  # noqa

        pillow_installed = True
    except ImportError:
        pillow_installed = False

    if not pillow_installed:
        skip_marker = pytest.mark.skip(reason="pillow (or PIL) not installed!")
        for item in items:
            if item.name in [
                "sklearn.feature_extraction.image.PatchExtractor",
                "sklearn.feature_extraction.image.extract_patches_2d",
            ]:
                item.add_marker(skip_marker)


@pytest.fixture(scope="function")
def pyplot():
    """Setup and teardown fixture for matplotlib.

    This fixture checks if we can import matplotlib. If not, the tests will be
    skipped. Otherwise, we close the figures before and after running the
    functions.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pyplot.close("all")
    yield pyplot
    pyplot.close("all")


def pytest_configure(config):
    # Use matplotlib agg backend during the tests including doctests
    try:
        import matplotlib

        matplotlib.use("agg")
    except ImportError:
        pass

    allowed_parallelism = joblib.cpu_count(only_physical_cores=True)
    xdist_worker_count = environ.get("PYTEST_XDIST_WORKER_COUNT")
    if xdist_worker_count is not None:
        # Set the number of OpenMP and BLAS threads based on the number of workers
        # xdist is using to prevent oversubscription.
        allowed_parallelism = max(allowed_parallelism // int(xdist_worker_count), 1)
    threadpool_limits(allowed_parallelism)

    # Register global_random_seed plugin if it is not already registered
    if not config.pluginmanager.hasplugin("sklearn.tests.random_seed"):
        config.pluginmanager.register(random_seed)


@pytest.fixture
def hide_available_pandas(monkeypatch):
    """Pretend pandas was not installed."""
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


@pytest.fixture
def print_changed_only_false():
    """Set `print_changed_only` to False for the duration of the test."""
    set_config(print_changed_only=False)
    yield
    set_config(print_changed_only=True)  # reset to default
