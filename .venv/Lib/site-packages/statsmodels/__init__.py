from statsmodels.compat.patsy import monkey_patch_cat_dtype

from statsmodels._version import __version__, __version_tuple__

__version_info__ = __version_tuple__

monkey_patch_cat_dtype()

debug_warnings = False

if debug_warnings:
    import warnings

    warnings.simplefilter("default")
    # use the following to raise an exception for debugging specific warnings
    # warnings.filterwarnings("error", message=".*integer.*")


def test(extra_args=None, exit=False):
    """
    Run the test suite

    Parameters
    ----------
    extra_args : list[str]
        List of argument to pass to pytest when running the test suite. The
        default is ['--tb=short', '--disable-pytest-warnings'].
    exit : bool
        Flag indicating whether the test runner should exist when finished.

    Returns
    -------
    int
        The status code from the test run if exit is False.
    """
    from .tools._testing import PytestTester

    tst = PytestTester(package_path=__file__)
    return tst(extra_args=extra_args, exit=exit)


__all__ = ["__version__", "__version_info__", "__version_tuple__", "test"]
