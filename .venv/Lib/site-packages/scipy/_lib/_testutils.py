"""
Generic test utilities.

"""

import os
import re
import sys
import numpy as np
import inspect


__all__ = ['PytestTester', 'check_free_memory', '_TestPythranFunc']


class FPUModeChangeWarning(RuntimeWarning):
    """Warning about FPU mode change"""
    pass


class PytestTester:
    """
    Run tests for this namespace

    ``scipy.test()`` runs tests for all of SciPy, with the default settings.
    When used from a submodule (e.g., ``scipy.cluster.test()``, only the tests
    for that namespace are run.

    Parameters
    ----------
    label : {'fast', 'full'}, optional
        Whether to run only the fast tests, or also those marked as slow.
        Default is 'fast'.
    verbose : int, optional
        Test output verbosity. Default is 1.
    extra_argv : list, optional
        Arguments to pass through to Pytest.
    doctests : bool, optional
        Whether to run doctests or not. Default is False.
    coverage : bool, optional
        Whether to run tests with code coverage measurements enabled.
        Default is False.
    tests : list of str, optional
        List of module names to run tests for. By default, uses the module
        from which the ``test`` function is called.
    parallel : int, optional
        Run tests in parallel with pytest-xdist, if number given is larger than
        1. Default is 1.

    """
    def __init__(self, module_name):
        self.module_name = module_name

    def __call__(self, label="fast", verbose=1, extra_argv=None, doctests=False,
                 coverage=False, tests=None, parallel=None):
        import pytest

        module = sys.modules[self.module_name]
        module_path = os.path.abspath(module.__path__[0])

        pytest_args = ['--showlocals', '--tb=short']

        if doctests:
            raise ValueError("Doctests not supported")

        if extra_argv:
            pytest_args += list(extra_argv)

        if verbose and int(verbose) > 1:
            pytest_args += ["-" + "v"*(int(verbose)-1)]

        if coverage:
            pytest_args += ["--cov=" + module_path]

        if label == "fast":
            pytest_args += ["-m", "not slow"]
        elif label != "full":
            pytest_args += ["-m", label]

        if tests is None:
            tests = [self.module_name]

        if parallel is not None and parallel > 1:
            if _pytest_has_xdist():
                pytest_args += ['-n', str(parallel)]
            else:
                import warnings
                warnings.warn('Could not run tests in parallel because '
                              'pytest-xdist plugin is not available.')

        pytest_args += ['--pyargs'] + list(tests)

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return (code == 0)


class _TestPythranFunc:
    '''
    These are situations that can be tested in our pythran tests:
    - A function with multiple array arguments and then
      other positional and keyword arguments.
    - A function with array-like keywords (e.g. `def somefunc(x0, x1=None)`.
    Note: list/tuple input is not yet tested!

    `self.arguments`: A dictionary which key is the index of the argument,
                      value is tuple(array value, all supported dtypes)
    `self.partialfunc`: A function used to freeze some non-array argument
                        that of no interests in the original function
    '''
    ALL_INTEGER = [np.int8, np.int16, np.int32, np.int64, np.intc, np.intp]
    ALL_FLOAT = [np.float32, np.float64]
    ALL_COMPLEX = [np.complex64, np.complex128]

    def setup_method(self):
        self.arguments = {}
        self.partialfunc = None
        self.expected = None

    def get_optional_args(self, func):
        # get optional arguments with its default value,
        # used for testing keywords
        signature = inspect.signature(func)
        optional_args = {}
        for k, v in signature.parameters.items():
            if v.default is not inspect.Parameter.empty:
                optional_args[k] = v.default
        return optional_args

    def get_max_dtype_list_length(self):
        # get the max supported dtypes list length in all arguments
        max_len = 0
        for arg_idx in self.arguments:
            cur_len = len(self.arguments[arg_idx][1])
            if cur_len > max_len:
                max_len = cur_len
        return max_len

    def get_dtype(self, dtype_list, dtype_idx):
        # get the dtype from dtype_list via index
        # if the index is out of range, then return the last dtype
        if dtype_idx > len(dtype_list)-1:
            return dtype_list[-1]
        else:
            return dtype_list[dtype_idx]

    def test_all_dtypes(self):
        for type_idx in range(self.get_max_dtype_list_length()):
            args_array = []
            for arg_idx in self.arguments:
                new_dtype = self.get_dtype(self.arguments[arg_idx][1],
                                           type_idx)
                args_array.append(self.arguments[arg_idx][0].astype(new_dtype))
            self.pythranfunc(*args_array)

    def test_views(self):
        args_array = []
        for arg_idx in self.arguments:
            args_array.append(self.arguments[arg_idx][0][::-1][::-1])
        self.pythranfunc(*args_array)

    def test_strided(self):
        args_array = []
        for arg_idx in self.arguments:
            args_array.append(np.repeat(self.arguments[arg_idx][0],
                                        2, axis=0)[::2])
        self.pythranfunc(*args_array)


def _pytest_has_xdist():
    """
    Check if the pytest-xdist plugin is installed, providing parallel tests
    """
    # Check xdist exists without importing, otherwise pytests emits warnings
    from importlib.util import find_spec
    return find_spec('xdist') is not None


def check_free_memory(free_mb):
    """
    Check *free_mb* of memory is available, otherwise do pytest.skip
    """
    import pytest

    try:
        mem_free = _parse_size(os.environ['SCIPY_AVAILABLE_MEM'])
        msg = '{} MB memory required, but environment SCIPY_AVAILABLE_MEM={}'.format(
            free_mb, os.environ['SCIPY_AVAILABLE_MEM'])
    except KeyError:
        mem_free = _get_mem_available()
        if mem_free is None:
            pytest.skip("Could not determine available memory; set SCIPY_AVAILABLE_MEM "
                        "variable to free memory in MB to run the test.")
        msg = '{} MB memory required, but {} MB available'.format(
            free_mb, mem_free/1e6)

    if mem_free < free_mb * 1e6:
        pytest.skip(msg)


def _parse_size(size_str):
    suffixes = {'': 1e6,
                'b': 1.0,
                'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
                'kb': 1e3, 'Mb': 1e6, 'Gb': 1e9, 'Tb': 1e12,
                'kib': 1024.0, 'Mib': 1024.0**2, 'Gib': 1024.0**3, 'Tib': 1024.0**4}
    m = re.match(r'^\s*(\d+)\s*({})\s*$'.format('|'.join(suffixes.keys())),
                 size_str,
                 re.I)
    if not m or m.group(2) not in suffixes:
        raise ValueError("Invalid size string")

    return float(m.group(1)) * suffixes[m.group(2)]


def _get_mem_available():
    """
    Get information about memory available, not counting swap.
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass

    if sys.platform.startswith('linux'):
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                p = line.split()
                info[p[0].strip(':').lower()] = float(p[1]) * 1e3

        if 'memavailable' in info:
            # Linux >= 3.14
            return info['memavailable']
        else:
            return info['memfree'] + info['cached']

    return None
