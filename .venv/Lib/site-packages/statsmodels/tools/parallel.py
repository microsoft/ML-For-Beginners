"""Parallel utility function using joblib

copied from https://github.com/mne-tools/mne-python

Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
License: Simplified BSD

changes for statsmodels (Josef Perktold)
- try import from joblib directly, (does not import all of sklearn)

"""

from statsmodels.tools.sm_exceptions import (ModuleUnavailableWarning,
                                             module_unavailable_doc)


def parallel_func(func, n_jobs, verbose=5):
    """Return parallel instance with delayed function

    Util function to use joblib only if available

    Parameters
    ----------
    func : callable
        A function
    n_jobs : int
        Number of jobs to run in parallel
    verbose : int
        Verbosity level

    Returns
    -------
    parallel : instance of joblib.Parallel or list
        The parallel object
    my_func : callable
        func if not parallel or delayed(func)
    n_jobs : int
        Number of jobs >= 0

    Examples
    --------
    >>> from math import sqrt
    >>> from statsmodels.tools.parallel import parallel_func
    >>> parallel, p_func, n_jobs = parallel_func(sqrt, n_jobs=-1, verbose=0)
    >>> print(n_jobs)
    >>> parallel(p_func(i**2) for i in range(10))
    """
    try:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            from sklearn.externals.joblib import Parallel, delayed

        parallel = Parallel(n_jobs, verbose=verbose)
        my_func = delayed(func)

        if n_jobs == -1:
            try:
                import multiprocessing
                n_jobs = multiprocessing.cpu_count()
            except (ImportError, NotImplementedError):
                import warnings
                warnings.warn(module_unavailable_doc.format('multiprocessing'),
                              ModuleUnavailableWarning)
                n_jobs = 1

    except ImportError:
        import warnings
        warnings.warn(module_unavailable_doc.format('joblib'),
                      ModuleUnavailableWarning)
        n_jobs = 1
        my_func = func
        parallel = list
    return parallel, my_func, n_jobs
