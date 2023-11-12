import numpy as np


def add_indep(x, varnames, dtype=None):
    '''
    construct array with independent columns

    x is either iterable (list, tuple) or instance of ndarray or a subclass
    of it.  If x is an ndarray, then each column is assumed to represent a
    variable with observations in rows.
    '''
    # TODO: this needs tests for subclasses

    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = x.T

    nvars_orig = len(x)
    nobs = len(x[0])
    if not dtype:
        dtype = np.asarray(x[0]).dtype
    xout = np.zeros((nobs, nvars_orig), dtype=dtype)
    count = 0
    rank_old = 0
    varnames_new = []
    varnames_dropped = []
    keepindx = []
    for (xi, ni) in zip(x, varnames):
        xout[:, count] = xi
        rank_new = np.linalg.matrix_rank(xout)
        if rank_new > rank_old:
            varnames_new.append(ni)
            rank_old = rank_new
            count += 1
        else:
            varnames_dropped.append(ni)

    return xout[:, :count], varnames_new
