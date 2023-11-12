from functools import wraps

import numpy as np

import statsmodels.tools.validation.validation as v


def array_like(
    pos,
    name,
    dtype=np.double,
    ndim=None,
    maxdim=None,
    shape=None,
    order="C",
    contiguous=False,
):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if pos < len(args):
                arg = args[pos]
                arg = v.array_like(
                    arg, name, dtype, ndim, maxdim, shape, order, contiguous
                )
                if pos == 0:
                    args = (arg,) + args[1:]
                else:
                    args = args[:pos] + (arg,) + args[pos + 1:]
            else:
                arg = kwargs[name]
                arg = v.array_like(
                    arg, name, dtype, ndim, maxdim, shape, order, contiguous
                )
                kwargs[name] = arg

            return func(*args, **kwargs)

        return wrapper

    return inner
