from functools import wraps

from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import freq_to_period


def _get_pandas_wrapper(X, trim_head=None, trim_tail=None, names=None):
    index = X.index
    #TODO: allow use index labels
    if trim_head is None and trim_tail is None:
        index = index
    elif trim_tail is None:
        index = index[trim_head:]
    elif trim_head is None:
        index = index[:-trim_tail]
    else:
        index = index[trim_head:-trim_tail]
    if hasattr(X, "columns"):
        if names is None:
            names = X.columns
        return lambda x : X.__class__(x, index=index, columns=names)
    else:
        if names is None:
            names = X.name
        return lambda x : X.__class__(x, index=index, name=names)


def pandas_wrapper(func, trim_head=None, trim_tail=None, names=None, *args,
                   **kwargs):
    @wraps(func)
    def new_func(X, *args, **kwargs):
        # quick pass-through for do nothing case
        if not _is_using_pandas(X, None):
            return func(X, *args, **kwargs)

        wrapper_func = _get_pandas_wrapper(X, trim_head, trim_tail,
                                           names)
        ret = func(X, *args, **kwargs)
        ret = wrapper_func(ret)
        return ret

    return new_func


def pandas_wrapper_bunch(func, trim_head=None, trim_tail=None,
                         names=None, *args, **kwargs):
    @wraps(func)
    def new_func(X, *args, **kwargs):
        # quick pass-through for do nothing case
        if not _is_using_pandas(X, None):
            return func(X, *args, **kwargs)

        wrapper_func = _get_pandas_wrapper(X, trim_head, trim_tail,
                                           names)
        ret = func(X, *args, **kwargs)
        ret = wrapper_func(ret)
        return ret

    return new_func


def pandas_wrapper_predict(func, trim_head=None, trim_tail=None,
                           columns=None, *args, **kwargs):
    raise NotImplementedError


def pandas_wrapper_freq(func, trim_head=None, trim_tail=None,
                        freq_kw='freq', columns=None, *args, **kwargs):
    """
    Return a new function that catches the incoming X, checks if it's pandas,
    calls the functions as is. Then wraps the results in the incoming index.

    Deals with frequencies. Expects that the function returns a tuple,
    a Bunch object, or a pandas-object.
    """

    @wraps(func)
    def new_func(X, *args, **kwargs):
        # quick pass-through for do nothing case
        if not _is_using_pandas(X, None):
            return func(X, *args, **kwargs)

        wrapper_func = _get_pandas_wrapper(X, trim_head, trim_tail,
                                           columns)
        index = X.index
        freq = index.inferred_freq
        kwargs.update({freq_kw : freq_to_period(freq)})
        ret = func(X, *args, **kwargs)
        ret = wrapper_func(ret)
        return ret

    return new_func
