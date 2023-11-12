from statsmodels.tools.sm_exceptions import CacheWriteWarning
from statsmodels.compat.pandas import cache_readonly as PandasCacheReadonly

import warnings

__all__ = ['cache_readonly', 'cache_writable', 'deprecated_alias',
           'ResettableCache']


class ResettableCache(dict):
    """DO NOT USE. BACKWARD COMPAT ONLY"""
    def __init__(self, *args, **kwargs):
        super(ResettableCache, self).__init__(*args, **kwargs)
        self.__dict__ = self


def deprecated_alias(old_name, new_name, remove_version=None, msg=None,
                     warning=FutureWarning):
    """
    Deprecate attribute in favor of alternative name.

    Parameters
    ----------
    old_name : str
        Old, deprecated name
    new_name : str
        New name
    remove_version : str, optional
        Version that the alias will be removed
    msg : str, optional
        Message to show.  Default is
        `old_name` is a deprecated alias for `new_name`
    warning : Warning, optional
        Warning class to give.  Default is FutureWarning.

    Notes
    -----
    Older or less-used classes may not conform to statsmodels naming
    conventions.  `deprecated_alias` lets us bring them into conformance
    without breaking backward-compatibility.

    Example
    -------
    Instances of the `Foo` class have a `nvars` attribute, but it _should_
    be called `neqs`:

    class Foo:
        nvars = deprecated_alias('nvars', 'neqs')
        def __init__(self, neqs):
            self.neqs = neqs

    >>> foo = Foo(3)
    >>> foo.nvars
    __main__:1: FutureWarning: nvars is a deprecated alias for neqs
    3
    """

    if msg is None:
        msg = '%s is a deprecated alias for %s' % (old_name, new_name)
        if remove_version is not None:
            msg += ', will be removed in version %s' % remove_version

    def fget(self):
        warnings.warn(msg, warning, stacklevel=2)
        return getattr(self, new_name)

    def fset(self, value):
        warnings.warn(msg, warning, stacklevel=2)
        setattr(self, new_name, value)

    res = property(fget=fget, fset=fset)
    return res


class CachedAttribute:

    def __init__(self, func, cachename=None):
        self.fget = func
        self.name = func.__name__
        self.cachename = cachename or '_cache'

    def __get__(self, obj, type=None):
        if obj is None:
            return self.fget
        # Get the cache or set a default one if needed
        _cachename = self.cachename
        _cache = getattr(obj, _cachename, None)
        if _cache is None:
            setattr(obj, _cachename, {})
            _cache = getattr(obj, _cachename)
        # Get the name of the attribute to set and cache
        name = self.name
        _cachedval = _cache.get(name, None)
        if _cachedval is None:
            _cachedval = self.fget(obj)
            _cache[name] = _cachedval

        return _cachedval

    def __set__(self, obj, value):
        errmsg = "The attribute '%s' cannot be overwritten" % self.name
        warnings.warn(errmsg, CacheWriteWarning)


class CachedWritableAttribute(CachedAttribute):
    def __set__(self, obj, value):
        _cache = getattr(obj, self.cachename)
        name = self.name
        _cache[name] = value


class _cache_readonly(property):
    """
    Decorator for CachedAttribute
    """

    def __init__(self, cachename=None):
        self.func = None
        self.cachename = cachename

    def __call__(self, func):
        return CachedAttribute(func,
                               cachename=self.cachename)


class cache_writable(_cache_readonly):
    """
    Decorator for CachedWritableAttribute
    """
    def __call__(self, func):
        return CachedWritableAttribute(func,
                                       cachename=self.cachename)


# Use pandas since it works with docs correctly
cache_readonly = PandasCacheReadonly
# cached_value and cached_data behave identically to cache_readonly, but
# are used by `remove_data` to
#   a) identify array-like attributes to remove (cached_data)
#   b) make sure certain values are evaluated before caching (cached_value)
# TODO: Disabled since the subclasses break doc strings
# class cached_data(PandasCacheReadonly):
#     pass

cached_data = PandasCacheReadonly

# class cached_value(PandasCacheReadonly):
#     pass

cached_value = PandasCacheReadonly
