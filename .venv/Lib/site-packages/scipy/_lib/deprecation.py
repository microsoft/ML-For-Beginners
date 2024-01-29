from inspect import Parameter, signature
import functools
import warnings
from importlib import import_module


__all__ = ["_deprecated"]


# Object to use as default value for arguments to be deprecated. This should
# be used over 'None' as the user could parse 'None' as a positional argument
_NoValue = object()

def _sub_module_deprecation(*, sub_package, module, private_modules, all,
                            attribute, correct_module=None):
    """Helper function for deprecating modules that are public but were
    intended to be private.

    Parameters
    ----------
    sub_package : str
        Subpackage the module belongs to eg. stats
    module : str
        Public but intended private module to deprecate
    private_modules : list
        Private replacement(s) for `module`; should contain the
        content of ``all``, possibly spread over several modules.
    all : list
        ``__all__`` belonging to `module`
    attribute : str
        The attribute in `module` being accessed
    correct_module : str, optional
        Module in `sub_package` that `attribute` should be imported from.
        Default is that `attribute` should be imported from ``scipy.sub_package``.
    """
    if correct_module is not None:
        correct_import = f"scipy.{sub_package}.{correct_module}"
    else:
        correct_import = f"scipy.{sub_package}"

    if attribute not in all:
        raise AttributeError(
            f"`scipy.{sub_package}.{module}` has no attribute `{attribute}`; "
            f"furthermore, `scipy.{sub_package}.{module}` is deprecated "
            f"and will be removed in SciPy 2.0.0."
        )

    attr = getattr(import_module(correct_import), attribute, None)

    if attr is not None:
        message = (
            f"Please import `{attribute}` from the `{correct_import}` namespace; "
            f"the `scipy.{sub_package}.{module}` namespace is deprecated "
            f"and will be removed in SciPy 2.0.0."
        )
    else:
        message = (
            f"`scipy.{sub_package}.{module}.{attribute}` is deprecated along with "
            f"the `scipy.{sub_package}.{module}` namespace. "
            f"`scipy.{sub_package}.{module}.{attribute}` will be removed "
            f"in SciPy 1.14.0, and the `scipy.{sub_package}.{module}` namespace "
            f"will be removed in SciPy 2.0.0."
        )

    warnings.warn(message, category=DeprecationWarning, stacklevel=3)

    for module in private_modules:
        try:
            return getattr(import_module(f"scipy.{sub_package}.{module}"), attribute)
        except AttributeError as e:
            # still raise an error if the attribute isn't in any of the expected
            # private modules
            if module == private_modules[-1]:
                raise e
            continue
    

def _deprecated(msg, stacklevel=2):
    """Deprecate a function by emitting a warning on use."""
    def wrap(fun):
        if isinstance(fun, type):
            warnings.warn(
                f"Trying to deprecate class {fun!r}",
                category=RuntimeWarning, stacklevel=2)
            return fun

        @functools.wraps(fun)
        def call(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning,
                          stacklevel=stacklevel)
            return fun(*args, **kwargs)
        call.__doc__ = fun.__doc__
        return call

    return wrap


class _DeprecationHelperStr:
    """
    Helper class used by deprecate_cython_api
    """
    def __init__(self, content, message):
        self._content = content
        self._message = message

    def __hash__(self):
        return hash(self._content)

    def __eq__(self, other):
        res = (self._content == other)
        if res:
            warnings.warn(self._message, category=DeprecationWarning,
                          stacklevel=2)
        return res


def deprecate_cython_api(module, routine_name, new_name=None, message=None):
    """
    Deprecate an exported cdef function in a public Cython API module.

    Only functions can be deprecated; typedefs etc. cannot.

    Parameters
    ----------
    module : module
        Public Cython API module (e.g. scipy.linalg.cython_blas).
    routine_name : str
        Name of the routine to deprecate. May also be a fused-type
        routine (in which case its all specializations are deprecated).
    new_name : str
        New name to include in the deprecation warning message
    message : str
        Additional text in the deprecation warning message

    Examples
    --------
    Usually, this function would be used in the top-level of the
    module ``.pyx`` file:

    >>> from scipy._lib.deprecation import deprecate_cython_api
    >>> import scipy.linalg.cython_blas as mod
    >>> deprecate_cython_api(mod, "dgemm", "dgemm_new",
    ...                      message="Deprecated in Scipy 1.5.0")
    >>> del deprecate_cython_api, mod

    After this, Cython modules that use the deprecated function emit a
    deprecation warning when they are imported.

    """
    old_name = f"{module.__name__}.{routine_name}"

    if new_name is None:
        depdoc = "`%s` is deprecated!" % old_name
    else:
        depdoc = f"`{old_name}` is deprecated, use `{new_name}` instead!"

    if message is not None:
        depdoc += "\n" + message

    d = module.__pyx_capi__

    # Check if the function is a fused-type function with a mangled name
    j = 0
    has_fused = False
    while True:
        fused_name = f"__pyx_fuse_{j}{routine_name}"
        if fused_name in d:
            has_fused = True
            d[_DeprecationHelperStr(fused_name, depdoc)] = d.pop(fused_name)
            j += 1
        else:
            break

    # If not, apply deprecation to the named routine
    if not has_fused:
        d[_DeprecationHelperStr(routine_name, depdoc)] = d.pop(routine_name)


# taken from scikit-learn, see
# https://github.com/scikit-learn/scikit-learn/blob/1.3.0/sklearn/utils/validation.py#L38
def _deprecate_positional_args(func=None, *, version=None):
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
    version : callable, default=None
        The version when positional arguments will result in error.
    """
    if version is None:
        msg = "Need to specify a version where signature will be changed"
        raise ValueError(msg)

    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @functools.wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0
            args_msg = [
                f"{name}={arg}"
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            args_msg = ", ".join(args_msg)
            warnings.warn(
                (
                    f"You are passing {args_msg} as a positional argument. "
                    "Please change your invocation to use keyword arguments. "
                    f"From SciPy {version}, passing these as positional "
                    "arguments will result in an error."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)

    return _inner_deprecate_positional_args
