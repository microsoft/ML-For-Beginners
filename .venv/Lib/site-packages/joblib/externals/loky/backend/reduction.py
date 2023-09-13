###############################################################################
# Customizable Pickler with some basic reducers
#
# author: Thomas Moreau
#
# adapted from multiprocessing/reduction.py (17/02/2017)
#  * Replace the ForkingPickler with a similar _LokyPickler,
#  * Add CustomizableLokyPickler to allow customizing pickling process
#    on the fly.
#
import copyreg
import io
import functools
import types
import sys
import os

from multiprocessing import util
from pickle import loads, HIGHEST_PROTOCOL

###############################################################################
# Enable custom pickling in Loky.

_dispatch_table = {}


def register(type_, reduce_function):
    _dispatch_table[type_] = reduce_function


###############################################################################
# Registers extra pickling routines to improve picklization  for loky


# make methods picklable
def _reduce_method(m):
    if m.__self__ is None:
        return getattr, (m.__class__, m.__func__.__name__)
    else:
        return getattr, (m.__self__, m.__func__.__name__)


class _C:
    def f(self):
        pass

    @classmethod
    def h(cls):
        pass


register(type(_C().f), _reduce_method)
register(type(_C.h), _reduce_method)


if not hasattr(sys, "pypy_version_info"):
    # PyPy uses functions instead of method_descriptors and wrapper_descriptors
    def _reduce_method_descriptor(m):
        return getattr, (m.__objclass__, m.__name__)

    register(type(list.append), _reduce_method_descriptor)
    register(type(int.__add__), _reduce_method_descriptor)


# Make partial func pickable
def _reduce_partial(p):
    return _rebuild_partial, (p.func, p.args, p.keywords or {})


def _rebuild_partial(func, args, keywords):
    return functools.partial(func, *args, **keywords)


register(functools.partial, _reduce_partial)

if sys.platform != "win32":
    from ._posix_reduction import _mk_inheritable  # noqa: F401
else:
    from . import _win_reduction  # noqa: F401

# global variable to change the pickler behavior
try:
    from joblib.externals import cloudpickle  # noqa: F401

    DEFAULT_ENV = "cloudpickle"
except ImportError:
    # If cloudpickle is not present, fallback to pickle
    DEFAULT_ENV = "pickle"

ENV_LOKY_PICKLER = os.environ.get("LOKY_PICKLER", DEFAULT_ENV)
_LokyPickler = None
_loky_pickler_name = None


def set_loky_pickler(loky_pickler=None):
    global _LokyPickler, _loky_pickler_name

    if loky_pickler is None:
        loky_pickler = ENV_LOKY_PICKLER

    loky_pickler_cls = None

    # The default loky_pickler is cloudpickle
    if loky_pickler in ["", None]:
        loky_pickler = "cloudpickle"

    if loky_pickler == _loky_pickler_name:
        return

    if loky_pickler == "cloudpickle":
        from joblib.externals.cloudpickle import CloudPickler as loky_pickler_cls
    else:
        try:
            from importlib import import_module

            module_pickle = import_module(loky_pickler)
            loky_pickler_cls = module_pickle.Pickler
        except (ImportError, AttributeError) as e:
            extra_info = (
                "\nThis error occurred while setting loky_pickler to"
                f" '{loky_pickler}', as required by the env variable "
                "LOKY_PICKLER or the function set_loky_pickler."
            )
            e.args = (e.args[0] + extra_info,) + e.args[1:]
            e.msg = e.args[0]
            raise e

    util.debug(
        f"Using '{loky_pickler if loky_pickler else 'cloudpickle'}' for "
        "serialization."
    )

    class CustomizablePickler(loky_pickler_cls):
        _loky_pickler_cls = loky_pickler_cls

        def _set_dispatch_table(self, dispatch_table):
            for ancestor_class in self._loky_pickler_cls.mro():
                dt_attribute = getattr(ancestor_class, "dispatch_table", None)
                if isinstance(dt_attribute, types.MemberDescriptorType):
                    # Ancestor class (typically _pickle.Pickler) has a
                    # member_descriptor for its "dispatch_table" attribute. Use
                    # it to set the dispatch_table as a member instead of a
                    # dynamic attribute in the __dict__ of the instance,
                    # otherwise it will not be taken into account by the C
                    # implementation of the dump method if a subclass defines a
                    # class-level dispatch_table attribute as was done in
                    # cloudpickle 1.6.0:
                    # https://github.com/joblib/loky/pull/260
                    dt_attribute.__set__(self, dispatch_table)
                    break

            # On top of member descriptor set, also use setattr such that code
            # that directly access self.dispatch_table gets a consistent view
            # of the same table.
            self.dispatch_table = dispatch_table

        def __init__(self, writer, reducers=None, protocol=HIGHEST_PROTOCOL):
            loky_pickler_cls.__init__(self, writer, protocol=protocol)
            if reducers is None:
                reducers = {}

            if hasattr(self, "dispatch_table"):
                # Force a copy that we will update without mutating the
                # any class level defined dispatch_table.
                loky_dt = dict(self.dispatch_table)
            else:
                # Use standard reducers as bases
                loky_dt = copyreg.dispatch_table.copy()

            # Register loky specific reducers
            loky_dt.update(_dispatch_table)

            # Set the new dispatch table, taking care of the fact that we
            # need to use the member_descriptor when we inherit from a
            # subclass of the C implementation of the Pickler base class
            # with an class level dispatch_table attribute.
            self._set_dispatch_table(loky_dt)

            # Register the reducers
            for type, reduce_func in reducers.items():
                self.register(type, reduce_func)

        def register(self, type, reduce_func):
            """Attach a reducer function to a given type in the dispatch table."""
            self.dispatch_table[type] = reduce_func

    _LokyPickler = CustomizablePickler
    _loky_pickler_name = loky_pickler


def get_loky_pickler_name():
    global _loky_pickler_name
    return _loky_pickler_name


def get_loky_pickler():
    global _LokyPickler
    return _LokyPickler


# Set it to its default value
set_loky_pickler()


def dump(obj, file, reducers=None, protocol=None):
    """Replacement for pickle.dump() using _LokyPickler."""
    global _LokyPickler
    _LokyPickler(file, reducers=reducers, protocol=protocol).dump(obj)


def dumps(obj, reducers=None, protocol=None):
    global _LokyPickler

    buf = io.BytesIO()
    dump(obj, buf, reducers=reducers, protocol=protocol)
    return buf.getbuffer()


__all__ = ["dump", "dumps", "loads", "register", "set_loky_pickler"]

if sys.platform == "win32":
    from multiprocessing.reduction import duplicate

    __all__ += ["duplicate"]
