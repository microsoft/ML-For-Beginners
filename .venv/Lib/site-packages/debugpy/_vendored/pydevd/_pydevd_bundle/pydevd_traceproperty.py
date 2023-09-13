'''For debug purpose we are replacing actual builtin property by the debug property
'''
from _pydevd_bundle.pydevd_comm import get_global_debugger
from _pydev_bundle import pydev_log


#=======================================================================================================================
# replace_builtin_property
#=======================================================================================================================
def replace_builtin_property(new_property=None):
    if new_property is None:
        new_property = DebugProperty
    original = property
    try:
        import builtins
        builtins.__dict__['property'] = new_property
    except:
        pydev_log.exception()  # @Reimport
    return original


#=======================================================================================================================
# DebugProperty
#=======================================================================================================================
class DebugProperty(object):
    """A custom property which allows python property to get
    controlled by the debugger and selectively disable/re-enable
    the tracing.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        global_debugger = get_global_debugger()
        try:
            if global_debugger is not None and global_debugger.disable_property_getter_trace:
                global_debugger.disable_tracing()
            if self.fget is None:
                raise AttributeError("unreadable attribute")
            return self.fget(obj)
        finally:
            if global_debugger is not None:
                global_debugger.enable_tracing()

    def __set__(self, obj, value):
        global_debugger = get_global_debugger()
        try:
            if global_debugger is not None and global_debugger.disable_property_setter_trace:
                global_debugger.disable_tracing()
            if self.fset is None:
                raise AttributeError("can't set attribute")
            self.fset(obj, value)
        finally:
            if global_debugger is not None:
                global_debugger.enable_tracing()

    def __delete__(self, obj):
        global_debugger = get_global_debugger()
        try:
            if global_debugger is not None and global_debugger.disable_property_deleter_trace:
                global_debugger.disable_tracing()
            if self.fdel is None:
                raise AttributeError("can't delete attribute")
            self.fdel(obj)
        finally:
            if global_debugger is not None:
                global_debugger.enable_tracing()

    def getter(self, fget):
        """Overriding getter decorator for the property
        """
        self.fget = fget
        return self

    def setter(self, fset):
        """Overriding setter decorator for the property
        """
        self.fset = fset
        return self

    def deleter(self, fdel):
        """Overriding deleter decorator for the property
        """
        self.fdel = fdel
        return self

