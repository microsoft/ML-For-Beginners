import types

from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_trace_api

try:
    from pydevd_plugins import django_debug
except:
    django_debug = None
    pydev_log.debug('Unable to load django_debug plugin')

try:
    from pydevd_plugins import jinja2_debug
except:
    jinja2_debug = None
    pydev_log.debug('Unable to load jinja2_debug plugin')

def load_plugins():
    plugins = []
    if django_debug is not None:
        plugins.append(django_debug)

    if jinja2_debug is not None:
        plugins.append(jinja2_debug)
    return plugins


def bind_func_to_method(func, obj, method_name):
    bound_method = types.MethodType(func, obj)

    setattr(obj, method_name, bound_method)
    return bound_method


class PluginManager(object):

    def __init__(self, main_debugger):
        self.plugins = load_plugins()
        self.active_plugins = []
        self.main_debugger = main_debugger
        self.rebind_methods()

    def add_breakpoint(self, func_name, *args, **kwargs):
        # add breakpoint for plugin and remember which plugin to use in tracing
        for plugin in self.plugins:
            if hasattr(plugin, func_name):
                func = getattr(plugin, func_name)
                result = func(self, *args, **kwargs)
                if result:
                    self.activate(plugin)

                    return result
        return None

    def activate(self, plugin):
        if plugin not in self.active_plugins:
            self.active_plugins.append(plugin)
            self.rebind_methods()

    def rebind_methods(self):
        if len(self.active_plugins) == 0:
            self.bind_functions(pydevd_trace_api, getattr, pydevd_trace_api)
        elif len(self.active_plugins) == 1:
            self.bind_functions(pydevd_trace_api, getattr, self.active_plugins[0])
        else:
            self.bind_functions(pydevd_trace_api, create_dispatch, self.active_plugins)

    def bind_functions(self, interface, function_factory, arg):
        for name in dir(interface):
            func = function_factory(arg, name)
            if type(func) == types.FunctionType:
                bind_func_to_method(func, self, name)


def create_dispatch(obj, name):
    def dispatch(self, *args, **kwargs):
        result = None
        for p in self.active_plugins:
            r = getattr(p, name)(self, *args, **kwargs)
            if not result:
                result = r
        return result
    return dispatch








