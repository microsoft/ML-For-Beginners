
import sys
import traceback
from types import ModuleType
from _pydevd_bundle.pydevd_constants import DebugInfoHolder

import builtins


class ImportHookManager(ModuleType):

    def __init__(self, name, system_import):
        ModuleType.__init__(self, name)
        self._system_import = system_import
        self._modules_to_patch = {}

    def add_module_name(self, module_name, activate_function):
        self._modules_to_patch[module_name] = activate_function

    def do_import(self, name, *args, **kwargs):
        module = self._system_import(name, *args, **kwargs)
        try:
            activate_func = self._modules_to_patch.pop(name, None)
            if activate_func:
                activate_func()  # call activate function
        except:
            if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 2:
                traceback.print_exc()

        # Restore normal system importer to reduce performance impact
        # of calling this method every time an import statement is invoked
        if not self._modules_to_patch:
            builtins.__import__ = self._system_import

        return module


import_hook_manager = ImportHookManager(__name__ + '.import_hook', builtins.__import__)
builtins.__import__ = import_hook_manager.do_import
sys.modules[import_hook_manager.__name__] = import_hook_manager
