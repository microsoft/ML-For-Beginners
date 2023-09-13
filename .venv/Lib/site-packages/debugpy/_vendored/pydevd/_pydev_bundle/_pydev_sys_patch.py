import sys


def patch_sys_module():

    def patched_exc_info(fun):

        def pydev_debugger_exc_info():
            type, value, traceback = fun()
            if type == ImportError:
                # we should not show frame added by plugin_import call
                if traceback and hasattr(traceback, "tb_next"):
                    return type, value, traceback.tb_next
            return type, value, traceback

        return pydev_debugger_exc_info

    system_exc_info = sys.exc_info
    sys.exc_info = patched_exc_info(system_exc_info)
    if not hasattr(sys, "system_exc_info"):
        sys.system_exc_info = system_exc_info


def patched_reload(orig_reload):

    def pydev_debugger_reload(module):
        orig_reload(module)
        if module.__name__ == "sys":
            # if sys module was reloaded we should patch it again
            patch_sys_module()

    return pydev_debugger_reload


def patch_reload():
    import builtins  # Py3

    if hasattr(builtins, "reload"):
        sys.builtin_orig_reload = builtins.reload
        builtins.reload = patched_reload(sys.builtin_orig_reload)  # @UndefinedVariable
        try:
            import imp
            sys.imp_orig_reload = imp.reload
            imp.reload = patched_reload(sys.imp_orig_reload)  # @UndefinedVariable
        except:
            pass
    else:
        try:
            import importlib
            sys.importlib_orig_reload = importlib.reload  # @UndefinedVariable
            importlib.reload = patched_reload(sys.importlib_orig_reload)  # @UndefinedVariable
        except:
            pass

    del builtins


def cancel_patches_in_sys_module():
    sys.exc_info = sys.system_exc_info  # @UndefinedVariable
    import builtins  # Py3

    if hasattr(sys, "builtin_orig_reload"):
        builtins.reload = sys.builtin_orig_reload

    if hasattr(sys, "imp_orig_reload"):
        import imp
        imp.reload = sys.imp_orig_reload

    if hasattr(sys, "importlib_orig_reload"):
        import importlib
        importlib.reload = sys.importlib_orig_reload

    del builtins
