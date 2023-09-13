'''
This module holds the customization settings for the debugger.
'''

from _pydevd_bundle.pydevd_constants import QUOTED_LINE_PROTOCOL
from _pydev_bundle import pydev_log
import sys


class PydevdCustomization(object):
    DEFAULT_PROTOCOL: str = QUOTED_LINE_PROTOCOL

    # Debug mode may be set to 'debugpy-dap'.
    #
    # In 'debugpy-dap' mode the following settings are done to PyDB:
    #
    # py_db.skip_suspend_on_breakpoint_exception = (BaseException,)
    # py_db.skip_print_breakpoint_exception = (NameError,)
    # py_db.multi_threads_single_notification = True
    DEBUG_MODE: str = ''

    # This may be a <sys_path_entry>;<module_name> to be pre-imported
    # Something as: 'c:/temp/foo;my_module.bar'
    #
    # What's done in this case is something as:
    #
    # sys.path.insert(0, <sys_path_entry>)
    # try:
    #     import <module_name>
    # finally:
    #     del sys.path[0]
    #
    # If the pre-import fails an output message is
    # sent (but apart from that debugger execution
    # should continue).
    PREIMPORT: str = ''


def on_pydb_init(py_db):
    if PydevdCustomization.DEBUG_MODE == 'debugpy-dap':
        pydev_log.debug('Apply debug mode: debugpy-dap')
        py_db.skip_suspend_on_breakpoint_exception = (BaseException,)
        py_db.skip_print_breakpoint_exception = (NameError,)
        py_db.multi_threads_single_notification = True
    elif not PydevdCustomization.DEBUG_MODE:
        pydev_log.debug('Apply debug mode: default')
    else:
        pydev_log.debug('WARNING: unknown debug mode: %s', PydevdCustomization.DEBUG_MODE)

    if PydevdCustomization.PREIMPORT:
        pydev_log.debug('Preimport: %s', PydevdCustomization.PREIMPORT)
        try:
            sys_path_entry, module_name = PydevdCustomization.PREIMPORT.rsplit(';', maxsplit=1)
        except Exception:
            pydev_log.exception("Expected ';' in %s" % (PydevdCustomization.PREIMPORT,))
        else:
            try:
                sys.path.insert(0, sys_path_entry)
                try:
                    __import__(module_name)
                finally:
                    sys.path.remove(sys_path_entry)
            except Exception:
                pydev_log.exception(
                    "Error importing %s (with sys.path entry: %s)" % (module_name, sys_path_entry))

