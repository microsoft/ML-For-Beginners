from __future__ import nested_scopes

from _pydev_bundle._pydev_saved_modules import threading
import os
from _pydev_bundle import pydev_log


def set_trace_in_qt():
    from _pydevd_bundle.pydevd_comm import get_global_debugger
    py_db = get_global_debugger()
    if py_db is not None:
        threading.current_thread()  # Create the dummy thread for qt.
        py_db.enable_tracing()


_patched_qt = False


def patch_qt(qt_support_mode):
    '''
    This method patches qt (PySide2, PySide, PyQt4, PyQt5) so that we have hooks to set the tracing for QThread.
    '''
    if not qt_support_mode:
        return

    if qt_support_mode is True or qt_support_mode == 'True':
        # do not break backward compatibility
        qt_support_mode = 'auto'

    if qt_support_mode == 'auto':
        qt_support_mode = os.getenv('PYDEVD_PYQT_MODE', 'auto')

    # Avoid patching more than once
    global _patched_qt
    if _patched_qt:
        return

    pydev_log.debug('Qt support mode: %s', qt_support_mode)

    _patched_qt = True

    if qt_support_mode == 'auto':

        patch_qt_on_import = None
        try:
            import PySide2  # @UnresolvedImport @UnusedImport
            qt_support_mode = 'pyside2'
        except:
            try:
                import Pyside  # @UnresolvedImport @UnusedImport
                qt_support_mode = 'pyside'
            except:
                try:
                    import PyQt5  # @UnresolvedImport @UnusedImport
                    qt_support_mode = 'pyqt5'
                except:
                    try:
                        import PyQt4  # @UnresolvedImport @UnusedImport
                        qt_support_mode = 'pyqt4'
                    except:
                        return

    if qt_support_mode == 'pyside2':
        try:
            import PySide2.QtCore  # @UnresolvedImport
            _internal_patch_qt(PySide2.QtCore, qt_support_mode)
        except:
            return

    elif qt_support_mode == 'pyside':
        try:
            import PySide.QtCore  # @UnresolvedImport
            _internal_patch_qt(PySide.QtCore, qt_support_mode)
        except:
            return

    elif qt_support_mode == 'pyqt5':
        try:
            import PyQt5.QtCore  # @UnresolvedImport
            _internal_patch_qt(PyQt5.QtCore)
        except:
            return

    elif qt_support_mode == 'pyqt4':
        # Ok, we have an issue here:
        # PyDev-452: Selecting PyQT API version using sip.setapi fails in debug mode
        # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
        # Mostly, if the user uses a different API version (i.e.: v2 instead of v1),
        # that has to be done before importing PyQt4 modules (PySide/PyQt5 don't have this issue
        # as they only implements v2).
        patch_qt_on_import = 'PyQt4'

        def get_qt_core_module():
            import PyQt4.QtCore  # @UnresolvedImport
            return PyQt4.QtCore

        _patch_import_to_patch_pyqt_on_import(patch_qt_on_import, get_qt_core_module)

    else:
        raise ValueError('Unexpected qt support mode: %s' % (qt_support_mode,))


def _patch_import_to_patch_pyqt_on_import(patch_qt_on_import, get_qt_core_module):
    # I don't like this approach very much as we have to patch __import__, but I like even less
    # asking the user to configure something in the client side...
    # So, our approach is to patch PyQt4 right before the user tries to import it (at which
    # point he should've set the sip api version properly already anyways).

    pydev_log.debug('Setting up Qt post-import monkeypatch.')

    dotted = patch_qt_on_import + '.'
    original_import = __import__

    from _pydev_bundle._pydev_sys_patch import patch_sys_module, patch_reload, cancel_patches_in_sys_module

    patch_sys_module()
    patch_reload()

    def patched_import(name, *args, **kwargs):
        if patch_qt_on_import == name or name.startswith(dotted):
            builtins.__import__ = original_import
            cancel_patches_in_sys_module()
            _internal_patch_qt(get_qt_core_module())  # Patch it only when the user would import the qt module
        return original_import(name, *args, **kwargs)

    import builtins  # Py3

    builtins.__import__ = patched_import


def _internal_patch_qt(QtCore, qt_support_mode='auto'):
    pydev_log.debug('Patching Qt: %s', QtCore)

    _original_thread_init = QtCore.QThread.__init__
    _original_runnable_init = QtCore.QRunnable.__init__
    _original_QThread = QtCore.QThread

    class FuncWrapper:

        def __init__(self, original):
            self._original = original

        def __call__(self, *args, **kwargs):
            set_trace_in_qt()
            return self._original(*args, **kwargs)

    class StartedSignalWrapper(QtCore.QObject):  # Wrapper for the QThread.started signal

        try:
            _signal = QtCore.Signal()  # @UndefinedVariable
        except:
            _signal = QtCore.pyqtSignal()  # @UndefinedVariable

        def __init__(self, thread, original_started):
            QtCore.QObject.__init__(self)
            self.thread = thread
            self.original_started = original_started
            if qt_support_mode in ('pyside', 'pyside2'):
                self._signal = original_started
            else:
                self._signal.connect(self._on_call)
                self.original_started.connect(self._signal)

        def connect(self, func, *args, **kwargs):
            if qt_support_mode in ('pyside', 'pyside2'):
                return self._signal.connect(FuncWrapper(func), *args, **kwargs)
            else:
                return self._signal.connect(func, *args, **kwargs)

        def disconnect(self, *args, **kwargs):
            return self._signal.disconnect(*args, **kwargs)

        def emit(self, *args, **kwargs):
            return self._signal.emit(*args, **kwargs)

        def _on_call(self, *args, **kwargs):
            set_trace_in_qt()

    class ThreadWrapper(QtCore.QThread):  # Wrapper for QThread

        def __init__(self, *args, **kwargs):
            _original_thread_init(self, *args, **kwargs)

            # In PyQt5 the program hangs when we try to call original run method of QThread class.
            # So we need to distinguish instances of QThread class and instances of QThread inheritors.
            if self.__class__.run == _original_QThread.run:
                self.run = self._exec_run
            else:
                self._original_run = self.run
                self.run = self._new_run
            self._original_started = self.started
            self.started = StartedSignalWrapper(self, self.started)

        def _exec_run(self):
            set_trace_in_qt()
            self.exec_()
            return None

        def _new_run(self):
            set_trace_in_qt()
            return self._original_run()

    class RunnableWrapper(QtCore.QRunnable):  # Wrapper for QRunnable

        def __init__(self, *args, **kwargs):
            _original_runnable_init(self, *args, **kwargs)

            self._original_run = self.run
            self.run = self._new_run

        def _new_run(self):
            set_trace_in_qt()
            return self._original_run()

    QtCore.QThread = ThreadWrapper
    QtCore.QRunnable = RunnableWrapper
