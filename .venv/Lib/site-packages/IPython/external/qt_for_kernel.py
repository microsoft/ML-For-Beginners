""" Import Qt in a manner suitable for an IPython kernel.

This is the import used for the `gui=qt` or `matplotlib=qt` initialization.

Import Priority:

if Qt has been imported anywhere else:
   use that

if matplotlib has been imported and doesn't support v2 (<= 1.0.1):
    use PyQt4 @v1

Next, ask QT_API env variable

if QT_API not set:
    ask matplotlib what it's using. If Qt4Agg or Qt5Agg, then use the
        version matplotlib is configured with

    else: (matplotlib said nothing)
        # this is the default path - nobody told us anything
        try in this order:
            PyQt default version, PySide, PyQt5
else:
    use what QT_API says

    Note that %gui's implementation will always set a `QT_API`, see
    `IPython.terminal.pt_inputhooks.get_inputhook_name_and_func`

"""
# NOTE: This is no longer an external, third-party module, and should be
# considered part of IPython. For compatibility however, it is being kept in
# IPython/external.

import os
import sys

from IPython.external.qt_loaders import (
    load_qt,
    loaded_api,
    enum_factory,
    # QT6
    QT_API_PYQT6,
    QT_API_PYSIDE6,
    # QT5
    QT_API_PYQT5,
    QT_API_PYSIDE2,
    # QT4
    QT_API_PYQT,
    QT_API_PYSIDE,
    # default
    QT_API_PYQT_DEFAULT,
)

_qt_apis = (
    # QT6
    QT_API_PYQT6,
    QT_API_PYSIDE6,
    # QT5
    QT_API_PYQT5,
    QT_API_PYSIDE2,
    # default
    QT_API_PYQT_DEFAULT,
)


def matplotlib_options(mpl):
    """Constraints placed on an imported matplotlib."""
    if mpl is None:
        return
    backend = mpl.rcParams.get('backend', None)
    if backend == 'Qt4Agg':
        mpqt = mpl.rcParams.get('backend.qt4', None)
        if mpqt is None:
            return None
        if mpqt.lower() == 'pyside':
            return [QT_API_PYSIDE]
        elif mpqt.lower() == 'pyqt4':
            return [QT_API_PYQT_DEFAULT]
        elif mpqt.lower() == 'pyqt4v2':
            return [QT_API_PYQT]
        raise ImportError("unhandled value for backend.qt4 from matplotlib: %r" %
                          mpqt)
    elif backend == 'Qt5Agg':
        mpqt = mpl.rcParams.get('backend.qt5', None)
        if mpqt is None:
            return None
        if mpqt.lower() == 'pyqt5':
            return [QT_API_PYQT5]
        raise ImportError("unhandled value for backend.qt5 from matplotlib: %r" %
                          mpqt)

def get_options():
    """Return a list of acceptable QT APIs, in decreasing order of preference."""
    #already imported Qt somewhere. Use that
    loaded = loaded_api()
    if loaded is not None:
        return [loaded]

    mpl = sys.modules.get("matplotlib", None)

    if mpl is not None and tuple(mpl.__version__.split(".")) < ("1", "0", "2"):
        # 1.0.1 only supports PyQt4 v1
        return [QT_API_PYQT_DEFAULT]

    qt_api = os.environ.get('QT_API', None)
    if qt_api is None:
        #no ETS variable. Ask mpl, then use default fallback path
        return matplotlib_options(mpl) or [
            QT_API_PYQT_DEFAULT,
            QT_API_PYQT6,
            QT_API_PYSIDE6,
            QT_API_PYQT5,
            QT_API_PYSIDE2,
        ]
    elif qt_api not in _qt_apis:
        raise RuntimeError("Invalid Qt API %r, valid values are: %r" %
                           (qt_api, ', '.join(_qt_apis)))
    else:
        return [qt_api]


api_opts = get_options()
QtCore, QtGui, QtSvg, QT_API = load_qt(api_opts)
enum_helper = enum_factory(QT_API, QtCore)
