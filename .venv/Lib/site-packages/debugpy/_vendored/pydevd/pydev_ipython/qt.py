""" A Qt API selector that can be used to switch between PyQt and PySide.

This uses the ETS 4.0 selection pattern of:
PySide first, PyQt with API v2. second.

Do not use this if you need PyQt with the old QString/QVariant API.
"""

import os

from pydev_ipython.qt_loaders import (load_qt, QT_API_PYSIDE,
                                         QT_API_PYQT, QT_API_PYQT5)

QT_API = os.environ.get('QT_API', None)
if QT_API not in [QT_API_PYSIDE, QT_API_PYQT, QT_API_PYQT5, None]:
    raise RuntimeError("Invalid Qt API %r, valid values are: %r, %r" %
                       (QT_API, QT_API_PYSIDE, QT_API_PYQT, QT_API_PYQT5))
if QT_API is None:
    api_opts = [QT_API_PYSIDE, QT_API_PYQT, QT_API_PYQT5]
else:
    api_opts = [QT_API]

QtCore, QtGui, QtSvg, QT_API = load_qt(api_opts)
