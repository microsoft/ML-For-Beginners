# coding: utf-8
"""
Support for creating GUI apps and starting event loops.

IPython's GUI integration allows interactive plotting and GUI usage in IPython
session. IPython has two different types of GUI integration:

1. The terminal based IPython supports GUI event loops through Python's
   PyOS_InputHook. PyOS_InputHook is a hook that Python calls periodically
   whenever raw_input is waiting for a user to type code. We implement GUI
   support in the terminal by setting PyOS_InputHook to a function that
   iterates the event loop for a short while. It is important to note that
   in this situation, the real GUI event loop is NOT run in the normal
   manner, so you can't use the normal means to detect that it is running.
2. In the two process IPython kernel/frontend, the GUI event loop is run in
   the kernel. In this case, the event loop is run in the normal manner by
   calling the function or method of the GUI toolkit that starts the event
   loop.

In addition to starting the GUI event loops in one of these two ways, IPython
will *always* create an appropriate GUI application object when GUi
integration is enabled.

If you want your GUI apps to run in IPython you need to do two things:

1. Test to see if there is already an existing main application object. If
   there is, you should use it. If there is not an existing application object
   you should create one.
2. Test to see if the GUI event loop is running. If it is, you should not
   start it. If the event loop is not running you may start it.

This module contains functions for each toolkit that perform these things
in a consistent manner. Because of how PyOS_InputHook runs the event loop
you cannot detect if the event loop is running using the traditional calls
(such as ``wx.GetApp.IsMainLoopRunning()`` in wxPython). If PyOS_InputHook is
set These methods will return a false negative. That is, they will say the
event loop is not running, when is actually is. To work around this limitation
we proposed the following informal protocol:

* Whenever someone starts the event loop, they *must* set the ``_in_event_loop``
  attribute of the main application object to ``True``. This should be done
  regardless of how the event loop is actually run.
* Whenever someone stops the event loop, they *must* set the ``_in_event_loop``
  attribute of the main application object to ``False``.
* If you want to see if the event loop is running, you *must* use ``hasattr``
  to see if ``_in_event_loop`` attribute has been set. If it is set, you
  *must* use its value. If it has not been set, you can query the toolkit
  in the normal manner.
* If you want GUI support and no one else has created an application or
  started the event loop you *must* do this. We don't want projects to
  attempt to defer these things to someone else if they themselves need it.

The functions below implement this logic for each GUI toolkit. If you need
to create custom application subclasses, you will likely have to modify this
code for your own purposes. This code can be copied into your own project
so you don't have to depend on IPython.

"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from IPython.core.getipython import get_ipython

#-----------------------------------------------------------------------------
# wx
#-----------------------------------------------------------------------------

def get_app_wx(*args, **kwargs):
    """Create a new wx app or return an exiting one."""
    import wx
    app = wx.GetApp()
    if app is None:
        if 'redirect' not in kwargs:
            kwargs['redirect'] = False
        app = wx.PySimpleApp(*args, **kwargs)
    return app

def is_event_loop_running_wx(app=None):
    """Is the wx event loop running."""
    # New way: check attribute on shell instance
    ip = get_ipython()
    if ip is not None:
        if ip.active_eventloop and ip.active_eventloop == 'wx':
            return True
        # Fall through to checking the application, because Wx has a native way
        # to check if the event loop is running, unlike Qt.

    # Old way: check Wx application
    if app is None:
        app = get_app_wx()
    if hasattr(app, '_in_event_loop'):
        return app._in_event_loop
    else:
        return app.IsMainLoopRunning()

def start_event_loop_wx(app=None):
    """Start the wx event loop in a consistent manner."""
    if app is None:
        app = get_app_wx()
    if not is_event_loop_running_wx(app):
        app._in_event_loop = True
        app.MainLoop()
        app._in_event_loop = False
    else:
        app._in_event_loop = True

#-----------------------------------------------------------------------------
# Qt
#-----------------------------------------------------------------------------

def get_app_qt4(*args, **kwargs):
    """Create a new Qt app or return an existing one."""
    from IPython.external.qt_for_kernel import QtGui
    app = QtGui.QApplication.instance()
    if app is None:
        if not args:
            args = ([""],)
        app = QtGui.QApplication(*args, **kwargs)
    return app

def is_event_loop_running_qt4(app=None):
    """Is the qt event loop running."""
    # New way: check attribute on shell instance
    ip = get_ipython()
    if ip is not None:
        return ip.active_eventloop and ip.active_eventloop.startswith('qt')

    # Old way: check attribute on QApplication singleton
    if app is None:
        app = get_app_qt4([""])
    if hasattr(app, '_in_event_loop'):
        return app._in_event_loop
    else:
        # Does qt provide a other way to detect this?
        return False

def start_event_loop_qt4(app=None):
    """Start the qt event loop in a consistent manner."""
    if app is None:
        app = get_app_qt4([""])
    if not is_event_loop_running_qt4(app):
        app._in_event_loop = True
        app.exec_()
        app._in_event_loop = False
    else:
        app._in_event_loop = True

#-----------------------------------------------------------------------------
# Tk
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# gtk
#-----------------------------------------------------------------------------
