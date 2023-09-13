# coding: utf-8
"""
Inputhook management for GUI event loop integration.
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import sys
import select

#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

# Constants for identifying the GUI toolkits.
GUI_WX = 'wx'
GUI_QT = 'qt'
GUI_QT4 = 'qt4'
GUI_QT5 = 'qt5'
GUI_GTK = 'gtk'
GUI_TK = 'tk'
GUI_OSX = 'osx'
GUI_GLUT = 'glut'
GUI_PYGLET = 'pyglet'
GUI_GTK3 = 'gtk3'
GUI_NONE = 'none'  # i.e. disable

#-----------------------------------------------------------------------------
# Utilities
#-----------------------------------------------------------------------------


def ignore_CTRL_C():
    """Ignore CTRL+C (not implemented)."""
    pass


def allow_CTRL_C():
    """Take CTRL+C into account (not implemented)."""
    pass

#-----------------------------------------------------------------------------
# Main InputHookManager class
#-----------------------------------------------------------------------------


class InputHookManager(object):
    """Manage PyOS_InputHook for different GUI toolkits.

    This class installs various hooks under ``PyOSInputHook`` to handle
    GUI event loop integration.
    """

    def __init__(self):
        self._return_control_callback = None
        self._apps = {}
        self._reset()
        self.pyplot_imported = False

    def _reset(self):
        self._callback_pyfunctype = None
        self._callback = None
        self._current_gui = None

    def set_return_control_callback(self, return_control_callback):
        self._return_control_callback = return_control_callback

    def get_return_control_callback(self):
        return self._return_control_callback

    def return_control(self):
        return self._return_control_callback()

    def get_inputhook(self):
        return self._callback

    def set_inputhook(self, callback):
        """Set inputhook to callback."""
        # We don't (in the context of PyDev console) actually set PyOS_InputHook, but rather
        # while waiting for input on xmlrpc we run this code
        self._callback = callback

    def clear_inputhook(self, app=None):
        """Clear input hook.

        Parameters
        ----------
        app : optional, ignored
          This parameter is allowed only so that clear_inputhook() can be
          called with a similar interface as all the ``enable_*`` methods.  But
          the actual value of the parameter is ignored.  This uniform interface
          makes it easier to have user-level entry points in the main IPython
          app like :meth:`enable_gui`."""
        self._reset()

    def clear_app_refs(self, gui=None):
        """Clear IPython's internal reference to an application instance.

        Whenever we create an app for a user on qt4 or wx, we hold a
        reference to the app.  This is needed because in some cases bad things
        can happen if a user doesn't hold a reference themselves.  This
        method is provided to clear the references we are holding.

        Parameters
        ----------
        gui : None or str
            If None, clear all app references.  If ('wx', 'qt4') clear
            the app for that toolkit.  References are not held for gtk or tk
            as those toolkits don't have the notion of an app.
        """
        if gui is None:
            self._apps = {}
        elif gui in self._apps:
            del self._apps[gui]

    def enable_wx(self, app=None):
        """Enable event loop integration with wxPython.

        Parameters
        ----------
        app : WX Application, optional.
            Running application to use.  If not given, we probe WX for an
            existing application object, and create a new one if none is found.

        Notes
        -----
        This methods sets the ``PyOS_InputHook`` for wxPython, which allows
        the wxPython to integrate with terminal based applications like
        IPython.

        If ``app`` is not given we probe for an existing one, and return it if
        found.  If no existing app is found, we create an :class:`wx.App` as
        follows::

            import wx
            app = wx.App(redirect=False, clearSigInt=False)
        """
        import wx
        from distutils.version import LooseVersion as V
        wx_version = V(wx.__version__).version  # @UndefinedVariable

        if wx_version < [2, 8]:
            raise ValueError("requires wxPython >= 2.8, but you have %s" % wx.__version__)  # @UndefinedVariable

        from pydev_ipython.inputhookwx import inputhook_wx
        self.set_inputhook(inputhook_wx)
        self._current_gui = GUI_WX

        if app is None:
            app = wx.GetApp()  # @UndefinedVariable
        if app is None:
            app = wx.App(redirect=False, clearSigInt=False)  # @UndefinedVariable
        app._in_event_loop = True
        self._apps[GUI_WX] = app
        return app

    def disable_wx(self):
        """Disable event loop integration with wxPython.

        This merely sets PyOS_InputHook to NULL.
        """
        if GUI_WX in self._apps:
            self._apps[GUI_WX]._in_event_loop = False
        self.clear_inputhook()

    def enable_qt(self, app=None):
        from pydev_ipython.qt_for_kernel import QT_API, QT_API_PYQT5
        if QT_API == QT_API_PYQT5:
            self.enable_qt5(app)
        else:
            self.enable_qt4(app)

    def enable_qt4(self, app=None):
        """Enable event loop integration with PyQt4.

        Parameters
        ----------
        app : Qt Application, optional.
            Running application to use.  If not given, we probe Qt for an
            existing application object, and create a new one if none is found.

        Notes
        -----
        This methods sets the PyOS_InputHook for PyQt4, which allows
        the PyQt4 to integrate with terminal based applications like
        IPython.

        If ``app`` is not given we probe for an existing one, and return it if
        found.  If no existing app is found, we create an :class:`QApplication`
        as follows::

            from PyQt4 import QtCore
            app = QtGui.QApplication(sys.argv)
        """
        from pydev_ipython.inputhookqt4 import create_inputhook_qt4
        app, inputhook_qt4 = create_inputhook_qt4(self, app)
        self.set_inputhook(inputhook_qt4)

        self._current_gui = GUI_QT4
        app._in_event_loop = True
        self._apps[GUI_QT4] = app
        return app

    def disable_qt4(self):
        """Disable event loop integration with PyQt4.

        This merely sets PyOS_InputHook to NULL.
        """
        if GUI_QT4 in self._apps:
            self._apps[GUI_QT4]._in_event_loop = False
        self.clear_inputhook()

    def enable_qt5(self, app=None):
        from pydev_ipython.inputhookqt5 import create_inputhook_qt5
        app, inputhook_qt5 = create_inputhook_qt5(self, app)
        self.set_inputhook(inputhook_qt5)

        self._current_gui = GUI_QT5
        app._in_event_loop = True
        self._apps[GUI_QT5] = app
        return app

    def disable_qt5(self):
        if GUI_QT5 in self._apps:
            self._apps[GUI_QT5]._in_event_loop = False
        self.clear_inputhook()

    def enable_gtk(self, app=None):
        """Enable event loop integration with PyGTK.

        Parameters
        ----------
        app : ignored
           Ignored, it's only a placeholder to keep the call signature of all
           gui activation methods consistent, which simplifies the logic of
           supporting magics.

        Notes
        -----
        This methods sets the PyOS_InputHook for PyGTK, which allows
        the PyGTK to integrate with terminal based applications like
        IPython.
        """
        from pydev_ipython.inputhookgtk import create_inputhook_gtk
        self.set_inputhook(create_inputhook_gtk(self._stdin_file))
        self._current_gui = GUI_GTK

    def disable_gtk(self):
        """Disable event loop integration with PyGTK.

        This merely sets PyOS_InputHook to NULL.
        """
        self.clear_inputhook()

    def enable_tk(self, app=None):
        """Enable event loop integration with Tk.

        Parameters
        ----------
        app : toplevel :class:`Tkinter.Tk` widget, optional.
            Running toplevel widget to use.  If not given, we probe Tk for an
            existing one, and create a new one if none is found.

        Notes
        -----
        If you have already created a :class:`Tkinter.Tk` object, the only
        thing done by this method is to register with the
        :class:`InputHookManager`, since creating that object automatically
        sets ``PyOS_InputHook``.
        """
        self._current_gui = GUI_TK
        if app is None:
            try:
                import Tkinter as _TK
            except:
                # Python 3
                import tkinter as _TK  # @UnresolvedImport
            app = _TK.Tk()
            app.withdraw()
            self._apps[GUI_TK] = app

        from pydev_ipython.inputhooktk import create_inputhook_tk
        self.set_inputhook(create_inputhook_tk(app))
        return app

    def disable_tk(self):
        """Disable event loop integration with Tkinter.

        This merely sets PyOS_InputHook to NULL.
        """
        self.clear_inputhook()

    def enable_glut(self, app=None):
        """ Enable event loop integration with GLUT.

        Parameters
        ----------

        app : ignored
            Ignored, it's only a placeholder to keep the call signature of all
            gui activation methods consistent, which simplifies the logic of
            supporting magics.

        Notes
        -----

        This methods sets the PyOS_InputHook for GLUT, which allows the GLUT to
        integrate with terminal based applications like IPython. Due to GLUT
        limitations, it is currently not possible to start the event loop
        without first creating a window. You should thus not create another
        window but use instead the created one. See 'gui-glut.py' in the
        docs/examples/lib directory.

        The default screen mode is set to:
        glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH
        """

        import OpenGL.GLUT as glut  # @UnresolvedImport
        from pydev_ipython.inputhookglut import glut_display_mode, \
                                              glut_close, glut_display, \
                                              glut_idle, inputhook_glut

        if GUI_GLUT not in self._apps:
            argv = getattr(sys, 'argv', [])
            glut.glutInit(argv)
            glut.glutInitDisplayMode(glut_display_mode)
            # This is specific to freeglut
            if bool(glut.glutSetOption):
                glut.glutSetOption(glut.GLUT_ACTION_ON_WINDOW_CLOSE,
                                    glut.GLUT_ACTION_GLUTMAINLOOP_RETURNS)
            glut.glutCreateWindow(argv[0] if len(argv) > 0 else '')
            glut.glutReshapeWindow(1, 1)
            glut.glutHideWindow()
            glut.glutWMCloseFunc(glut_close)
            glut.glutDisplayFunc(glut_display)
            glut.glutIdleFunc(glut_idle)
        else:
            glut.glutWMCloseFunc(glut_close)
            glut.glutDisplayFunc(glut_display)
            glut.glutIdleFunc(glut_idle)
        self.set_inputhook(inputhook_glut)
        self._current_gui = GUI_GLUT
        self._apps[GUI_GLUT] = True

    def disable_glut(self):
        """Disable event loop integration with glut.

        This sets PyOS_InputHook to NULL and set the display function to a
        dummy one and set the timer to a dummy timer that will be triggered
        very far in the future.
        """
        import OpenGL.GLUT as glut  # @UnresolvedImport
        from glut_support import glutMainLoopEvent  # @UnresolvedImport

        glut.glutHideWindow()  # This is an event to be processed below
        glutMainLoopEvent()
        self.clear_inputhook()

    def enable_pyglet(self, app=None):
        """Enable event loop integration with pyglet.

        Parameters
        ----------
        app : ignored
           Ignored, it's only a placeholder to keep the call signature of all
           gui activation methods consistent, which simplifies the logic of
           supporting magics.

        Notes
        -----
        This methods sets the ``PyOS_InputHook`` for pyglet, which allows
        pyglet to integrate with terminal based applications like
        IPython.

        """
        from pydev_ipython.inputhookpyglet import inputhook_pyglet
        self.set_inputhook(inputhook_pyglet)
        self._current_gui = GUI_PYGLET
        return app

    def disable_pyglet(self):
        """Disable event loop integration with pyglet.

        This merely sets PyOS_InputHook to NULL.
        """
        self.clear_inputhook()

    def enable_gtk3(self, app=None):
        """Enable event loop integration with Gtk3 (gir bindings).

        Parameters
        ----------
        app : ignored
           Ignored, it's only a placeholder to keep the call signature of all
           gui activation methods consistent, which simplifies the logic of
           supporting magics.

        Notes
        -----
        This methods sets the PyOS_InputHook for Gtk3, which allows
        the Gtk3 to integrate with terminal based applications like
        IPython.
        """
        from pydev_ipython.inputhookgtk3 import create_inputhook_gtk3
        self.set_inputhook(create_inputhook_gtk3(self._stdin_file))
        self._current_gui = GUI_GTK

    def disable_gtk3(self):
        """Disable event loop integration with PyGTK.

        This merely sets PyOS_InputHook to NULL.
        """
        self.clear_inputhook()

    def enable_mac(self, app=None):
        """ Enable event loop integration with MacOSX.

        We call function pyplot.pause, which updates and displays active
        figure during pause. It's not MacOSX-specific, but it enables to
        avoid inputhooks in native MacOSX backend.
        Also we shouldn't import pyplot, until user does it. Cause it's
        possible to choose backend before importing pyplot for the first
        time only.
        """

        def inputhook_mac(app=None):
            if self.pyplot_imported:
                pyplot = sys.modules['matplotlib.pyplot']
                try:
                    pyplot.pause(0.01)
                except:
                    pass
            else:
                if 'matplotlib.pyplot' in sys.modules:
                    self.pyplot_imported = True

        self.set_inputhook(inputhook_mac)
        self._current_gui = GUI_OSX

    def disable_mac(self):
        self.clear_inputhook()

    def current_gui(self):
        """Return a string indicating the currently active GUI or None."""
        return self._current_gui


inputhook_manager = InputHookManager()

enable_wx = inputhook_manager.enable_wx
disable_wx = inputhook_manager.disable_wx
enable_qt = inputhook_manager.enable_qt
enable_qt4 = inputhook_manager.enable_qt4
disable_qt4 = inputhook_manager.disable_qt4
enable_qt5 = inputhook_manager.enable_qt5
disable_qt5 = inputhook_manager.disable_qt5
enable_gtk = inputhook_manager.enable_gtk
disable_gtk = inputhook_manager.disable_gtk
enable_tk = inputhook_manager.enable_tk
disable_tk = inputhook_manager.disable_tk
enable_glut = inputhook_manager.enable_glut
disable_glut = inputhook_manager.disable_glut
enable_pyglet = inputhook_manager.enable_pyglet
disable_pyglet = inputhook_manager.disable_pyglet
enable_gtk3 = inputhook_manager.enable_gtk3
disable_gtk3 = inputhook_manager.disable_gtk3
enable_mac = inputhook_manager.enable_mac
disable_mac = inputhook_manager.disable_mac
clear_inputhook = inputhook_manager.clear_inputhook
set_inputhook = inputhook_manager.set_inputhook
current_gui = inputhook_manager.current_gui
clear_app_refs = inputhook_manager.clear_app_refs

# We maintain this as stdin_ready so that the individual inputhooks
# can diverge as little as possible from their IPython sources
stdin_ready = inputhook_manager.return_control
set_return_control_callback = inputhook_manager.set_return_control_callback
get_return_control_callback = inputhook_manager.get_return_control_callback
get_inputhook = inputhook_manager.get_inputhook


# Convenience function to switch amongst them
def enable_gui(gui=None, app=None):
    """Switch amongst GUI input hooks by name.

    This is just a utility wrapper around the methods of the InputHookManager
    object.

    Parameters
    ----------
    gui : optional, string or None
      If None (or 'none'), clears input hook, otherwise it must be one
      of the recognized GUI names (see ``GUI_*`` constants in module).

    app : optional, existing application object.
      For toolkits that have the concept of a global app, you can supply an
      existing one.  If not given, the toolkit will be probed for one, and if
      none is found, a new one will be created.  Note that GTK does not have
      this concept, and passing an app if ``gui=="GTK"`` will raise an error.

    Returns
    -------
    The output of the underlying gui switch routine, typically the actual
    PyOS_InputHook wrapper object or the GUI toolkit app created, if there was
    one.
    """

    if get_return_control_callback() is None:
        raise ValueError("A return_control_callback must be supplied as a reference before a gui can be enabled")

    guis = {GUI_NONE: clear_inputhook,
            GUI_OSX: enable_mac,
            GUI_TK: enable_tk,
            GUI_GTK: enable_gtk,
            GUI_WX: enable_wx,
            GUI_QT: enable_qt,
            GUI_QT4: enable_qt4,
            GUI_QT5: enable_qt5,
            GUI_GLUT: enable_glut,
            GUI_PYGLET: enable_pyglet,
            GUI_GTK3: enable_gtk3,
            }
    try:
        gui_hook = guis[gui]
    except KeyError:
        if gui is None or gui == '':
            gui_hook = clear_inputhook
        else:
            e = "Invalid GUI request %r, valid ones are:%s" % (gui, list(guis.keys()))
            raise ValueError(e)
    return gui_hook(app)


__all__ = [
    "GUI_WX",
    "GUI_QT",
    "GUI_QT4",
    "GUI_QT5",
    "GUI_GTK",
    "GUI_TK",
    "GUI_OSX",
    "GUI_GLUT",
    "GUI_PYGLET",
    "GUI_GTK3",
    "GUI_NONE",

    "ignore_CTRL_C",
    "allow_CTRL_C",

    "InputHookManager",

    "inputhook_manager",

    "enable_wx",
    "disable_wx",
    "enable_qt",
    "enable_qt4",
    "disable_qt4",
    "enable_qt5",
    "disable_qt5",
    "enable_gtk",
    "disable_gtk",
    "enable_tk",
    "disable_tk",
    "enable_glut",
    "disable_glut",
    "enable_pyglet",
    "disable_pyglet",
    "enable_gtk3",
    "disable_gtk3",
    "enable_mac",
    "disable_mac",
    "clear_inputhook",
    "set_inputhook",
    "current_gui",
    "clear_app_refs",

    "stdin_ready",
    "set_return_control_callback",
    "get_return_control_callback",
    "get_inputhook",

    "enable_gui"]
