"""Event loop integration for the ZeroMQ-based kernels."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import platform
import sys
from functools import partial

import zmq
from packaging.version import Version as V  # noqa
from traitlets.config.application import Application


def _use_appnope():
    """Should we use appnope for dealing with OS X app nap?

    Checks if we are on OS X 10.9 or greater.
    """
    return sys.platform == "darwin" and V(platform.mac_ver()[0]) >= V("10.9")


# mapping of keys to loop functions
loop_map = {
    "inline": None,
    "nbagg": None,
    "webagg": None,
    "notebook": None,
    "ipympl": None,
    "widget": None,
    None: None,
}


def register_integration(*toolkitnames):
    """Decorator to register an event loop to integrate with the IPython kernel

    The decorator takes names to register the event loop as for the %gui magic.
    You can provide alternative names for the same toolkit.

    The decorated function should take a single argument, the IPython kernel
    instance, arrange for the event loop to call ``kernel.do_one_iteration()``
    at least every ``kernel._poll_interval`` seconds, and start the event loop.

    :mod:`ipykernel.eventloops` provides and registers such functions
    for a few common event loops.
    """

    def decorator(func):
        """Integration registration decorator."""
        for name in toolkitnames:
            loop_map[name] = func

        func.exit_hook = lambda kernel: None

        def exit_decorator(exit_func):
            """@func.exit is now a decorator

            to register a function to be called on exit
            """
            func.exit_hook = exit_func
            return exit_func

        func.exit = exit_decorator
        return func

    return decorator


def _notify_stream_qt(kernel):
    import operator
    from functools import lru_cache

    from IPython.external.qt_for_kernel import QtCore

    try:
        from IPython.external.qt_for_kernel import enum_helper
    except ImportError:

        @lru_cache(None)
        def enum_helper(name):
            return operator.attrgetter(name.rpartition(".")[0])(sys.modules[QtCore.__package__])

    def process_stream_events():
        """fall back to main loop when there's a socket event"""
        # call flush to ensure that the stream doesn't lose events
        # due to our consuming of the edge-triggered FD
        # flush returns the number of events consumed.
        # if there were any, wake it up
        if kernel.shell_stream.flush(limit=1):
            kernel._qt_notifier.setEnabled(False)
            kernel.app.qt_event_loop.quit()

    if not hasattr(kernel, "_qt_notifier"):
        fd = kernel.shell_stream.getsockopt(zmq.FD)
        kernel._qt_notifier = QtCore.QSocketNotifier(
            fd, enum_helper('QtCore.QSocketNotifier.Type').Read, kernel.app.qt_event_loop
        )
        kernel._qt_notifier.activated.connect(process_stream_events)
    else:
        kernel._qt_notifier.setEnabled(True)

    # there may already be unprocessed events waiting.
    # these events will not wake zmq's edge-triggered FD
    # since edge-triggered notification only occurs on new i/o activity.
    # process all the waiting events immediately
    # so we start in a clean state ensuring that any new i/o events will notify.
    # schedule first call on the eventloop as soon as it's running,
    # so we don't block here processing events
    if not hasattr(kernel, "_qt_timer"):
        kernel._qt_timer = QtCore.QTimer(kernel.app)
        kernel._qt_timer.setSingleShot(True)
        kernel._qt_timer.timeout.connect(process_stream_events)
    kernel._qt_timer.start(0)


@register_integration("qt", "qt5", "qt6")
def loop_qt(kernel):
    """Event loop for all supported versions of Qt."""
    _notify_stream_qt(kernel)  # install hook to stop event loop.

    # Start the event loop.
    kernel.app._in_event_loop = True

    # `exec` blocks until there's ZMQ activity.
    el = kernel.app.qt_event_loop  # for brevity
    el.exec() if hasattr(el, 'exec') else el.exec_()
    kernel.app._in_event_loop = False


# NOTE: To be removed in version 7
loop_qt5 = loop_qt


# exit and watch are the same for qt 4 and 5
@loop_qt.exit
def loop_qt_exit(kernel):
    kernel.app.exit()


def _loop_wx(app):
    """Inner-loop for running the Wx eventloop

    Pulled from guisupport.start_event_loop in IPython < 5.2,
    since IPython 5.2 only checks `get_ipython().active_eventloop` is defined,
    rather than if the eventloop is actually running.
    """
    app._in_event_loop = True
    app.MainLoop()
    app._in_event_loop = False


@register_integration("wx")
def loop_wx(kernel):
    """Start a kernel with wx event loop support."""

    import wx

    # Wx uses milliseconds
    poll_interval = int(1000 * kernel._poll_interval)

    def wake():
        """wake from wx"""
        if kernel.shell_stream.flush(limit=1):
            kernel.app.ExitMainLoop()
            return

    # We have to put the wx.Timer in a wx.Frame for it to fire properly.
    # We make the Frame hidden when we create it in the main app below.
    class TimerFrame(wx.Frame):
        def __init__(self, func):
            wx.Frame.__init__(self, None, -1)
            self.timer = wx.Timer(self)
            # Units for the timer are in milliseconds
            self.timer.Start(poll_interval)
            self.Bind(wx.EVT_TIMER, self.on_timer)
            self.func = func

        def on_timer(self, event):
            self.func()

    # We need a custom wx.App to create our Frame subclass that has the
    # wx.Timer to defer back to the tornado event loop.
    class IPWxApp(wx.App):
        def OnInit(self):
            self.frame = TimerFrame(wake)
            self.frame.Show(False)
            return True

    # The redirect=False here makes sure that wx doesn't replace
    # sys.stdout/stderr with its own classes.
    if not (getattr(kernel, "app", None) and isinstance(kernel.app, wx.App)):
        kernel.app = IPWxApp(redirect=False)

    # The import of wx on Linux sets the handler for signal.SIGINT
    # to 0.  This is a bug in wx or gtk.  We fix by just setting it
    # back to the Python default.
    import signal

    if not callable(signal.getsignal(signal.SIGINT)):
        signal.signal(signal.SIGINT, signal.default_int_handler)

    _loop_wx(kernel.app)


@loop_wx.exit
def loop_wx_exit(kernel):
    """Exit the wx loop."""
    import wx

    wx.Exit()


@register_integration("tk")
def loop_tk(kernel):
    """Start a kernel with the Tk event loop."""

    from tkinter import READABLE, Tk

    app = Tk()
    # Capability detection:
    # per https://docs.python.org/3/library/tkinter.html#file-handlers
    # file handlers are not available on Windows
    if hasattr(app, "createfilehandler"):
        # A basic wrapper for structural similarity with the Windows version
        class BasicAppWrapper:
            def __init__(self, app):
                self.app = app
                self.app.withdraw()

        def process_stream_events(stream, *a, **kw):
            """fall back to main loop when there's a socket event"""
            if stream.flush(limit=1):
                app.tk.deletefilehandler(stream.getsockopt(zmq.FD))
                app.quit()
                app.destroy()
                del kernel.app_wrapper

        # For Tkinter, we create a Tk object and call its withdraw method.
        kernel.app_wrapper = BasicAppWrapper(app)

        notifier = partial(process_stream_events, kernel.shell_stream)
        # seems to be needed for tk
        notifier.__name__ = "notifier"  # type:ignore[attr-defined]
        app.tk.createfilehandler(kernel.shell_stream.getsockopt(zmq.FD), READABLE, notifier)
        # schedule initial call after start
        app.after(0, notifier)

        app.mainloop()

    else:
        import asyncio

        import nest_asyncio

        nest_asyncio.apply()

        doi = kernel.do_one_iteration
        # Tk uses milliseconds
        poll_interval = int(1000 * kernel._poll_interval)

        class TimedAppWrapper:
            def __init__(self, app, func):
                self.app = app
                self.app.withdraw()
                self.func = func

            def on_timer(self):
                loop = asyncio.get_event_loop()
                try:
                    loop.run_until_complete(self.func())
                except Exception:
                    kernel.log.exception("Error in message handler")
                self.app.after(poll_interval, self.on_timer)

            def start(self):
                self.on_timer()  # Call it once to get things going.
                self.app.mainloop()

        kernel.app_wrapper = TimedAppWrapper(app, doi)
        kernel.app_wrapper.start()


@loop_tk.exit
def loop_tk_exit(kernel):
    """Exit the tk loop."""
    try:
        kernel.app_wrapper.app.destroy()
        del kernel.app_wrapper
    except (RuntimeError, AttributeError):
        pass


@register_integration("gtk")
def loop_gtk(kernel):
    """Start the kernel, coordinating with the GTK event loop"""
    from .gui.gtkembed import GTKEmbed

    gtk_kernel = GTKEmbed(kernel)
    gtk_kernel.start()
    kernel._gtk = gtk_kernel


@loop_gtk.exit
def loop_gtk_exit(kernel):
    """Exit the gtk loop."""
    kernel._gtk.stop()


@register_integration("gtk3")
def loop_gtk3(kernel):
    """Start the kernel, coordinating with the GTK event loop"""
    from .gui.gtk3embed import GTKEmbed

    gtk_kernel = GTKEmbed(kernel)
    gtk_kernel.start()
    kernel._gtk = gtk_kernel


@loop_gtk3.exit
def loop_gtk3_exit(kernel):
    """Exit the gtk3 loop."""
    kernel._gtk.stop()


@register_integration("osx")
def loop_cocoa(kernel):
    """Start the kernel, coordinating with the Cocoa CFRunLoop event loop
    via the matplotlib MacOSX backend.
    """
    from ._eventloop_macos import mainloop, stop

    real_excepthook = sys.excepthook

    def handle_int(etype, value, tb):
        """don't let KeyboardInterrupts look like crashes"""
        # wake the eventloop when we get a signal
        stop()
        if etype is KeyboardInterrupt:
            print("KeyboardInterrupt caught in CFRunLoop", file=sys.__stdout__)
        else:
            real_excepthook(etype, value, tb)

    while not kernel.shell.exit_now:
        try:
            # double nested try/except, to properly catch KeyboardInterrupt
            # due to pyzmq Issue #130
            try:
                # don't let interrupts during mainloop invoke crash_handler:
                sys.excepthook = handle_int
                mainloop(kernel._poll_interval)
                if kernel.shell_stream.flush(limit=1):
                    # events to process, return control to kernel
                    return
            except BaseException:
                raise
        except KeyboardInterrupt:
            # Ctrl-C shouldn't crash the kernel
            print("KeyboardInterrupt caught in kernel", file=sys.__stdout__)
        finally:
            # ensure excepthook is restored
            sys.excepthook = real_excepthook


@loop_cocoa.exit
def loop_cocoa_exit(kernel):
    """Exit the cocoa loop."""
    from ._eventloop_macos import stop

    stop()


@register_integration("asyncio")
def loop_asyncio(kernel):
    """Start a kernel with asyncio event loop support."""
    import asyncio

    loop = asyncio.get_event_loop()
    # loop is already running (e.g. tornado 5), nothing left to do
    if loop.is_running():
        return

    if loop.is_closed():
        # main loop is closed, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop._should_close = False  # type:ignore[attr-defined]

    # pause eventloop when there's an event on a zmq socket
    def process_stream_events(stream):
        """fall back to main loop when there's a socket event"""
        if stream.flush(limit=1):
            loop.stop()

    notifier = partial(process_stream_events, kernel.shell_stream)
    loop.add_reader(kernel.shell_stream.getsockopt(zmq.FD), notifier)
    loop.call_soon(notifier)

    while True:
        error = None
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            continue
        except Exception as e:
            error = e
        if loop._should_close:  # type:ignore[attr-defined]
            loop.close()
        if error is not None:
            raise error
        break


@loop_asyncio.exit
def loop_asyncio_exit(kernel):
    """Exit hook for asyncio"""
    import asyncio

    loop = asyncio.get_event_loop()

    async def close_loop():
        if hasattr(loop, "shutdown_asyncgens"):
            yield loop.shutdown_asyncgens()
        loop._should_close = True  # type:ignore[attr-defined]
        loop.stop()

    if loop.is_running():
        close_loop()

    elif not loop.is_closed():
        loop.run_until_complete(close_loop)  # type:ignore
        loop.close()


def set_qt_api_env_from_gui(gui):
    """
    Sets the QT_API environment variable by trying to import PyQtx or PySidex.

    The user can generically request `qt` or a specific Qt version, e.g. `qt6`.
    For a generic Qt request, we let the mechanism in IPython choose the best
    available version by leaving the `QT_API` environment variable blank.

    For specific versions, we check to see whether the PyQt or PySide
    implementations are present and set `QT_API` accordingly to indicate to
    IPython which version we want. If neither implementation is present, we
    leave the environment variable set so IPython will generate a helpful error
    message.

    Notes
    -----
    - If the environment variable is already set, it will be used unchanged,
      regardless of what the user requested.
    """
    qt_api = os.environ.get("QT_API", None)

    from IPython.external.qt_loaders import (
        QT_API_PYQT5,
        QT_API_PYQT6,
        QT_API_PYSIDE2,
        QT_API_PYSIDE6,
        loaded_api,
    )

    loaded = loaded_api()

    qt_env2gui = {
        QT_API_PYSIDE2: 'qt5',
        QT_API_PYQT5: 'qt5',
        QT_API_PYSIDE6: 'qt6',
        QT_API_PYQT6: 'qt6',
    }
    if loaded is not None and gui != 'qt' and qt_env2gui[loaded] != gui:
        print(f'Cannot switch Qt versions for this session; you must use {qt_env2gui[loaded]}.')
        return

    if qt_api is not None and gui != 'qt':
        if qt_env2gui[qt_api] != gui:
            print(
                f'Request for "{gui}" will be ignored because `QT_API` '
                f'environment variable is set to "{qt_api}"'
            )
            return
    else:
        if gui == 'qt5':
            try:
                import PyQt5  # noqa

                os.environ["QT_API"] = "pyqt5"
            except ImportError:
                try:
                    import PySide2  # noqa

                    os.environ["QT_API"] = "pyside2"
                except ImportError:
                    os.environ["QT_API"] = "pyqt5"
        elif gui == 'qt6':
            try:
                import PyQt6  # noqa

                os.environ["QT_API"] = "pyqt6"
            except ImportError:
                try:
                    import PySide6  # noqa

                    os.environ["QT_API"] = "pyside6"
                except ImportError:
                    os.environ["QT_API"] = "pyqt6"
        elif gui == 'qt':
            # Don't set QT_API; let IPython logic choose the version.
            if 'QT_API' in os.environ:
                del os.environ['QT_API']
        else:
            print(f'Unrecognized Qt version: {gui}. Should be "qt5", "qt6", or "qt".')
            return

    # Do the actual import now that the environment variable is set to make sure it works.
    try:
        from IPython.external.qt_for_kernel import QtCore, QtGui  # noqa
    except Exception as e:
        # Clear the environment variable for the next attempt.
        if 'QT_API' in os.environ:
            del os.environ["QT_API"]
            print(f"QT_API couldn't be set due to error {e}")
        return


def make_qt_app_for_kernel(gui, kernel):
    """Sets the `QT_API` environment variable if it isn't already set."""
    if hasattr(kernel, 'app'):
        # Kernel is already running a Qt event loop, so there's no need to
        # create another app for it.
        return

    set_qt_api_env_from_gui(gui)

    # This import is guaranteed to work now:
    from IPython.external.qt_for_kernel import QtCore
    from IPython.lib.guisupport import get_app_qt4

    kernel.app = get_app_qt4([" "])
    kernel.app.qt_event_loop = QtCore.QEventLoop(kernel.app)


def enable_gui(gui, kernel=None):
    """Enable integration with a given GUI"""
    if gui not in loop_map:
        e = f"Invalid GUI request {gui!r}, valid ones are:{loop_map.keys()}"
        raise ValueError(e)
    if kernel is None:
        if Application.initialized():
            kernel = getattr(Application.instance(), "kernel", None)
        if kernel is None:
            msg = (
                "You didn't specify a kernel,"
                " and no IPython Application with a kernel appears to be running."
            )
            raise RuntimeError(msg)
    if gui is None:
        # User wants to turn off integration; clear any evidence if Qt was the last one.
        if hasattr(kernel, 'app'):
            delattr(kernel, 'app')
    else:
        if gui.startswith('qt'):
            # Prepare the kernel here so any exceptions are displayed in the client.
            make_qt_app_for_kernel(gui, kernel)

    loop = loop_map[gui]
    if loop and kernel.eventloop is not None and kernel.eventloop is not loop:
        msg = "Cannot activate multiple GUI eventloops"
        raise RuntimeError(msg)
    kernel.eventloop = loop
    # We set `eventloop`; the function the user chose is executed in `Kernel.enter_eventloop`, thus
    # any exceptions raised during the event loop will not be shown in the client.
