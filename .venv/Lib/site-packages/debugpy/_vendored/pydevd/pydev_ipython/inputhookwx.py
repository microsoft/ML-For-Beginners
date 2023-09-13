# encoding: utf-8
"""
Enable wxPython to be used interacive by setting PyOS_InputHook.

Authors:  Robin Dunn, Brian Granger, Ondrej Certik
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
import signal
from _pydev_bundle._pydev_saved_modules import time
from timeit import default_timer as clock
import wx

from pydev_ipython.inputhook import stdin_ready


#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

def inputhook_wx1():
    """Run the wx event loop by processing pending events only.

    This approach seems to work, but its performance is not great as it
    relies on having PyOS_InputHook called regularly.
    """
    try:
        app = wx.GetApp()  # @UndefinedVariable
        if app is not None:
            assert wx.Thread_IsMain()  # @UndefinedVariable

            # Make a temporary event loop and process system events until
            # there are no more waiting, then allow idle events (which
            # will also deal with pending or posted wx events.)
            evtloop = wx.EventLoop()  # @UndefinedVariable
            ea = wx.EventLoopActivator(evtloop)  # @UndefinedVariable
            while evtloop.Pending():
                evtloop.Dispatch()
            app.ProcessIdle()
            del ea
    except KeyboardInterrupt:
        pass
    return 0

class EventLoopTimer(wx.Timer):  # @UndefinedVariable

    def __init__(self, func):
        self.func = func
        wx.Timer.__init__(self)  # @UndefinedVariable

    def Notify(self):
        self.func()

class EventLoopRunner(object):

    def Run(self, time):
        self.evtloop = wx.EventLoop()  # @UndefinedVariable
        self.timer = EventLoopTimer(self.check_stdin)
        self.timer.Start(time)
        self.evtloop.Run()

    def check_stdin(self):
        if stdin_ready():
            self.timer.Stop()
            self.evtloop.Exit()

def inputhook_wx2():
    """Run the wx event loop, polling for stdin.

    This version runs the wx eventloop for an undetermined amount of time,
    during which it periodically checks to see if anything is ready on
    stdin.  If anything is ready on stdin, the event loop exits.

    The argument to elr.Run controls how often the event loop looks at stdin.
    This determines the responsiveness at the keyboard.  A setting of 1000
    enables a user to type at most 1 char per second.  I have found that a
    setting of 10 gives good keyboard response.  We can shorten it further,
    but eventually performance would suffer from calling select/kbhit too
    often.
    """
    try:
        app = wx.GetApp()  # @UndefinedVariable
        if app is not None:
            assert wx.Thread_IsMain()  # @UndefinedVariable
            elr = EventLoopRunner()
            # As this time is made shorter, keyboard response improves, but idle
            # CPU load goes up.  10 ms seems like a good compromise.
            elr.Run(time=10)  # CHANGE time here to control polling interval
    except KeyboardInterrupt:
        pass
    return 0

def inputhook_wx3():
    """Run the wx event loop by processing pending events only.

    This is like inputhook_wx1, but it keeps processing pending events
    until stdin is ready.  After processing all pending events, a call to
    time.sleep is inserted.  This is needed, otherwise, CPU usage is at 100%.
    This sleep time should be tuned though for best performance.
    """
    # We need to protect against a user pressing Control-C when IPython is
    # idle and this is running. We trap KeyboardInterrupt and pass.
    try:
        app = wx.GetApp()  # @UndefinedVariable
        if app is not None:
            if hasattr(wx, 'IsMainThread'):
                assert wx.IsMainThread()  # @UndefinedVariable
            else:
                assert wx.Thread_IsMain()  # @UndefinedVariable

            # The import of wx on Linux sets the handler for signal.SIGINT
            # to 0.  This is a bug in wx or gtk.  We fix by just setting it
            # back to the Python default.
            if not callable(signal.getsignal(signal.SIGINT)):
                signal.signal(signal.SIGINT, signal.default_int_handler)

            evtloop = wx.EventLoop()  # @UndefinedVariable
            ea = wx.EventLoopActivator(evtloop)  # @UndefinedVariable
            t = clock()
            while not stdin_ready():
                while evtloop.Pending():
                    t = clock()
                    evtloop.Dispatch()
                app.ProcessIdle()
                # We need to sleep at this point to keep the idle CPU load
                # low.  However, if sleep to long, GUI response is poor.  As
                # a compromise, we watch how often GUI events are being processed
                # and switch between a short and long sleep time.  Here are some
                # stats useful in helping to tune this.
                # time    CPU load
                # 0.001   13%
                # 0.005   3%
                # 0.01    1.5%
                # 0.05    0.5%
                used_time = clock() - t
                if used_time > 10.0:
                    # print 'Sleep for 1 s'  # dbg
                    time.sleep(1.0)
                elif used_time > 0.1:
                    # Few GUI events coming in, so we can sleep longer
                    # print 'Sleep for 0.05 s'  # dbg
                    time.sleep(0.05)
                else:
                    # Many GUI events coming in, so sleep only very little
                    time.sleep(0.001)
            del ea
    except KeyboardInterrupt:
        pass
    return 0

if sys.platform == 'darwin':
    # On OSX, evtloop.Pending() always returns True, regardless of there being
    # any events pending. As such we can't use implementations 1 or 3 of the
    # inputhook as those depend on a pending/dispatch loop.
    inputhook_wx = inputhook_wx2
else:
    # This is our default implementation
    inputhook_wx = inputhook_wx3
