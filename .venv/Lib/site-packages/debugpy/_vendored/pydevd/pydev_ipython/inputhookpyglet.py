# encoding: utf-8
"""
Enable pyglet to be used interacive by setting PyOS_InputHook.

Authors
-------

* Nicolas P. Rougier
* Fernando Perez
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

import os
import sys
from _pydev_bundle._pydev_saved_modules import time
from timeit import default_timer as clock
import pyglet  # @UnresolvedImport
from pydev_ipython.inputhook import stdin_ready


# On linux only, window.flip() has a bug that causes an AttributeError on
# window close.  For details, see:
# http://groups.google.com/group/pyglet-users/browse_thread/thread/47c1aab9aa4a3d23/c22f9e819826799e?#c22f9e819826799e

if sys.platform.startswith('linux'):
    def flip(window):
        try:
            window.flip()
        except AttributeError:
            pass
else:
    def flip(window):
        window.flip()

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

def inputhook_pyglet():
    """Run the pyglet event loop by processing pending events only.

    This keeps processing pending events until stdin is ready.  After
    processing all pending events, a call to time.sleep is inserted.  This is
    needed, otherwise, CPU usage is at 100%.  This sleep time should be tuned
    though for best performance.
    """
    # We need to protect against a user pressing Control-C when IPython is
    # idle and this is running. We trap KeyboardInterrupt and pass.
    try:
        t = clock()
        while not stdin_ready():
            pyglet.clock.tick()
            for window in pyglet.app.windows:
                window.switch_to()
                window.dispatch_events()
                window.dispatch_event('on_draw')
                flip(window)

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
    except KeyboardInterrupt:
        pass
    return 0
