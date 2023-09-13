# coding: utf-8
"""
GLUT Inputhook support functions
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

# GLUT is quite an old library and it is difficult to ensure proper
# integration within IPython since original GLUT does not allow to handle
# events one by one. Instead, it requires for the mainloop to be entered
# and never returned (there is not even a function to exit he
# mainloop). Fortunately, there are alternatives such as freeglut
# (available for linux and windows) and the OSX implementation gives
# access to a glutCheckLoop() function that blocks itself until a new
# event is received. This means we have to setup the idle callback to
# ensure we got at least one event that will unblock the function.
#
# Furthermore, it is not possible to install these handlers without a window
# being first created. We choose to make this window invisible. This means that
# display mode options are set at this level and user won't be able to change
# them later without modifying the code. This should probably be made available
# via IPython options system.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import os
import sys
from _pydev_bundle._pydev_saved_modules import time
import signal
import OpenGL.GLUT as glut  # @UnresolvedImport
import OpenGL.platform as platform  # @UnresolvedImport
from timeit import default_timer as clock
from pydev_ipython.inputhook import stdin_ready

#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

# Frame per second : 60
# Should probably be an IPython option
glut_fps = 60

# Display mode : double buffeed + rgba + depth
# Should probably be an IPython option
glut_display_mode = (glut.GLUT_DOUBLE |
                     glut.GLUT_RGBA |
                     glut.GLUT_DEPTH)

glutMainLoopEvent = None
if sys.platform == 'darwin':
    try:
        glutCheckLoop = platform.createBaseFunction(
            'glutCheckLoop', dll=platform.GLUT, resultType=None,
            argTypes=[],
            doc='glutCheckLoop(  ) -> None',
            argNames=(),
            )
    except AttributeError:
        raise RuntimeError(
            '''Your glut implementation does not allow interactive sessions'''
            '''Consider installing freeglut.''')
    glutMainLoopEvent = glutCheckLoop
elif glut.HAVE_FREEGLUT:
    glutMainLoopEvent = glut.glutMainLoopEvent
else:
    raise RuntimeError(
        '''Your glut implementation does not allow interactive sessions. '''
        '''Consider installing freeglut.''')

#-----------------------------------------------------------------------------
# Callback functions
#-----------------------------------------------------------------------------


def glut_display():
    # Dummy display function
    pass


def glut_idle():
    # Dummy idle function
    pass


def glut_close():
    # Close function only hides the current window
    glut.glutHideWindow()
    glutMainLoopEvent()


def glut_int_handler(signum, frame):
    # Catch sigint and print the defautl message
    signal.signal(signal.SIGINT, signal.default_int_handler)
    print('\nKeyboardInterrupt')
    # Need to reprint the prompt at this stage


#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------
def inputhook_glut():
    """Run the pyglet event loop by processing pending events only.

    This keeps processing pending events until stdin is ready.  After
    processing all pending events, a call to time.sleep is inserted.  This is
    needed, otherwise, CPU usage is at 100%.  This sleep time should be tuned
    though for best performance.
    """
    # We need to protect against a user pressing Control-C when IPython is
    # idle and this is running. We trap KeyboardInterrupt and pass.

    signal.signal(signal.SIGINT, glut_int_handler)

    try:
        t = clock()

        # Make sure the default window is set after a window has been closed
        if glut.glutGetWindow() == 0:
            glut.glutSetWindow(1)
            glutMainLoopEvent()
            return 0

        while not stdin_ready():
            glutMainLoopEvent()
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
