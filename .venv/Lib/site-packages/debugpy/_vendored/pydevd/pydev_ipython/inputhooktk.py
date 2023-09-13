# encoding: utf-8
# Unlike what IPython does, we need to have an explicit inputhook because tkinter handles
# input hook in the C Source code

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from pydev_ipython.inputhook import stdin_ready

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

TCL_DONT_WAIT = 1 << 1

def create_inputhook_tk(app):
    def inputhook_tk():
        while app.dooneevent(TCL_DONT_WAIT) == 1:
            if stdin_ready():
                break
        return 0
    return inputhook_tk
