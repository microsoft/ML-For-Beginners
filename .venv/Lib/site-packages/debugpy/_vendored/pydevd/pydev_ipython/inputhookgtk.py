# encoding: utf-8
"""
Enable pygtk to be used interacive by setting PyOS_InputHook.

Authors: Brian Granger
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

import gtk, gobject  # @UnresolvedImport

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


def _main_quit(*args, **kwargs):
    gtk.main_quit()
    return False

def create_inputhook_gtk(stdin_file):
    def inputhook_gtk():
        gobject.io_add_watch(stdin_file, gobject.IO_IN, _main_quit)
        gtk.main()
        return 0
    return inputhook_gtk

