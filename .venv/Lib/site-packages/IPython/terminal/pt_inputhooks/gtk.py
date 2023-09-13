# Code borrowed from python-prompt-toolkit examples
# https://github.com/jonathanslenders/python-prompt-toolkit/blob/77cdcfbc7f4b4c34a9d2f9a34d422d7152f16209/examples/inputhook.py

# Copyright (c) 2014, Jonathan Slenders
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# * Neither the name of the {organization} nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
PyGTK input hook for prompt_toolkit.

Listens on the pipe prompt_toolkit sets up for a notification that it should
return control to the terminal event loop.
"""

import gtk, gobject

# Enable threading in GTK. (Otherwise, GTK will keep the GIL.)
gtk.gdk.threads_init()


def inputhook(context):
    """
    When the eventloop of prompt-toolkit is idle, call this inputhook.

    This will run the GTK main loop until the file descriptor
    `context.fileno()` becomes ready.

    :param context: An `InputHookContext` instance.
    """

    def _main_quit(*a, **kw):
        gtk.main_quit()
        return False

    gobject.io_add_watch(context.fileno(), gobject.IO_IN, _main_quit)
    gtk.main()
