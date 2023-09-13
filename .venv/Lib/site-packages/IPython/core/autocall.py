# encoding: utf-8
"""
Autocall capabilities for IPython.core.

Authors:

* Brian Granger
* Fernando Perez
* Thomas Kluyver

Notes
-----
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


#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class IPyAutocall(object):
    """ Instances of this class are always autocalled
    
    This happens regardless of 'autocall' variable state. Use this to
    develop macro-like mechanisms.
    """
    _ip = None
    rewrite = True
    def __init__(self, ip=None):
        self._ip = ip
    
    def set_ip(self, ip):
        """Will be used to set _ip point to current ipython instance b/f call

        Override this method if you don't want this to happen.

        """
        self._ip = ip


class ExitAutocall(IPyAutocall):
    """An autocallable object which will be added to the user namespace so that
    exit, exit(), quit or quit() are all valid ways to close the shell."""
    rewrite = False
    
    def __call__(self):
        self._ip.ask_exit()
        
class ZMQExitAutocall(ExitAutocall):
    """Exit IPython. Autocallable, so it needn't be explicitly called.
    
    Parameters
    ----------
    keep_kernel : bool
      If True, leave the kernel alive. Otherwise, tell the kernel to exit too
      (default).
    """
    def __call__(self, keep_kernel=False):
        self._ip.keepkernel_on_exit = keep_kernel
        self._ip.ask_exit()
