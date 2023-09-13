"""Implementation of magic functions for the extension machinery.
"""
#-----------------------------------------------------------------------------
#  Copyright (c) 2012 The IPython Development Team.
#
#  Distributed under the terms of the Modified BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------


# Our own packages
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic

#-----------------------------------------------------------------------------
# Magic implementation classes
#-----------------------------------------------------------------------------

@magics_class
class ExtensionMagics(Magics):
    """Magics to manage the IPython extensions system."""

    @line_magic
    def load_ext(self, module_str):
        """Load an IPython extension by its module name."""
        if not module_str:
            raise UsageError('Missing module name.')
        res = self.shell.extension_manager.load_extension(module_str)
        
        if res == 'already loaded':
            print("The %s extension is already loaded. To reload it, use:" % module_str)
            print("  %reload_ext", module_str)
        elif res == 'no load function':
            print("The %s module is not an IPython extension." % module_str)

    @line_magic
    def unload_ext(self, module_str):
        """Unload an IPython extension by its module name.

        Not all extensions can be unloaded, only those which define an
        ``unload_ipython_extension`` function.
        """
        if not module_str:
            raise UsageError('Missing module name.')
        
        res = self.shell.extension_manager.unload_extension(module_str)
        
        if res == 'no unload function':
            print("The %s extension doesn't define how to unload it." % module_str)
        elif res == "not loaded":
            print("The %s extension is not loaded." % module_str)

    @line_magic
    def reload_ext(self, module_str):
        """Reload an IPython extension by its module name."""
        if not module_str:
            raise UsageError('Missing module name.')
        self.shell.extension_manager.reload_extension(module_str)
