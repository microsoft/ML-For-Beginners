"""Implementation of magic functions that control various automatic behaviors.
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
from IPython.core.magic import Bunch, Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from logging import error

#-----------------------------------------------------------------------------
# Magic implementation classes
#-----------------------------------------------------------------------------

@magics_class
class AutoMagics(Magics):
    """Magics that control various autoX behaviors."""

    def __init__(self, shell):
        super(AutoMagics, self).__init__(shell)
        # namespace for holding state we may need
        self._magic_state = Bunch()

    @line_magic
    def automagic(self, parameter_s=''):
        """Make magic functions callable without having to type the initial %.

        Without arguments toggles on/off (when off, you must call it as
        %automagic, of course).  With arguments it sets the value, and you can
        use any of (case insensitive):

         - on, 1, True: to activate

         - off, 0, False: to deactivate.

        Note that magic functions have lowest priority, so if there's a
        variable whose name collides with that of a magic fn, automagic won't
        work for that function (you get the variable instead). However, if you
        delete the variable (del var), the previously shadowed magic function
        becomes visible to automagic again."""

        arg = parameter_s.lower()
        mman = self.shell.magics_manager
        if arg in ('on', '1', 'true'):
            val = True
        elif arg in ('off', '0', 'false'):
            val = False
        else:
            val = not mman.auto_magic
        mman.auto_magic = val
        print('\n' + self.shell.magics_manager.auto_status())

    @skip_doctest
    @line_magic
    def autocall(self, parameter_s=''):
        """Make functions callable without having to type parentheses.

        Usage:

           %autocall [mode]

        The mode can be one of: 0->Off, 1->Smart, 2->Full.  If not given, the
        value is toggled on and off (remembering the previous state).

        In more detail, these values mean:

        0 -> fully disabled

        1 -> active, but do not apply if there are no arguments on the line.

        In this mode, you get::

          In [1]: callable
          Out[1]: <built-in function callable>

          In [2]: callable 'hello'
          ------> callable('hello')
          Out[2]: False

        2 -> Active always.  Even if no arguments are present, the callable
        object is called::

          In [2]: float
          ------> float()
          Out[2]: 0.0

        Note that even with autocall off, you can still use '/' at the start of
        a line to treat the first argument on the command line as a function
        and add parentheses to it::

          In [8]: /str 43
          ------> str(43)
          Out[8]: '43'

        # all-random (note for auto-testing)
        """

        valid_modes = {
            0: "Off",
            1: "Smart",
            2: "Full",
        }

        def errorMessage() -> str:
            error = "Valid modes: "
            for k, v in valid_modes.items():
                error += str(k) + "->" + v + ", "
            error = error[:-2]  # remove tailing `, ` after last element
            return error

        if parameter_s:
            if not parameter_s in map(str, valid_modes.keys()):
                error(errorMessage())
                return
            arg = int(parameter_s)
        else:
            arg = 'toggle'

        if not arg in (*list(valid_modes.keys()), "toggle"):
            error(errorMessage())
            return

        if arg in (valid_modes.keys()):
            self.shell.autocall = arg
        else: # toggle
            if self.shell.autocall:
                self._magic_state.autocall_save = self.shell.autocall
                self.shell.autocall = 0
            else:
                try:
                    self.shell.autocall = self._magic_state.autocall_save
                except AttributeError:
                    self.shell.autocall = self._magic_state.autocall_save = 1

        print("Automatic calling is:", list(valid_modes.values())[self.shell.autocall])
