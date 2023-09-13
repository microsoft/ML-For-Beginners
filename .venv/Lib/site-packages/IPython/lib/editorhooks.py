""" 'editor' hooks for common editors that work well with ipython

They should honor the line number argument, at least.

Contributions are *very* welcome.
"""

import os
import shlex
import subprocess
import sys

from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat


def install_editor(template, wait=False):
    """Installs the editor that is called by IPython for the %edit magic.

    This overrides the default editor, which is generally set by your EDITOR
    environment variable or is notepad (windows) or vi (linux). By supplying a
    template string `run_template`, you can control how the editor is invoked
    by IPython -- (e.g. the format in which it accepts command line options)

    Parameters
    ----------
    template : basestring
        run_template acts as a template for how your editor is invoked by
        the shell. It should contain '{filename}', which will be replaced on
        invocation with the file name, and '{line}', $line by line number
        (or 0) to invoke the file with.
    wait : bool
        If `wait` is true, wait until the user presses enter before returning,
        to facilitate non-blocking editors that exit immediately after
        the call.
    """

    # not all editors support $line, so we'll leave out this check
    # for substitution in ['$file', '$line']:
    #    if not substitution in run_template:
    #        raise ValueError(('run_template should contain %s'
    #        ' for string substitution. You supplied "%s"' % (substitution,
    #            run_template)))

    def call_editor(self, filename, line=0):
        if line is None:
            line = 0
        cmd = template.format(filename=shlex.quote(filename), line=line)
        print(">", cmd)
        # shlex.quote doesn't work right on Windows, but it does after splitting
        if sys.platform.startswith('win'):
            cmd = shlex.split(cmd)
        proc = subprocess.Popen(cmd, shell=True)
        if proc.wait() != 0:
            raise TryNext()
        if wait:
            py3compat.input("Press Enter when done editing:")

    get_ipython().set_hook('editor', call_editor)
    get_ipython().editor = template


# in these, exe is always the path/name of the executable. Useful
# if you don't have the editor directory in your path
def komodo(exe=u'komodo'):
    """ Activestate Komodo [Edit] """
    install_editor(exe + u' -l {line} {filename}', wait=True)


def scite(exe=u"scite"):
    """ SciTE or Sc1 """
    install_editor(exe + u' {filename} -goto:{line}')


def notepadplusplus(exe=u'notepad++'):
    """ Notepad++ http://notepad-plus.sourceforge.net """
    install_editor(exe + u' -n{line} {filename}')


def jed(exe=u'jed'):
    """ JED, the lightweight emacsish editor """
    install_editor(exe + u' +{line} {filename}')


def idle(exe=u'idle'):
    """ Idle, the editor bundled with python

    Parameters
    ----------
    exe : str, None
        If none, should be pretty smart about finding the executable.
    """
    if exe is None:
        import idlelib
        p = os.path.dirname(idlelib.__filename__)
        # i'm not sure if this actually works. Is this idle.py script
        # guaranteed to be executable?
        exe = os.path.join(p, 'idle.py')
    install_editor(exe + u' {filename}')


def mate(exe=u'mate'):
    """ TextMate, the missing editor"""
    # wait=True is not required since we're using the -w flag to mate
    install_editor(exe + u' -w -l {line} {filename}')


# ##########################################
# these are untested, report any problems
# ##########################################


def emacs(exe=u'emacs'):
    install_editor(exe + u' +{line} {filename}')


def gnuclient(exe=u'gnuclient'):
    install_editor(exe + u' -nw +{line} {filename}')


def crimson_editor(exe=u'cedt.exe'):
    install_editor(exe + u' /L:{line} {filename}')


def kate(exe=u'kate'):
    install_editor(exe + u' -u -l {line} {filename}')
