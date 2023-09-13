# encoding: utf-8
"""
Paging capabilities for IPython.core

Notes
-----

For now this uses IPython hooks, so it can't be in IPython.utils.  If we can get
rid of that dependency, we could move it there.
-----
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


import os
import io
import re
import sys
import tempfile
import subprocess

from io import UnsupportedOperation
from pathlib import Path

from IPython import get_ipython
from IPython.display import display
from IPython.core.error import TryNext
from IPython.utils.data import chop
from IPython.utils.process import system
from IPython.utils.terminal import get_terminal_size
from IPython.utils import py3compat


def display_page(strng, start=0, screen_lines=25):
    """Just display, no paging. screen_lines is ignored."""
    if isinstance(strng, dict):
        data = strng
    else:
        if start:
            strng = u'\n'.join(strng.splitlines()[start:])
        data = { 'text/plain': strng }
    display(data, raw=True)


def as_hook(page_func):
    """Wrap a pager func to strip the `self` arg

    so it can be called as a hook.
    """
    return lambda self, *args, **kwargs: page_func(*args, **kwargs)


esc_re = re.compile(r"(\x1b[^m]+m)")

def page_dumb(strng, start=0, screen_lines=25):
    """Very dumb 'pager' in Python, for when nothing else works.

    Only moves forward, same interface as page(), except for pager_cmd and
    mode.
    """
    if isinstance(strng, dict):
        strng = strng.get('text/plain', '')
    out_ln  = strng.splitlines()[start:]
    screens = chop(out_ln,screen_lines-1)
    if len(screens) == 1:
        print(os.linesep.join(screens[0]))
    else:
        last_escape = ""
        for scr in screens[0:-1]:
            hunk = os.linesep.join(scr)
            print(last_escape + hunk)
            if not page_more():
                return
            esc_list = esc_re.findall(hunk)
            if len(esc_list) > 0:
                last_escape = esc_list[-1]
        print(last_escape + os.linesep.join(screens[-1]))

def _detect_screen_size(screen_lines_def):
    """Attempt to work out the number of lines on the screen.

    This is called by page(). It can raise an error (e.g. when run in the
    test suite), so it's separated out so it can easily be called in a try block.
    """
    TERM = os.environ.get('TERM',None)
    if not((TERM=='xterm' or TERM=='xterm-color') and sys.platform != 'sunos5'):
        # curses causes problems on many terminals other than xterm, and
        # some termios calls lock up on Sun OS5.
        return screen_lines_def
    
    try:
        import termios
        import curses
    except ImportError:
        return screen_lines_def
    
    # There is a bug in curses, where *sometimes* it fails to properly
    # initialize, and then after the endwin() call is made, the
    # terminal is left in an unusable state.  Rather than trying to
    # check every time for this (by requesting and comparing termios
    # flags each time), we just save the initial terminal state and
    # unconditionally reset it every time.  It's cheaper than making
    # the checks.
    try:
        term_flags = termios.tcgetattr(sys.stdout)
    except termios.error as err:
        # can fail on Linux 2.6, pager_page will catch the TypeError
        raise TypeError('termios error: {0}'.format(err)) from err

    try:
        scr = curses.initscr()
    except AttributeError:
        # Curses on Solaris may not be complete, so we can't use it there
        return screen_lines_def
    
    screen_lines_real,screen_cols = scr.getmaxyx()
    curses.endwin()

    # Restore terminal state in case endwin() didn't.
    termios.tcsetattr(sys.stdout,termios.TCSANOW,term_flags)
    # Now we have what we needed: the screen size in rows/columns
    return screen_lines_real
    #print '***Screen size:',screen_lines_real,'lines x',\
    #screen_cols,'columns.' # dbg

def pager_page(strng, start=0, screen_lines=0, pager_cmd=None):
    """Display a string, piping through a pager after a certain length.

    strng can be a mime-bundle dict, supplying multiple representations,
    keyed by mime-type.

    The screen_lines parameter specifies the number of *usable* lines of your
    terminal screen (total lines minus lines you need to reserve to show other
    information).

    If you set screen_lines to a number <=0, page() will try to auto-determine
    your screen size and will only use up to (screen_size+screen_lines) for
    printing, paging after that. That is, if you want auto-detection but need
    to reserve the bottom 3 lines of the screen, use screen_lines = -3, and for
    auto-detection without any lines reserved simply use screen_lines = 0.

    If a string won't fit in the allowed lines, it is sent through the
    specified pager command. If none given, look for PAGER in the environment,
    and ultimately default to less.

    If no system pager works, the string is sent through a 'dumb pager'
    written in python, very simplistic.
    """
    
    # for compatibility with mime-bundle form:
    if isinstance(strng, dict):
        strng = strng['text/plain']

    # Ugly kludge, but calling curses.initscr() flat out crashes in emacs
    TERM = os.environ.get('TERM','dumb')
    if TERM in ['dumb','emacs'] and os.name != 'nt':
        print(strng)
        return
    # chop off the topmost part of the string we don't want to see
    str_lines = strng.splitlines()[start:]
    str_toprint = os.linesep.join(str_lines)
    num_newlines = len(str_lines)
    len_str = len(str_toprint)

    # Dumb heuristics to guesstimate number of on-screen lines the string
    # takes.  Very basic, but good enough for docstrings in reasonable
    # terminals. If someone later feels like refining it, it's not hard.
    numlines = max(num_newlines,int(len_str/80)+1)

    screen_lines_def = get_terminal_size()[1]

    # auto-determine screen size
    if screen_lines <= 0:
        try:
            screen_lines += _detect_screen_size(screen_lines_def)
        except (TypeError, UnsupportedOperation):
            print(str_toprint)
            return

    #print 'numlines',numlines,'screenlines',screen_lines  # dbg
    if numlines <= screen_lines :
        #print '*** normal print'  # dbg
        print(str_toprint)
    else:
        # Try to open pager and default to internal one if that fails.
        # All failure modes are tagged as 'retval=1', to match the return
        # value of a failed system command.  If any intermediate attempt
        # sets retval to 1, at the end we resort to our own page_dumb() pager.
        pager_cmd = get_pager_cmd(pager_cmd)
        pager_cmd += ' ' + get_pager_start(pager_cmd,start)
        if os.name == 'nt':
            if pager_cmd.startswith('type'):
                # The default WinXP 'type' command is failing on complex strings.
                retval = 1
            else:
                fd, tmpname = tempfile.mkstemp('.txt')
                tmppath = Path(tmpname)
                try:
                    os.close(fd)
                    with tmppath.open("wt", encoding="utf-8") as tmpfile:
                        tmpfile.write(strng)
                        cmd = "%s < %s" % (pager_cmd, tmppath)
                    # tmpfile needs to be closed for windows
                    if os.system(cmd):
                        retval = 1
                    else:
                        retval = None
                finally:
                    Path.unlink(tmppath)
        else:
            try:
                retval = None
                # Emulate os.popen, but redirect stderr
                proc = subprocess.Popen(
                    pager_cmd,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                pager = os._wrap_close(
                    io.TextIOWrapper(proc.stdin, encoding="utf-8"), proc
                )
                try:
                    pager_encoding = pager.encoding or sys.stdout.encoding
                    pager.write(strng)
                finally:
                    retval = pager.close()
            except IOError as msg:  # broken pipe when user quits
                if msg.args == (32, 'Broken pipe'):
                    retval = None
                else:
                    retval = 1
            except OSError:
                # Other strange problems, sometimes seen in Win2k/cygwin
                retval = 1
        if retval is not None:
            page_dumb(strng,screen_lines=screen_lines)


def page(data, start=0, screen_lines=0, pager_cmd=None):
    """Display content in a pager, piping through a pager after a certain length.

    data can be a mime-bundle dict, supplying multiple representations,
    keyed by mime-type, or text.

    Pager is dispatched via the `show_in_pager` IPython hook.
    If no hook is registered, `pager_page` will be used.
    """
    # Some routines may auto-compute start offsets incorrectly and pass a
    # negative value.  Offset to 0 for robustness.
    start = max(0, start)

    # first, try the hook
    ip = get_ipython()
    if ip:
        try:
            ip.hooks.show_in_pager(data, start=start, screen_lines=screen_lines)
            return
        except TryNext:
            pass
    
    # fallback on default pager
    return pager_page(data, start, screen_lines, pager_cmd)


def page_file(fname, start=0, pager_cmd=None):
    """Page a file, using an optional pager command and starting line.
    """

    pager_cmd = get_pager_cmd(pager_cmd)
    pager_cmd += ' ' + get_pager_start(pager_cmd,start)

    try:
        if os.environ['TERM'] in ['emacs','dumb']:
            raise EnvironmentError
        system(pager_cmd + ' ' + fname)
    except:
        try:
            if start > 0:
                start -= 1
            page(open(fname, encoding="utf-8").read(), start)
        except:
            print('Unable to show file',repr(fname))


def get_pager_cmd(pager_cmd=None):
    """Return a pager command.

    Makes some attempts at finding an OS-correct one.
    """
    if os.name == 'posix':
        default_pager_cmd = 'less -R'  # -R for color control sequences
    elif os.name in ['nt','dos']:
        default_pager_cmd = 'type'

    if pager_cmd is None:
        try:
            pager_cmd = os.environ['PAGER']
        except:
            pager_cmd = default_pager_cmd
    
    if pager_cmd == 'less' and '-r' not in os.environ.get('LESS', '').lower():
        pager_cmd += ' -R'
    
    return pager_cmd


def get_pager_start(pager, start):
    """Return the string for paging files with an offset.

    This is the '+N' argument which less and more (under Unix) accept.
    """

    if pager in ['less','more']:
        if start:
            start_string = '+' + str(start)
        else:
            start_string = ''
    else:
        start_string = ''
    return start_string


# (X)emacs on win32 doesn't like to be bypassed with msvcrt.getch()
if os.name == 'nt' and os.environ.get('TERM','dumb') != 'emacs':
    import msvcrt
    def page_more():
        """ Smart pausing between pages

        @return:    True if need print more lines, False if quit
        """
        sys.stdout.write('---Return to continue, q to quit--- ')
        ans = msvcrt.getwch()
        if ans in ("q", "Q"):
            result = False
        else:
            result = True
        sys.stdout.write("\b"*37 + " "*37 + "\b"*37)
        return result
else:
    def page_more():
        ans = py3compat.input('---Return to continue, q to quit--- ')
        if ans.lower().startswith('q'):
            return False
        else:
            return True
