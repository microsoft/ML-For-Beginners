import os
import sys
import subprocess


def breaknow():
    print('break here')


if '--fork-in-subprocess' in sys.argv:
    popen = None
    if sys.platform == 'win32':
        popen = subprocess.Popen([sys.executable, __file__, '--forked'])
        pid = popen.pid
    else:
        pid = os.fork()
    try:
        ppid = os.getppid()
    except:
        ppid = '<unknown>'
    print('currently in pid: %s, ppid: %s' % (os.getpid(), ppid))
    print('os.fork returned', pid)
    breaknow()
    # i.e.: wait so that we check for the retcode so that we don't get a traceback such as the one
    # below (as that code in __del__ will only be called if returncode is None).
    # Traceback (most recent call last):
    # File "C:\hostedtoolcache\windows\Python\3.9.13\x64\lib\subprocess.py", line 1055, in __del__
    # self._internal_poll(_deadstate=_maxsize)
    # File "C:\hostedtoolcache\windows\Python\3.9.13\x64\lib\subprocess.py", line 1457, in _internal_poll
    # if _WaitForSingleObject(self._handle, 0) == _WAIT_OBJECT_0:
    # OSError: [WinError 6] The handle is invalid
    if popen is not None:
        popen.wait(20)

elif '--forked' in sys.argv:
    try:
        ppid = os.getppid()
    except:
        ppid = '<unknown>'
    print('currently in pid: %s, ppid: %s' % (os.getpid(), ppid))
    breaknow()

elif '--fork-in-subprocess' not in sys.argv:
    out = subprocess.check_output([sys.executable, __file__, '--fork-in-subprocess'])
    breaknow()
    print('\n\nin pid %s, output from subprocess.run:\n%s' % (os.getpid(), out.decode('utf-8')))
    print('TEST SUCEEDED!')
