import time
import subprocess
import sys
import os

if __name__ == '__main__':
    if 'launch-subprocesses' in sys.argv:
        n = int(sys.argv[-1])
        if n != 0:
            subprocess.Popen([sys.executable, __file__, 'launch-subprocesses', str(n - 1)])
        if hasattr(os, 'getppid'):
            print('%screated %s (child of %s)' % ('\t' * (4 - n), os.getpid(), os.getppid()))
        else:
            print('%screated %s' % ('\t' * (4 - n), os.getpid()))

    elif 'check-subprocesses' in sys.argv or 'check-subprocesses-ignore-pid' in sys.argv:
        # Recursively create a process tree such as:
        # - parent (this process)
        #    - p3
        #      - p2
        #        - p1
        #          - p0
        #    - p3
        #      - p2
        #        - p1
        #          - p0
        p0 = subprocess.Popen([sys.executable, __file__, 'launch-subprocesses', '3'])
        p1 = subprocess.Popen([sys.executable, __file__, 'launch-subprocesses', '3'])

        if 'check-subprocesses-ignore-pid' in sys.argv:
            import pydevd
            pydevd.add_dont_terminate_child_pid(p0.pid)

        print('created', os.getpid())

    while True:
        time.sleep(.1)
