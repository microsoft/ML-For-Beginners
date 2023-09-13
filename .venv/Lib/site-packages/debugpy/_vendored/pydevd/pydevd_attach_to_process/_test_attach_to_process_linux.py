'''
This module is just for testing concepts. It should be erased later on.

Experiments:

// gdb -p 4957
// call dlopen("/home/fabioz/Desktop/dev/PyDev.Debugger/pydevd_attach_to_process/linux/attach_linux.so", 2)
// call dlsym($1, "hello")
// call hello()


// call open("/home/fabioz/Desktop/dev/PyDev.Debugger/pydevd_attach_to_process/linux/attach_linux.so", 2)
// call mmap(0, 6672, 1 | 2 | 4, 1, 3 , 0)
// add-symbol-file
// cat /proc/pid/maps

// call dlopen("/home/fabioz/Desktop/dev/PyDev.Debugger/pydevd_attach_to_process/linux/attach_linux.so", 1|8)
// call dlsym($1, "hello")
// call hello()
'''

import subprocess
import sys
import os
import time

if __name__ == '__main__':

    linux_dir = os.path.join(os.path.dirname(__file__), 'linux')
    os.chdir(linux_dir)
    so_location = os.path.join(linux_dir, 'attach_linux.so')
    try:
        os.remove(so_location)
    except:
        pass
    subprocess.call('g++ -shared -o attach_linux.so -fPIC -nostartfiles attach_linux.c'.split())
    print('Finished compiling')
    assert os.path.exists('/home/fabioz/Desktop/dev/PyDev.Debugger/pydevd_attach_to_process/linux/attach_linux.so')
    os.chdir(os.path.dirname(linux_dir))
#     import attach_pydevd
#     attach_pydevd.main(attach_pydevd.process_command_line(['--pid', str(p.pid)]))
    p = subprocess.Popen([sys.executable, '-u', '_always_live_program.py'])
    print('Size of file: %s' % (os.stat(so_location).st_size))

    # (gdb) set architecture
    # Requires an argument. Valid arguments are i386, i386:x86-64, i386:x64-32, i8086, i386:intel, i386:x86-64:intel, i386:x64-32:intel, i386:nacl, i386:x86-64:nacl, i386:x64-32:nacl, auto.

    cmd = [
        'gdb',
        '--pid',
        str(p.pid),
        '--batch',
    ]

    arch = 'i386:x86-64'
    if arch:
        cmd.extend(["--eval-command='set architecture %s'" % arch])

    cmd.extend([
        "--eval-command='call dlopen(\"/home/fabioz/Desktop/dev/PyDev.Debugger/pydevd_attach_to_process/linux/attach_linux.so\", 2)'",
        "--eval-command='call (int)DoAttach(1, \"print(\\\"check11111check\\\")\", 0)'",
        # "--eval-command='call (int)SetSysTraceFunc(1, 0)'", -- never call this way, always use "--command='...gdb_threads_settrace.py'",
        # So that threads are all stopped!
    ])

    print(' '.join(cmd))
    time.sleep(.5)
    env = os.environ.copy()
    env.pop('PYTHONIOENCODING', None)
    env.pop('PYTHONPATH', None)
    p2 = subprocess.call(' '.join(cmd), env=env, shell=True)

    time.sleep(1)
    p.kill()
