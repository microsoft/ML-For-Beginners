import subprocess
import sys
import json
from _pydev_bundle import pydev_log
import os
import io


def gen_debug_info():
    from _pydevd_bundle.pydevd_constants import DebugInfoHolder
    dct = {}
    for name in (
        'PYDEVD_DEBUG_FILE',
        'DEBUG_TRACE_LEVEL',
        ):
        dct[name] = getattr(DebugInfoHolder, name)

    return dct


if __name__ == "__main__":
    if '-print-debug' in sys.argv:
        info = gen_debug_info()  # break on 2nd process
        pydev_log.info('Something in print-debug')

        print('>>> print-debug pid: %s' % os.getpid())
        print(json.dumps(info))

    else:
        # Note: when running tests we usually have logging setup,
        # so, we create a context so that our changes are restored
        # when it finishes (as the `log_to` function will just reset
        # whatever is there).
        s = io.StringIO()
        with pydev_log.log_context(trace_level=3, stream=s):
            target_log_file = os.getenv('TARGET_LOG_FILE')

            pydev_log.log_to(target_log_file, 1)
            new_debug_info = gen_debug_info()
            subprocess_pid = None
            with subprocess.Popen(
                    [sys.executable, __file__, '-print-debug'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ) as process:
                subprocess_pid = process.pid
                stdout, stderr = process.communicate(input)

            output = stdout.decode('utf-8')
            pydev_log.info('Something in initial')

        log_contents = open(pydev_log._compute_filename_with_pid(target_log_file)).read()
        assert 'Something in initial' in log_contents, 'Did not find "Something in initial" in %s' % (log_contents,)

        log_contents = open(pydev_log._compute_filename_with_pid(target_log_file, pid=subprocess_pid)).read()
        assert 'Something in print-debug' in log_contents, 'Did not find "Something in print-debug" in %s' % (log_contents,)

        output = ''.join(output.splitlines(keepends=True)[1:])  # Remove the first line
        loaded_debug_info = json.loads(output)
        assert loaded_debug_info == new_debug_info, 'Expected %s. Found: %s' % (new_debug_info, loaded_debug_info)
        print('>>> Initial pid: %s' % os.getpid())
        print(output)
        print('TEST SUCEEDED')
